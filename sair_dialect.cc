// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sair_dialect.h"

#include <algorithm>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_ops.h"
#include "sair_types.h"

using mlir::failed;
using mlir::succeeded;

namespace sair {

// Registers Sair types with MLIR.
SairDialect::SairDialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context,
                    TypeID::get<SairDialect>()) {
  addTypes<RangeType, ValueType>();
  addAttributes<DomainShapeAttr, MappingAttr, NamedMappingAttr, MappingDimExpr,
                MappingNoneExpr, MappingStripeExpr, MappingUnStripeExpr>();
  addOperations<
#define GET_OP_LIST
#include "sair_ops.cc.inc"
      >();

  register_ = mlir::StringAttr::get(context, "register");
  memory_ = mlir::StringAttr::get(context, "memory");
}

namespace {

// Parses the shape of a Sair domain, as expressed in Sair types. Returns
// nullptr in case of failure. Domains shapes are composed of a list of
// dimension types separated by 'x', with an optional dependency mapping for
// each dimensions, as shown below.
//
//   d0:range x d1:range(d0) x d2:range(d0, d1)
//
// Note that dependencies must be transitive. In example above, d2 must depend
// on d0 as it depends on d1 and d1 depends on d0.
DomainShapeAttr ParseDomainShape(mlir::DialectAsmParser &parser) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseRParen())) return nullptr;
    return DomainShapeAttr::get(context);
  }

  std::vector<DomainShapeDim> dimensions;
  do {
    // Parse the dimension name.
    std::string expected_name = "d" + std::to_string(dimensions.size());
    if (failed(parser.parseKeyword(expected_name))) return nullptr;

    if (failed(parser.parseColon()) ||
        failed(parser.parseKeyword(RangeType::Name()))) {
      return nullptr;
    }
    llvm::SMLoc loc = parser.getCurrentLocation();
    MappingAttr mapping = ParseOptionalMapping(parser, dimensions.size());
    if (mapping == nullptr) return nullptr;
    if (!mapping.IsFullySpecified()) {
      parser.emitError(loc) << "the mapping must map all dimensions";
      return nullptr;
    }

    std::vector<DomainShapeDim> arg_shape_dims;
    llvm::SmallBitVector seen_dimensions(dimensions.size());
    MappingAttr inversed_mapping = mapping.Inverse();
    for (MappingExpr expr : mapping) {
      llvm::SmallBitVector expr_dependencies =
          expr.DependencyMask(dimensions.size());

      llvm::SmallBitVector transitive_dependencies(dimensions.size());
      for (int dimension : expr_dependencies.set_bits()) {
        transitive_dependencies |= dimensions[dimension].DependencyMask();
      }

      if ((~seen_dimensions).anyCommon(transitive_dependencies)) {
        parser.emitError(loc) << "non-transitive dependency";
        return nullptr;
      }
      seen_dimensions |= expr_dependencies;
      arg_shape_dims.push_back(expr.AccessedShape(
          dimensions, inversed_mapping.ResizeUseDomain(arg_shape_dims.size())));
    }
    RangeType arg_type =
        RangeType::get(DomainShapeAttr::get(context, arg_shape_dims));
    dimensions.emplace_back(arg_type, mapping);
  } while (succeeded(parser.parseOptionalKeyword("x")));
  return DomainShapeAttr::get(context, dimensions);
}

NamedMappingAttr ParseNamedMapping(mlir::DialectAsmParser &parser) {
  if (mlir::failed(parser.parseLSquare())) return nullptr;
  llvm::SmallVector<mlir::StringAttr, 4> names;

  if (mlir::failed(parser.parseOptionalRSquare())) {
    do {
      std::string expected_name = "d" + std::to_string(names.size());
      mlir::StringAttr name;
      if (parser.parseKeyword(expected_name) || parser.parseColon() ||
          parser.parseAttribute(name)) {
        return nullptr;
      }
      names.push_back(name);
    } while (mlir::succeeded(parser.parseOptionalComma()));
    if (mlir::failed(parser.parseRSquare())) return nullptr;
  }

  if (parser.parseArrow() || parser.parseLParen()) return nullptr;
  MappingAttr mapping;
  if (mlir::succeeded(parser.parseOptionalRParen())) {
    mapping =
        MappingAttr::get(parser.getBuilder().getContext(), names.size(), {});
  } else {
    mapping = ParseMapping(parser, names.size());
    if (mapping == nullptr || parser.parseRParen()) return nullptr;
  }

  return NamedMappingAttr::get(names, mapping);
}

}  // namespace

// Parses the Sair dialect type. Returns nullptr in case of failure.
mlir::Type sair::SairDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (failed(parser.parseKeyword(&keyword))) return nullptr;

  if (keyword == sair::RangeType::Name()) {
    // Parse a type of the form '!sair.range<d0:range x d1:range>'.
    DomainShapeAttr domain;
    if (succeeded(parser.parseOptionalLess())) {
      domain = ParseDomainShape(parser);
      if (!domain) return nullptr;
      if (failed(parser.parseOptionalGreater())) {
        parser.emitError(parser.getNameLoc(), "expected 'x' or '>'");
        return nullptr;
      }
    } else {
      domain = DomainShapeAttr::get(getContext());
    }
    return RangeType::get(domain);
  }

  if (keyword == sair::ValueType::Name()) {
    if (failed(parser.parseLess())) return nullptr;
    auto domain = ParseDomainShape(parser);
    if (!domain || failed(parser.parseComma())) return nullptr;
    mlir::Type element_type;
    if (failed(parser.parseType(element_type))) return nullptr;
    if (failed(parser.parseOptionalGreater())) {
      parser.emitError(parser.getNameLoc(), "expected 'x' or '>'");
      return nullptr;
    }
    return ValueType::get(domain, element_type);
  }

  parser.emitError(parser.getNameLoc(), "invalid sair type");
  return nullptr;
}

// Parses the Sair dialect attribute. If 'type' Returns nullptr in case of
// failure. 'type' is expected to be null as Sair attributes do not have a type.
mlir::Attribute sair::SairDialect::parseAttribute(
    mlir::DialectAsmParser &parser, Type type) const {
  if (type) {
    parser.emitError(parser.getNameLoc()) << "unexpected type";
    return nullptr;
  }

  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword) || parser.parseLess()) return nullptr;
  mlir::Attribute attribute;
  if (keyword == "shape") {
    attribute = ParseDomainShape(parser);
  } else if (keyword == "mapping") {
    int num_dimensions;
    if (failed(parser.parseInteger(num_dimensions))) return nullptr;
    if (succeeded(parser.parseOptionalColon())) {
      attribute = ParseMapping(parser, num_dimensions);
    } else {
      attribute = MappingAttr::get(getContext(), num_dimensions, {});
    }
  } else if (keyword == "named_mapping") {
    attribute = ParseNamedMapping(parser);
  } else if (keyword == "mapping_expr") {
    attribute = ParseMappingExpr(parser);
  } else {
    parser.emitError(parser.getNameLoc())
        << "unexpected Sair attribute '" << keyword << "'";
    return nullptr;
  }
  if (!attribute || parser.parseGreater()) return nullptr;
  return attribute;
}

namespace {

// Prints an mapping expression, without the `#sair.mapping_expr` prefix.
// Accepts a ray stream so that it can be used from different flavors of
// printers.
void PrintMappingExpr(MappingExpr expr, llvm::raw_ostream &os) {
  if (auto none_expr = expr.dyn_cast<MappingNoneExpr>()) {
    os << MappingNoneExpr::kAttrName;
  } else if (auto dim_expr = expr.dyn_cast<MappingDimExpr>()) {
    os << "d" << dim_expr.dimension();
  } else if (auto stripe_expr = expr.dyn_cast<MappingStripeExpr>()) {
    os << MappingStripeExpr::kAttrName << "(";
    PrintMappingExpr(stripe_expr.operand(), os);
    os << ", " << stripe_expr.step();
    if (stripe_expr.size().hasValue()) {
      os << " size " << stripe_expr.size().getValue();
    }
    os << ")";
  } else if (auto unstripe_expr = expr.dyn_cast<MappingUnStripeExpr>()) {
    os << MappingUnStripeExpr::kAttrName << "(";
    for (auto operand : unstripe_expr.operands()) {
      PrintMappingExpr(operand, os);
      os << ", ";
    }
    os << "[";
    llvm::interleaveComma(unstripe_expr.factors(), os);
    os << "])";
  } else {
    llvm_unreachable("unknown mapping expression");
  }
}

// Prints the shape of an iteration domain. An iteration domain is a product of
// zero or more iteration dimensions separated by `x`. The 0-dimensional
// iteration domain is denoted `()`.
mlir::DialectAsmPrinter &operator<<(mlir::DialectAsmPrinter &os,
                                    DomainShapeAttr shape) {
  if (shape.Is0d()) {
    os << "()";
  } else {
    int i = 0;
    llvm::interleave(
        shape.Dimensions(), os,
        [&](const DomainShapeDim &dim) {
          os << "d" << i++ << ":" << RangeType::Name();
          if (dim.dependency_mapping().empty()) return;
          os << "(";
          PrintMapping(dim.dependency_mapping(), os.getStream());
          os << ")";
        },
        " x ");
  }
  return os;
}

// Prints the range type.
void Print(RangeType type, mlir::DialectAsmPrinter *os) {
  *os << RangeType::Name();
  if (!type.Shape().Is0d()) {
    *os << "<" << type.Shape() << ">";
  }
}

// Prints the value type.
void Print(ValueType type, mlir::DialectAsmPrinter *os) {
  *os << ValueType::Name() << "<" << type.Shape() << ", " << type.ElementType()
      << ">";
}

// Prints an mapping with the use domain size in front.
void PrintWithUseDomainSize(MappingAttr mapping, mlir::DialectAsmPrinter &os) {
  os << mapping.UseDomainSize();
  if (mapping.empty()) return;
  os << " : ";
  PrintMapping(mapping, os.getStream());
}

void Print(NamedMappingAttr named_mapping, mlir::DialectAsmPrinter &os) {
  os << "[";
  int i = 0;
  llvm::interleaveComma(named_mapping.names(), os, [&](mlir::StringAttr name) {
    os << 'd' << i++ << ":" << name;
  });
  os << "] -> (";
  PrintMapping(named_mapping.mapping(), os.getStream());
  os << ")";
}

}  // namespace

void PrintMapping(MappingAttr mapping, llvm::raw_ostream &os) {
  llvm::interleaveComma(mapping, os,
                        [&](MappingExpr expr) { PrintMappingExpr(expr, os); });
}

// Prints the Sair type using MLIR printing facilities.
void sair::SairDialect::printType(mlir::Type type,
                                  mlir::DialectAsmPrinter &os) const {
  if (auto range_type = type.dyn_cast<RangeType>())
    return Print(range_type, &os);

  Print(type.cast<ValueType>(), &os);
}

// Prints the Sair attribute using MLIR printing facilities.
void sair::SairDialect::printAttribute(mlir::Attribute attr,
                                       mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Attribute>(attr)
      .Case([&os](DomainShapeAttr shape_attr) {
        os << "shape<" << shape_attr << ">";
      })
      .Case([&os](MappingAttr mapping_attr) {
        os << "mapping<";
        PrintWithUseDomainSize(mapping_attr, os);
        os << ">";
      })
      .Case([&os](NamedMappingAttr named_mapping) {
        os << "named_mapping<";
        Print(named_mapping, os);
        os << ">";
      })
      .Case([&os](MappingExpr expr) {
        os << "mapping_expr<";
        PrintMappingExpr(expr, os.getStream());
        os << ">";
      });
}

}  // namespace sair
