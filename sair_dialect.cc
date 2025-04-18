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
#include "llvm/Support/Casting.h"
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
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "util.h"

using mlir::failed;
using mlir::succeeded;

namespace sair {

// Registers Sair types with MLIR.
SairDialect::SairDialect(mlir::MLIRContext *context)
    : mlir::Dialect(getDialectNamespace(), context,
                    TypeID::get<SairDialect>()) {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "sair_ops.cc.inc"
      >();

  register_ = mlir::StringAttr::get(context, "register");
  memory_ = mlir::StringAttr::get(context, "memory");
  RegisterExpansionPatterns(expansion_patterns_);
}

namespace {

// Parses the size and step of a static range type and returns the corresponding
// type.
StaticRangeType ParseStaticRangeType(mlir::DialectAsmParser &parser) {
  int size;
  int step = 1;
  if (parser.parseLess() || parser.parseInteger(size)) return nullptr;
  if (succeeded(parser.parseOptionalComma())) {
    if (mlir::failed(parser.parseInteger(step))) return nullptr;
  }
  if (failed(parser.parseGreater())) return nullptr;
  return StaticRangeType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); }, size, step,
      parser.getBuilder().getContext());
}

// Parse a static range shape dimension and appends it to `dimensions`.
ParseResult ParseStaticRangeShapeDim(
    mlir::DialectAsmParser &parser,
    llvm::SmallVector<DomainShapeDim> &dimensions) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  StaticRangeType type = ParseStaticRangeType(parser);
  if (type == nullptr) return failure();
  auto mapping = MappingAttr::get(context, dimensions.size(), {});
  dimensions.emplace_back(type, mapping);
  return success();
}

// Parses a range of shape dimension and appends it to `dimensions`.
ParseResult ParseRangeShapeDim(mlir::DialectAsmParser &parser,
                               llvm::SmallVector<DomainShapeDim> &dimensions) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  llvm::SMLoc loc = parser.getCurrentLocation();
  MappingAttr mapping = ParseOptionalMapping(parser, dimensions.size());
  if (mapping == nullptr) return failure();
  if (mapping.HasNoneExprs() || mapping.HasUnknownExprs()) {
    return parser.emitError(loc) << "the mapping must map all dimensions";
  }

  auto shape = DomainShapeAttr::get(context, dimensions);
  AttrLocation attr_loc(parser.getEncodedSourceLoc(loc), "operation shape");
  if (mlir::failed(VerifyMappingShape(attr_loc, mapping, shape))) {
    return mlir::failure();
  }

  llvm::SmallBitVector seen_dimensions(dimensions.size());
  for (MappingExpr expr : mapping) {
    llvm::SmallBitVector expr_dependencies =
        expr.DependencyMask(dimensions.size());

    llvm::SmallBitVector transitive_dependencies(dimensions.size());
    for (int dimension : expr_dependencies.set_bits()) {
      transitive_dependencies |= dimensions[dimension].DependencyMask();
    }

    if ((~seen_dimensions).anyCommon(transitive_dependencies)) {
      return parser.emitError(loc) << "non-transitive dependency";
    }
    seen_dimensions |= expr_dependencies;
  }

  DomainShapeAttr range_shape = shape.AccessedShape(mapping);
  dimensions.emplace_back(DynRangeType::get(range_shape), mapping);
  return success();
}

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

  llvm::SmallVector<DomainShapeDim> dimensions;
  do {
    // Parse the dimension name.
    std::string expected_name = "d" + std::to_string(dimensions.size());
    if (parser.parseKeyword(expected_name) || parser.parseColon()) {
      return nullptr;
    }

    if (succeeded(parser.parseOptionalKeyword(StaticRangeType::Name()))) {
      if (failed(ParseStaticRangeShapeDim(parser, dimensions))) return nullptr;
    } else if (parser.parseKeyword(DynRangeType::Name()) ||
               ParseRangeShapeDim(parser, dimensions)) {
      return nullptr;
    }
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

  if (keyword == sair::DynRangeType::Name()) {
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
    return DynRangeType::get(domain);
  }

  if (keyword == sair::StaticRangeType::Name()) {
    return ParseStaticRangeType(parser);
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
  mlir::Attribute attribute;
  mlir::OptionalParseResult parse_result = detail::ParseGeneratedAttribute(
      parser.getBuilder().getContext(), parser, &keyword, type, attribute);
  if (parse_result.has_value()) {
    if (parse_result.value().succeeded()) return attribute;
    return mlir::Attribute();
  }

  if (parser.parseLess()) return nullptr;
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
  if (llvm::isa<MappingNoneExpr>(expr)) {
    os << MappingNoneExpr::kAttrName;
  } else if (llvm::isa<MappingUnknownExpr>(expr)) {
    os << MappingUnknownExpr::kAttrName;
  } else if (auto dim_expr = llvm::dyn_cast<MappingDimExpr>(expr)) {
    os << "d" << dim_expr.dimension();
  } else if (auto stripe_expr = llvm::dyn_cast<MappingStripeExpr>(expr)) {
    os << MappingStripeExpr::kAttrName << "(";
    PrintMappingExpr(stripe_expr.operand(), os);
    os << ", [";
    llvm::interleaveComma(stripe_expr.factors(), os);
    os << "])";
  } else if (auto unstripe_expr = llvm::dyn_cast<MappingUnStripeExpr>(expr)) {
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

// Prints the static range type.
void Print(StaticRangeType type, mlir::DialectAsmPrinter &os) {
  os << StaticRangeType::Name() << "<" << type.size();
  if (type.getStep() != 1) {
    os << ", " << type.getStep();
  }
  os << ">";
}

void PrintDomainShapeDim(const DomainShapeDim &dimension,
                         mlir::DialectAsmPrinter &os) {
  if (auto static_range = llvm::dyn_cast<StaticRangeType>(dimension.type())) {
    Print(static_range, os);
  } else if (llvm::isa<DynRangeType>(dimension.type())) {
    os << DynRangeType::Name();
  } else {
    llvm_unreachable("unsupported dimension type");
  }

  if (dimension.dependency_mapping().empty()) return;
  os << "(";
  PrintMapping(dimension.dependency_mapping(), os.getStream());
  os << ")";
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
          os << "d" << i++ << ":";
          PrintDomainShapeDim(dim, os);
        },
        " x ");
  }
  return os;
}

// Prints the range type.
void Print(DynRangeType type, mlir::DialectAsmPrinter &os) {
  os << DynRangeType::Name();
  if (!type.Shape().Is0d()) {
    os << "<" << type.Shape() << ">";
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
void SairDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &os) const {
  if (auto range_type = llvm::dyn_cast<DynRangeType>(type)) {
    return Print(range_type, os);
  } else if (auto static_range_type = llvm::dyn_cast<StaticRangeType>(type)) {
    return Print(static_range_type, os);
  }

  Print(llvm::cast<ValueType>(type), &os);
}

// Prints the Sair attribute using MLIR printing facilities.
void SairDialect::printAttribute(mlir::Attribute attr,
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
      })
      .Default([&os](mlir::Attribute attr) {
        AssertSuccess(detail::PrintGeneratedAttribute(attr, os));
      });
}

const ExpansionPattern *SairDialect::GetExpansionPattern(
    llvm::StringRef name) const {
  auto it = expansion_patterns_.find(name);
  if (it == expansion_patterns_.end()) return nullptr;
  return it->second.get();
}

}  // namespace sair
