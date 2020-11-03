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
  addAttributes<DomainShapeAttr, AccessPatternAttr, IteratorAttr>();
  addOperations<
#define GET_OP_LIST
#include "sair_ops.cc.inc"
      >();
}

namespace {

// Parses an iterator on a dimension of the domain. Returns nullptr in case of
// failure.
IteratorAttr ParseIterator(mlir::DialectAsmParser &parser) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();

  int dimension;
  if (mlir::succeeded(parser.parseOptionalKeyword("remat"))) {
    return IteratorAttr::get(context);
  }

  if (mlir::failed(ParseDimensionName(parser, dimension))) {
    return nullptr;
  }

  int step = 1;
  if (mlir::succeeded(parser.parseOptionalKeyword("step"))) {
    llvm::SMLoc loc = parser.getCurrentLocation();
    if (mlir::failed(parser.parseInteger(step))) return nullptr;
    if (step <= 0) {
      parser.emitError(loc) << "step must be positive";
      return nullptr;
    }
  }

  return IteratorAttr::get(context, dimension, step);
}

// Parses the shape of a Sair domain, as expressed in Sair types. Returns
// nullptr in case of failure. Domains shapes are composed of a list of
// dimension types separated by 'x', with an optional dependency pattern for
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
    AccessPatternAttr access_pattern =
        ParseOptionalAccessPattern(parser, dimensions.size());
    if (access_pattern == nullptr) return nullptr;
    if (!access_pattern.IsFullySpecified()) {
      parser.emitError(loc) << "the access pattern must map all dimensions";
      return nullptr;
    }

    std::vector<DomainShapeDim> arg_shape_dims;
    llvm::SmallBitVector seen_dimensions(dimensions.size());
    for (int dimension : access_pattern) {
      llvm::SmallBitVector dependencies =
          dimensions[dimension].DependencyMask();
      if ((~seen_dimensions).anyCommon(dependencies)) {
        parser.emitError(loc) << "non-transitive dependency";
        return nullptr;
      }
      seen_dimensions.set(dimension);
      arg_shape_dims.push_back(
          dimensions[dimension].Inverse(access_pattern, arg_shape_dims.size()));
    }
    RangeType arg_type =
        RangeType::get(context, DomainShapeAttr::get(context, arg_shape_dims));
    dimensions.emplace_back(arg_type, access_pattern);
  } while (succeeded(parser.parseOptionalKeyword("x")));
  return DomainShapeAttr::get(context, dimensions);
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
    return RangeType::get(getContext(), domain);
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
    return ValueType::get(getContext(), domain, element_type);
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
  } else if (keyword == "pattern") {
    int num_dimensions;
    if (failed(parser.parseInteger(num_dimensions))) return nullptr;
    if (succeeded(parser.parseOptionalColon())) {
      attribute = ParseAccessPattern(parser, num_dimensions);
    } else {
      attribute = AccessPatternAttr::get(getContext(), num_dimensions, {});
    }
  } else if (keyword == "iter") {
    attribute = ParseIterator(parser);
  } else {
    parser.emitError(parser.getNameLoc())
        << "unexpected Sair attribute '" << keyword << "'";
    return nullptr;
  }
  if (!attribute || parser.parseGreater()) return nullptr;
  return attribute;
}

namespace {

// Prints an access pattern.
mlir::DialectAsmPrinter &operator<<(mlir::DialectAsmPrinter &os,
                                    AccessPatternAttr access_pattern) {
  llvm::interleaveComma(access_pattern.getValue(), os, [&](int dimension) {
    if (dimension == AccessPatternAttr::kNoDimension) {
      os << SairDialect::kNoneKeyword;
    } else {
      os << "d" << dimension;
    }
  });
  return os;
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
          if (dim.dependency_pattern().empty()) return;
          os << "(" << dim.dependency_pattern() << ")";
        },
        " x ");
  }
  return os;
}

// Prints the description of a generated loop.
mlir::DialectAsmPrinter &operator<<(mlir::DialectAsmPrinter &os,
                                    IteratorAttr iterator) {
  if (iterator.Rematerialize()) {
    return os << "remat";
  }

  os << "d" << iterator.Dimension();
  if (iterator.Step() > 1) {
    os << " step " << iterator.Step();
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

// Prints an access pattern with the use domain size in front.
void PrintWithUseDomainSize(AccessPatternAttr access_pattern,
                            mlir::DialectAsmPrinter &os) {
  os << access_pattern.UseDomainSize();
  if (access_pattern.empty()) return;
  os << " : " << access_pattern;
}

}  // namespace

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
      .Case([&os](AccessPatternAttr pattern_attr) {
        os << "pattern<";
        PrintWithUseDomainSize(pattern_attr, os);
        os << ">";
      })
      .Case(
          [&os](IteratorAttr iter_attr) { os << "iter<" << iter_attr << ">"; });
}

}  // namespace sair
