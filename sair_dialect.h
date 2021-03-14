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

#ifndef SAIR_SAIR_DIALECT_H_
#define SAIR_SAIR_DIALECT_H_

#include <limits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"

namespace sair {

// Structured Additive IR dialect. Contains and registers with MLIR context the
// lists of types, attributes and operations, and provides dialect-specific
// parsing and printing facilities.
class SairDialect : public mlir::Dialect {
 public:
  // The string identifier used for mapping attribute in Sair ops.
  static constexpr llvm::StringRef kMappingAttrName = "mapping_array";

  // The string identifier used for shape attribute in Sair ops.
  static constexpr llvm::StringRef kShapeAttrName = "shape";

  // The string identifier used for affine access map attribute in Sair ops.
  static constexpr llvm::StringRef kAccessMapAttrName = "access_map";

  // Identifiers for memory spaces.
  mlir::StringAttr register_attr() const { return register_; }
  mlir::StringAttr memory_attr() const { return memory_; }

  // Constructs the dialect in the provided context.
  explicit SairDialect(mlir::MLIRContext *context);

  // Returns the namespace used by the dialect. Hook for MLIR dialect lookup.
  static llvm::StringRef getDialectNamespace() { return "sair"; }

  // Parses the dialect-specific part of the MLIR dialect type and returns it.
  // Hook for MLIR parser.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  // Prints the dialect type to the given raw output stream. Hook for MLIR asm
  // printer.
  void printType(mlir::Type type, mlir::DialectAsmPrinter &os) const override;

  // Parses the Sair dialect attribute. Returns nullptr in case of failure.
  // 'type' is expected to be null as Sair attributes do not have a type.
  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;

  // Prints the dialect attribute. Hook for the MLIR asm printer.
  void printAttribute(mlir::Attribute attribute,
                      mlir::DialectAsmPrinter &os) const override;

 private:
  /// Register the attributes of this dialect.
  void registerAttributes();
  /// Register the types of this dialect.
  void registerTypes();

  mlir::StringAttr register_, memory_;
};

// Pretty-prints an mapping, for use in custom printers. In particular,
// the use domain size is omitted. This functions access a raw stream so that it
// can be used with different flavors of printers.
void PrintMapping(MappingAttr mapping, llvm::raw_ostream &os);

// Parses an integer in [min, max). Stores the result in `result`.
template <typename Parser>
mlir::ParseResult ParseInteger(
    Parser &parser, int &result, int min,
    llvm::Optional<int> max = llvm::Optional<int>()) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  mlir::IntegerAttr attr;
  if (mlir::failed(parser.parseAttribute(attr))) return mlir::failure();
  result = attr.getInt();
  if (result < min) {
    return parser.emitError(loc) << "expected an integer >= " << min;
  }
  if (max.hasValue() && result >= max.getValue()) {
    return parser.emitError(loc) << "expected an integer < " << max.getValue();
  }
  return mlir::success();
}

// Parses a dimension name of the form 'd<id>' where <id> is an integer in the
// half open interval [0, num_dimensions). Stores <id> in `dimension`.
template <typename Parser>
mlir::ParseResult ParseDimensionName(Parser &parser, int &dimension) {
  llvm::StringRef name;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseKeyword(&name))) return mlir::failure();
  if (!name.consume_front("d") || name.getAsInteger(10, dimension) ||
      dimension < 0) {
    return parser.emitError(loc) << "invalid dimension name";
  }

  return mlir::success();
}

// Parses an mapping expression in a context with `num_dimensions`.
// Returns `nullptr` on failure. Considers a context with an infinite number of
// dimensions if `num_dimensions` is `-1`.
template <typename Parser>
MappingExpr ParseMappingExpr(Parser &parser, int num_dimensions = -1) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  if (mlir::succeeded(
          parser.parseOptionalKeyword(MappingNoneExpr::kAttrName))) {
    return MappingNoneExpr::get(context);
  } else if (mlir::succeeded(
                 parser.parseOptionalKeyword(MappingStripeExpr::kAttrName))) {
    // Parse a stripe expression of the form:
    // `stripe` `(` <operand>`,` <step> (`size` <size>)? `)`
    if (mlir::failed(parser.parseLParen())) return MappingExpr();
    MappingExpr operand = ParseMappingExpr(parser, num_dimensions);
    int step;
    if (parser.parseComma() || ParseInteger(parser, step, 1)) {
      return MappingExpr();
    }
    llvm::Optional<int> size_opt;
    if (mlir::succeeded(parser.parseOptionalKeyword("size"))) {
      int size;
      if (mlir::failed(ParseInteger(parser, size, step))) {
        return MappingExpr();
      }
      size_opt = size;
    }
    if (mlir::failed(parser.parseRParen())) return MappingExpr();
    return MappingStripeExpr::get(operand, step, size_opt);
  } else if (mlir::succeeded(
                 parser.parseOptionalKeyword(MappingUnStripeExpr::kAttrName))) {
    // Parse an unstrip expression of the form:
    // `unstripe` `(` (<operand> `,`)+ `[` (<factor> (`,` <factor>)*)? `]` `)`
    llvm::SmallVector<int, 3> factors;
    llvm::SmallVector<MappingExpr, 4> operands;
    if (mlir::failed(parser.parseLParen())) return MappingExpr();
    do {
      MappingExpr operand = ParseMappingExpr(parser, num_dimensions);
      if (operand == nullptr || mlir::failed(parser.parseComma())) {
        return MappingExpr();
      }
      operands.push_back(operand);
    } while (mlir::failed(parser.parseOptionalLSquare()));

    if (operands.size() > 1 &&
        mlir::failed(ParseInteger(parser, factors.emplace_back(), 1))) {
      return MappingExpr();
    }
    for (int i = 1, e = operands.size() - 1; i < e; ++i) {
      int last_factor = factors.back();
      if (parser.parseComma() ||
          ParseInteger(parser, factors.emplace_back(), 1, last_factor)) {
        return MappingExpr();
      }
    }
    if (parser.parseRSquare() || parser.parseRParen()) {
      return MappingExpr();
    }
    return MappingUnStripeExpr::get(operands, factors);
  }

  int dimension_id;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (mlir::failed(ParseDimensionName(parser, dimension_id))) {
    return MappingExpr();
  }

  if (num_dimensions != -1 && dimension_id >= num_dimensions) {
    parser.emitError(loc) << "dimension 'd" << dimension_id
                          << "' is out of range (" << num_dimensions
                          << " dimensions)";
    return MappingExpr();
  }
  return MappingDimExpr::get(dimension_id, context);
}

// Parses a non-empty mapping. Returns nullptr if the parsing fails.
template <typename Parser>
MappingAttr ParseMapping(Parser &parser, int num_dimensions) {
  std::vector<MappingExpr> exprs;
  llvm::SmallBitVector seen_dimensions(num_dimensions);
  llvm::SMLoc mapping_loc = parser.getCurrentLocation();
  do {
    llvm::SMLoc loc = parser.getCurrentLocation();
    MappingExpr expr = ParseMappingExpr(parser, num_dimensions);
    if (expr == nullptr) return nullptr;
    exprs.push_back(expr);
  } while (succeeded(parser.parseOptionalComma()));

  auto mapping = MappingAttr::getChecked(parser.getBuilder().getContext(),
                                         num_dimensions, exprs);
  if (mapping == nullptr) {
    parser.emitError(mapping_loc) << "invalid mapping";
    return nullptr;
  }
  return mapping;
}

// Parses an mapping surrounded by parenthesis or returns the empty
// mapping if the next token is not a parenthesis. Returns nullptr if the
// parsing fails.
template <typename Parser>
MappingAttr ParseOptionalMapping(Parser &parser, int num_dimensions) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  if (failed(parser.parseOptionalLParen())) {
    return MappingAttr::get(context, num_dimensions, {});
  }

  MappingAttr mapping = ParseMapping(parser, num_dimensions);
  if (failed(parser.parseRParen())) return nullptr;
  return mapping;
}

}  // namespace sair

#endif  // SAIR_SAIR_DIALECT_H_
