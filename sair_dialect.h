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
  // The string identifier used for access pattern attribute in Sair ops.
  static constexpr llvm::StringRef kAccessPatternAttrName =
      "access_pattern_array";

  // The string identifier used for shape attribute in Sair ops.
  static constexpr llvm::StringRef kShapeAttrName = "shape";

  // String identifier used for dimensions of the accessed domain that do no map
  // to a dimension of the use domain in access patterns.
  static constexpr llvm::StringRef kNoneKeyword = "none";

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
};

// Pretty-prints an access pattern, for use in custom printers. In particular,
// the use domain size is omitted. This functions access a raw stream so that it
// can be used with different flavors of printers.
void PrintAccessPattern(AccessPatternAttr access_pattern,
                        llvm::raw_ostream &os);

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

// Parses an access pattern expression in a context with `num_dimensions`.
// Returns `nullptr` on failure. Considers a context with an infinite number of
// dimensions if `num_dimensions` is `-1`.
template <typename Parser>
AccessPatternExpr ParseAccessPatternExpr(Parser &parser,
                                         int num_dimensions = -1) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  if (mlir::succeeded(parser.parseOptionalKeyword(SairDialect::kNoneKeyword))) {
    return AccessPatternNoneExpr::get(context);
  }

  int dimension_id;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (mlir::failed(ParseDimensionName(parser, dimension_id))) {
    return AccessPatternExpr();
  }

  if (num_dimensions != -1 && dimension_id >= num_dimensions) {
    parser.emitError(loc) << "dimension 'd" << dimension_id
                          << "' is out of range (" << num_dimensions
                          << " dimensions)";
    return AccessPatternExpr();
  }
  return AccessPatternDimExpr::get(dimension_id, context);
}

// Parses a non-empty access pattern. Returns nullptr if the parsing fails.
template <typename Parser>
AccessPatternAttr ParseAccessPattern(Parser &parser, int num_dimensions) {
  std::vector<AccessPatternExpr> exprs;
  llvm::SmallBitVector seen_dimensions(num_dimensions);
  do {
    llvm::SMLoc loc = parser.getCurrentLocation();
    AccessPatternExpr expr = ParseAccessPatternExpr(parser, num_dimensions);
    if (expr == nullptr) return nullptr;
    llvm::SmallBitVector new_seen_dimensions(num_dimensions);
    expr.SetDependenciesInMask(new_seen_dimensions);
    exprs.push_back(expr);
    if (seen_dimensions.anyCommon(new_seen_dimensions)) {
      int dim = (seen_dimensions & new_seen_dimensions).find_first();
      parser.emitError(loc) << "dimension d" << dim << " appears twice";
      return nullptr;
    }
    seen_dimensions |= new_seen_dimensions;
  } while (succeeded(parser.parseOptionalComma()));

  return AccessPatternAttr::get(parser.getBuilder().getContext(),
                                num_dimensions, exprs);
}

// Parses an access pattern surrounded by parenthesis or returns the empty
// access pattern if the next token is not a parenthesis. Returns nullptr if the
// parsing fails.
template <typename Parser>
AccessPatternAttr ParseOptionalAccessPattern(Parser &parser,
                                             int num_dimensions) {
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  if (failed(parser.parseOptionalLParen())) {
    return AccessPatternAttr::get(context, num_dimensions, {});
  }

  AccessPatternAttr access_pattern = ParseAccessPattern(parser, num_dimensions);
  if (failed(parser.parseRParen())) return nullptr;
  return access_pattern;
}

}  // namespace sair

#endif  // SAIR_SAIR_DIALECT_H_
