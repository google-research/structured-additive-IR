// Copyright 2021 Google LLC
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

#ifndef SAIR_EXPANSION_H_
#define SAIR_EXPANSION_H_

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "util.h"

namespace sair {

constexpr llvm::StringRef kMapExpansionPattern = "map";
constexpr llvm::StringRef kCopyExpansionPattern = "copy";
constexpr llvm::StringRef kAllocExpansionPattern = "alloc";
constexpr llvm::StringRef kFreeExpansionPattern = "free";
constexpr llvm::StringRef kLoadExpansionPattern = "load";
constexpr llvm::StringRef kStoreExpansionPattern = "store";

// An expansion pattern for a Sair compute operation.
//
// The pattern is defined by a unique name, a `Match` method that tests if
// it can be used to implement a ComputeOp and a method to rewrite the ComputeOp
// into a SairMapOp that calls the pattern.
//
// This class is separate from MLIR rewrite patterns as `Match` needs to
// indicate if the pattern applies even if lowering hasn't started yet but
// `Emit` expects loop normalization, buffer materialization and copy
// insertion to have occured before running.
class ExpansionPattern {
 public:
  virtual ~ExpansionPattern() = default;

  // Indicates if `op` can be implemented by the pattern.
  virtual mlir::LogicalResult Match(ComputeOp op) const = 0;

  // Emits the pattern in `map_body`. Returns non-sair values that should be
  // returned from the map body.
  virtual llvm::SmallVector<mlir::Value> Emit(
      ComputeOp op, MapBodyBuilder &map_body,
      mlir::OpBuilder &builder) const = 0;
};

// A ExpansionPattern that only applies to ComputeOp of type OpTy.
template <typename OpTy>
class TypedExpansionPattern : public ExpansionPattern {
 public:
  virtual mlir::LogicalResult Match(OpTy op) const = 0;

  mlir::LogicalResult Match(ComputeOp op) const final {
    auto cast_op = dyn_cast<OpTy>(*op);
    if (cast_op == nullptr) return mlir::failure();
    return Match(cast_op);
  }

  virtual llvm::SmallVector<mlir::Value> Emit(
      OpTy op, MapBodyBuilder &map_body, mlir::OpBuilder &builder) const = 0;

  llvm::SmallVector<mlir::Value> Emit(ComputeOp op, MapBodyBuilder &map_body,
                                      mlir::OpBuilder &builder) const final {
    return Emit(cast<OpTy>(*op), map_body, builder);
  }
};

// Registers all expansion patterns in the map.
void RegisterExpansionPatterns(
    llvm::StringMap<std::unique_ptr<ExpansionPattern>> &map);

}  // namespace sair

#endif  // SAIR_EXPANSION_H_
