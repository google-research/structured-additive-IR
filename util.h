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

#ifndef THIRD_PARTY_SAIR_TRANSFORMS_UTIL_H_
#define THIRD_PARTY_SAIR_TRANSFORMS_UTIL_H_

#include "mlir/IR/Builders.h"
#include "sair_op_interfaces.h"

namespace sair {

// Position of an operation relative to another.
enum class Direction { kBefore, kAfter };

// Specifies where to insert an operation in the generated code. The operation
// is inserted before or after 'operation', depending on `direction` and
// is nested in 'loop_nest'.
struct InsertionPoint {
  mlir::Operation *operation;
  Direction direction;
  mlir::ArrayAttr loop_nest;

  // Sets the insertion point of the builder.
  void Set(mlir::OpBuilder &builder) const;
};

// Finds the first point in the program where it is possible to insert an
// operation nested in the first `num_loops` of `current_loop_nest`, when
// starting from `start`.
InsertionPoint FindInsertionPoint(
    SairOp start, llvm::ArrayRef<mlir::Attribute> current_loop_nest,
    int num_loops, Direction direction = Direction::kBefore);

// Forwards attributes of old_op to new_op. Skips attributes already set in
// `new_op`.
void ForwardAttributes(mlir::Operation *old_op, mlir::Operation *new_op,
                       llvm::ArrayRef<llvm::StringRef> ignore = {});

// Resolves a unfication constraint for dimension `dimension` of `op`.
//
// Adds dimension `dimension` to `target_domain` while making sure that the
// expression pointing to `dimension` in `target_domain` can be unified with
// `constraint`. Reports an error if unification fails. `origin` is used to
// indicate where the error comes from.
//
// The target domain is defined with regard to a number of dimensions, called
// its dependencies. Each value access in `target_domain` specifies how to
// access a range variable given the current index along dependencies.
// `target_deps_to_op` is a mapping from dependencies to `op` domain.
mlir::LogicalResult ResolveUnificationConstraint(
    ComputeOp op, int dimension, llvm::StringRef origin,
    MappingAttr target_deps_to_op, MappingExpr &constraint,
    llvm::SmallVectorImpl<ValueAccess> &target_domain);

}  // namespace sair

#endif  // THIRD_PARTY_SAIR_TRANSFORMS_UTIL_H_
