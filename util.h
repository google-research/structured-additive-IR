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
#include "sair_attributes.h"

namespace sair {

class ValueAccess;
class ComputeOp;
class SairOp;

// Position of an operation relative to another.
enum class Direction { kBefore, kAfter };

// Specifies where to insert an operation in the generated code. The operation
// is inserted before or after 'operation', depending on `direction` and
// is nested in 'loop_nest'.
// TODO(ulysse): merge with program point.
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

// Sets an element in the array attribute of an operation. If the array
// attribute is missing, creates a new array of the given size filled with
// `unit` attributes.
void SetInArrayAttr(mlir::Operation *operation, llvm::StringRef attr_name,
                    int array_size, int element, mlir::Attribute value);

// Materializes `value` as an mlir value.
mlir::Value Materialize(mlir::Location loc, mlir::OpFoldResult value,
                        mlir::OpBuilder &builder);

// Behaves like assert(mlir::succeeded(expr)) but always executes expr.
inline void AssertSuccess(mlir::LogicalResult result) {
  assert(mlir::succeeded(result));
}

}  // namespace sair

#endif  // THIRD_PARTY_SAIR_TRANSFORMS_UTIL_H_
