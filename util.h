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
#include "sair_op_interfaces.h"

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

// Forwards attributes of old_op to new_op. Skips attributes already set in
// `new_op`.
void ForwardAttributes(mlir::Operation *old_op, mlir::Operation *new_op,
                       llvm::ArrayRef<llvm::StringRef> ignore = {});

// Materializes `value` as an mlir value.
mlir::Value Materialize(mlir::Location loc, mlir::OpFoldResult value,
                        mlir::OpBuilder &builder);

// Behaves like assert(mlir::succeeded(expr)) but always executes expr.
inline void AssertSuccess(mlir::LogicalResult result) {
  assert(mlir::succeeded(result));
}

// Helper class to build a sair.map operation when the set of operands is not
// known in advance. Allocates a block to hold the map body and maintains the
// correspondance between block arguments and sair.map arguments. It is expected
// that the calling code will create the map operation and then transfer the
// body to the map operation.
class MapBodyBuilder {
 public:
  // Creates a map body builder for a sair.map with the given domain size.
  MapBodyBuilder(int domain_size, mlir::MLIRContext *context);

  // Returns the value holding the index of the given dimension.
  mlir::Value index(int dimension);
  mlir::ValueRange indices();

  // Returns the scalar value for an operand.
  mlir::Value block_input(int position);
  mlir::ValueRange block_inputs();

  // Return Sair value operands of the map operation.
  llvm::ArrayRef<ValueAccess> sair_values() { return operands_; }

  // Returns the block holding the map body.
  mlir::Block &block() { return region_.front(); }

  // Returns the region holding the map body.
  mlir::Region &region() { return region_; }

  // Adds an operand to the map operation and returns the scalar value to use
  // inside the map body.
  mlir::Value AddOperand(ValueAccess operand);

 private:
  int domain_size_;
  llvm::SmallVector<ValueAccess> operands_;
  mlir::Region region_;
};

// Parameters of a Sair range. Begin and end of the range can be either values
// or constants.
struct RangeParameters {
  // First index of the range.
  mlir::OpFoldResult begin;
  // End of the range.
  mlir::OpFoldResult end;
  // Step of the range.
  int step;
};

// Returns the parameters (start index, end index and step) of the ranges
// obtained by applying mapping to `source_domain`. Populates `current_body`
// with operations to compute parameters. Assumes that `current_to_source` is a
// mapping from the domain of the map wrapping `current_body` operation to
// `source_domain`.
llvm::SmallVector<RangeParameters> GetRangeParameters(
    mlir::Location loc, MappingAttr mapping,
    llvm::ArrayRef<ValueAccess> source_domain, MappingAttr current_to_source,
    MapBodyBuilder &current_body, mlir::OpBuilder &builder);

}  // namespace sair

#endif  // THIRD_PARTY_SAIR_TRANSFORMS_UTIL_H_
