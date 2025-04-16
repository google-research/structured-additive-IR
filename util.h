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
#include "mlir/Support/LLVM.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"

namespace sair {

class ValueAccess;
class ComputeOp;
class SairOp;

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

// Returns a function that applies a function to each element of an array
// attribute with elements of type T. Returns nullptr if the input is null.
//
// This returns a function rather that directly taking the array as argument in
// order to make it easier to nest function combinators.
template <typename T>
std::function<mlir::ArrayAttr(mlir::ArrayAttr)> MkArrayAttrMapper(
    std::function<T(T)> scalar_fn) {
  return [scalar_fn](mlir::ArrayAttr array) {
    if (array == nullptr) return array;
    llvm::SmallVector<mlir::Attribute> output;
    output.reserve(array.size());
    for (mlir::Attribute attr : array.getValue()) {
      output.push_back(scalar_fn(mlir::cast<T>(attr)));
    }
    return mlir::ArrayAttr::get(array.getContext(), output);
  };
}

// Returns a function that filters out elements of an array attribute based on a
// mask. Element `i` is kept if and only if `mask[i]` is true. Returns `nullptr`
// if the input is null.
//
// This returns a function rather that directly taking the array as argument in
// order to make it easier to nest function combinators.
std::function<mlir::ArrayAttr(mlir::ArrayAttr)> MkArrayAttrFilter(
    llvm::SmallBitVector mask);

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
