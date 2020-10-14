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

#ifndef SAIR_SAIR_OP_INTERFACES_H_
#define SAIR_SAIR_OP_INTERFACES_H_

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_traits.h"
#include "sair_types.h"

namespace sair {

// Represents wither a Sair value or a constant.
class ValueOrConstant {
 public:
  ValueOrConstant(mlir::Value value, AccessPatternAttr access_pattern)
      : value_(value), attribute_(access_pattern) {}
  ValueOrConstant(mlir::Attribute constant) : attribute_(constant) {}
  ValueOrConstant(ValueOperand &&operand)
      : ValueOrConstant(operand.value(), operand.AccessPattern()) { }

  bool is_constant() const { return value_ == nullptr; }
  bool is_value() const { return !is_constant(); }

  mlir::Value value() const {
    assert(is_value());
    return value_;
  }

  AccessPatternAttr access_pattern() const {
    assert(is_value());
    return attribute_.cast<AccessPatternAttr>();
  }

  mlir::Attribute constant() const {
    assert(is_constant());
    return attribute_;
  }

 private:
  mlir::Value value_;
  mlir::Attribute attribute_;
};

// Erases the operand of an operation with AttrSizedOperandSegments trait.
void EraseOperand(int position, llvm::StringRef segment_sizes_attribute_name,
                  mlir::Operation *op);

// Appends an operand to the last segment of an operation with
// AttrSizedOperandSegments trait.
void AppendOperand(mlir::Value operand,
                   llvm::StringRef segment_sizes_attribute_name,
                   mlir::Operation *op);

// Appends an access pattern to a Sair operation.
void AppendAccessPattern(AccessPatternAttr access_pattern, mlir::Operation *op);

class SairValueProducerOp;

// Returns the memory space of the given result.
llvm::Optional<int> GetMemorySpace(int result, mlir::Operation *op);

// Returns the memory space of a sair value, if set.
llvm::Optional<int> GetMemorySpace(mlir::Value value);

// Sets the memory space of the given result. Expects operation to be a
// `ValueProducerOp`.
void SetMemorySpace(int result, llvm::Optional<int> memory_space,
                    mlir::Operation *op);

// Verifies a `ValueProducerOp`.
mlir::LogicalResult VerifyValueProducerOp(mlir::Operation *op);

// Verifies a `ComputeOp`.
mlir::LogicalResult VerifyComputeOp(mlir::Operation *op);

// Verifies a `RangeOp`.
mlir::LogicalResult VerifyRangeOp(mlir::Operation *op);

using namespace mlir;  // NOLINT
#include "sair_op_interfaces.h.inc"

// A dependency of a Sair operation.
struct Dependency {
  // Operation the use operation depends on.
  ComputeOp def;
  // Dimensions of the def operation that must complete before the current
  // instance of the use operation execute.
  llvm::SmallBitVector def_only_dimensions;
  // Dimensions of the use operation that cannot execute before the accessed
  // instance of def is computed.
  llvm::SmallBitVector use_only_dimensions;
  // Dimension of the use operation that carry the dependency accross
  // iterations. They must be fused with the dimensions of the def operation
  // they map to.
  llvm::SmallBitVector carrying_dimensions;
  // Dimensions of the def operation that must complete at the previous
  // iteration of `carrying_dimensions`. In practice, this means that they are
  // nested in carrying dimensions.
  llvm::SmallBitVector prev_def_only_dimensions;
  // Point-to-point communication pattern from def to use.
  AccessPatternAttr mapped_dimensions;
};

// Adds dependencies of `op` to `dependencies` and for each dimension `di` of
// `op`, adds the dimensions `di` must be nested in to
// `dimension_dependencies[i]`.
void GetDependencies(
    SairOp op, llvm::SmallVectorImpl<Dependency> &dependencies,
    llvm::SmallVectorImpl<llvm::SmallBitVector> &dimension_dependencies);

}  // namespace sair

#endif  // SAIR_SAIR_OP_INTERFACES_H_
