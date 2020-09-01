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

// Removes the last iteration dimension for the domain of the operation with the
// SairOpWithBody trait.
template <typename ConcreteOp>
void RemoveInnermostDimension(ConcreteOp op) {
  assert(!op.domain().empty());
  int dimension_position = op.domain().size() - 1;
  op.block().eraseArgument(dimension_position);
  EraseOperand(dimension_position, ConcreteOp::getOperandSegmentSizeAttr(), op);
  llvm::ArrayRef<DomainShapeDim> shape_dimensions =
      op.shape().Dimensions().drop_back();
  op.shapeAttr(DomainShapeAttr::get(op.getContext(), shape_dimensions));
}

// Appends a !sair.value operand to the operation with the SairOpWithBody trait.
template <typename ConcreteOp>
mlir::BlockArgument AddValueOperand(ConcreteOp op, mlir::Value value,
                                    AccessPatternAttr access_pattern) {
  AppendOperand(value, ConcreteOp::getOperandSegmentSizeAttr(), op);
  AppendAccessPattern(access_pattern, op);
  auto value_type = value.getType().cast<ValueType>();
  return op.block().addArgument(value_type.ElementType());
}

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
  // Dimension of the use operation that must be fused with the dimensions of
  // the def operation they map to.
  llvm::SmallBitVector fuse_dimensions;
  // Point-to-point communication pattern from def to use.
  AccessPatternAttr mapped_dimensions;
};

// Returns the dependencies of `op`.
llvm::SmallVector<Dependency, 4> Dependencies(SairOp op);

}  // namespace sair

#endif  // SAIR_SAIR_OP_INTERFACES_H_
