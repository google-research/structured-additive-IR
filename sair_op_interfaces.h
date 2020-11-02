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

// A !sair.value operand of a Sair operation.
class ValueOperand {
 public:
  // Builds a 'ValueOperand' from a pointer to the MLIR operand and the index of
  // the access pattern in the array of access patterns of the Sair operation.
  //
  // Stores the 'operand' pointer without taking ownership.
  ValueOperand(mlir::OpOperand *operand, int index)
      : operand_(operand), index_(index) {}

  // Returns the value referenced by the operand.
  mlir::Value value() const { return operand_->get(); }

  // Returns the type of the value referenced by the operand.
  ValueType GetType() const {
    return operand_->get().getType().cast<ValueType>();
  }

  // Returns the operation owning the operand.
  mlir::Operation *getOwner() { return operand_->getOwner(); }

  // Returns the access pattern associated to the operand.
  AccessPatternAttr AccessPattern();

  // Returns the position of the operand.
  int position() const { return index_; }

  // Sets the value referenced by the operand.
  void set_value(mlir::Value value) { operand_->set(value); }

  // Sets the access pattern associated to the operand.
  void SetAccessPattern(AccessPatternAttr access_pattern);

  // Returns a mask of dimensions that must execute after the operand is
  // computed.
  llvm::SmallBitVector DependingDims() const;

  // Indicates if the operand can be used before it is defined.
  bool AllowUseBeforeDef() const;

  // If the operand is a loop-carried dependency, indicates along which
  // dimensions it is carried.
  llvm::SmallBitVector CarryingDims() const;

 private:
  mlir::OpOperand *operand_;
  int index_;
};

// Exposes the !sair.value operands of a Sair operation. Each element of the
// range is a `ValueOperand`.
class ValueOperandRange
    : public llvm::detail::indexed_accessor_range_base<
          ValueOperandRange, std::pair<mlir::OpOperand *, int>, ValueOperand,
          ValueOperand, ValueOperand> {
 public:
  // Import constructors from the base class.
  using RangeBaseT::RangeBaseT;
  // Constructs the empty range.
  ValueOperandRange();

  // Constructs a range from the list of !sair.value operands and the
  // corresponding list of access patterns. Both arguments must have the same
  // size.
  explicit ValueOperandRange(llvm::MutableArrayRef<mlir::OpOperand> operands);

 private:
  using PtrPair = std::pair<mlir::OpOperand *, int>;

  // See `llvm::detail::indexed_accessor_range_base` for details.
  static PtrPair offset_base(PtrPair base, ptrdiff_t offset);
  // See `llvm::detail::indexed_accessor_range_base` for details.
  static ValueOperand dereference_iterator(PtrPair base, ptrdiff_t offset);

  // Allow access to offset_base and dereference_iterator from the base type.
  friend RangeBaseT;
};

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

// Returns the memory space of the given result.
llvm::Optional<int> GetMemorySpace(int result, mlir::Operation *op);

// Returns the memory space of a sair value, if set.
llvm::Optional<int> GetMemorySpace(mlir::Value value);

// Sets the memory space of the given result. Expects operation to be a
// `ValueProducerOp`.
void SetMemorySpace(int result, llvm::Optional<int> memory_space,
                    mlir::Operation *op);

// Verifies a `SairOp`.
mlir::LogicalResult VerifySairOp(mlir::Operation *op);

// Verifies a `ValueProducerOp`.
mlir::LogicalResult VerifyValueProducerOp(mlir::Operation *op);

// Verifies a `ComputeOp`.
mlir::LogicalResult VerifyComputeOp(mlir::Operation *op);

// Verifies a `RangeOp`.
mlir::LogicalResult VerifyRangeOp(mlir::Operation *op);

// Returns the Sair value accessed by the operation, along with the
// corresponding access patterns.
template<typename ConcreteType>
::sair::ValueOperandRange ValueOperands(ConcreteType op) {
  auto operands = op.getOperation()->getOpOperands()
                      .drop_front(op.domain().size())
                      .take_front(op.access_pattern_array().size());
  return ::sair::ValueOperandRange(operands);
}

// Sets the access pattern at the given position.
template<typename ConcreteType>
void SetAccessPattern(ConcreteType op, int position,
                      ::sair::AccessPatternAttr access_pattern) {
  llvm::SmallVector<mlir::Attribute, 4> new_array =
      llvm::to_vector<4>(op.access_pattern_array());
  new_array[position] = access_pattern;
  mlir::ArrayAttr new_attr = mlir::ArrayAttr::get(new_array, op.getContext());
  op.setAttr(::sair::SairDialect::kAccessPatternAttrName, new_attr);
}

using namespace mlir;  // NOLINT
#include "sair_op_interfaces.h.inc"

}  // namespace sair

#endif  // SAIR_SAIR_OP_INTERFACES_H_
