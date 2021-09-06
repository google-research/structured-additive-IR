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

#include <variant>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_types.h"

namespace sair {

class IterationSpace;

// A Sair value accessed with a mapping.
struct ValueAccess {
  mlir::Value value;
  MappingAttr mapping;

  // Returns the element type of the value.
  mlir::Type ElementType() const;
};

bool operator==(const ValueAccess &lhs, const ValueAccess &rhs);
bool operator!=(const ValueAccess &lhs, const ValueAccess &rhs);

// A !sair.value operand of a Sair operation.
class ValueOperand {
 public:
  // Builds a 'ValueOperand' from a pointer to the MLIR operand and the index of
  // the mapping in the array of mappings of the Sair operation.
  //
  // Stores the 'operand' pointer without taking ownership.
  explicit ValueOperand(mlir::OpOperand *operand);

  // Returns the value referenced by the operand.
  mlir::Value value() const { return operand_->get(); }

  // Returns the mapping associated to the operand.
  MappingAttr Mapping() const;

  // Returns the value access of the operand.
  ValueAccess Get() const { return {value(), Mapping()}; }

  // Returns the type of the value referenced by the operand.
  ValueType GetType() const {
    return operand_->get().getType().cast<ValueType>();
  }

  // Returns the operation owning the operand.
  mlir::Operation *getOwner() const { return operand_->getOwner(); }

  // Returns the position of the operand.
  int position() const { return index_; }

  // Substitutes the value with a new one. The mlir value is replaced and the
  // new mapping is composed with the new one.
  void SubstituteValue(ValueAccess new_value);

  // Sets the value referenced by the operand.
  void set_value(mlir::Value value) { operand_->set(value); }

  // Sets the mapping associated to the operand.
  void SetMapping(MappingAttr mapping);

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

  // Constructs a range from the list of !sair.value operands.
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

// Update all uses of `value` to use `newValue` instead, and compose the access
// mapping with `mapping`.
void UpdateValueUses(mlir::Value value, ValueAccess new_value);

// Represents wither a Sair value or a constant.
class ValueOrConstant {
 public:
  ValueOrConstant(ValueAccess value) : variant_(value) {}
  ValueOrConstant(mlir::Attribute constant) : variant_(constant) {}

  bool is_constant() const { return variant_.index() == 1; }
  bool is_value() const { return !is_constant(); }

  const ValueAccess &value() const { return std::get<ValueAccess>(variant_); }
  mlir::Attribute constant() const {
    return std::get<mlir::Attribute>(variant_);
  }

  // Returns a value accessed through the given mapping: if this is a value,
  // composes mapping with the value access mapping.
  ValueOrConstant Map(MappingAttr mapping) const;

 private:
  std::variant<ValueAccess, mlir::Attribute> variant_;
};

// Verifies a `SairOp`.
mlir::LogicalResult VerifySairOp(mlir::Operation *op);

// Verifies a `ComputeOp`.
mlir::LogicalResult VerifyComputeOp(mlir::Operation *op);

// Verifies a `ValueProducerOp`.
mlir::LogicalResult VerifyValueProducerOp(mlir::Operation *operation);

// Returns the Sair value accessed by the operation, along with the
// corresponding mappings.
template<typename ConcreteType>
::sair::ValueOperandRange ValueOperands(ConcreteType op) {
  auto operands = op.getOperation()
                      ->getOpOperands()
                      .drop_front(op.domain().size())
                      .take_front(op.mapping_array().size());
  return ::sair::ValueOperandRange(operands);
}

class SairOp;

// Sets the mapping at the given position.
void SetMapping(SairOp op, int position, ::sair::MappingAttr mapping);

// Indicates if the Sair operation has exactly one instance and no copy.
bool HasExactlyOneInstance(SairOp op);

using namespace mlir;  // NOLINT
#include "sair_op_interfaces.h.inc"

// Abstraction around either a ComputeOp or a copy of a Sair value specified by
// the `copies` attribute of a ValueProducerOp.
class ComputeOpInstance {
 public:
  // Creates an operation instance that points to a ComputeOp.
  ComputeOpInstance(ComputeOp op, int index) : op_(op), index_(index) {}
  // Creates an operation instance that points to a value copy.
  ComputeOpInstance(ValueProducerOp op, int result, int index)
      : op_(op), result_(result), index_(index) {}

  // Indicates if the instance is the copy of a value.
  bool is_copy() { return op_.is<ValueProducerOp>(); }
  // Indicates if the instance is a duplicate of compute op.
  bool is_duplicate() { return op_.is<ComputeOp>(); }

  // Returns the ComputeOp pointed to by the instance. Fails if the instance is
  // a copy.
  ComputeOp GetComputeOp() { return op_.get<ComputeOp>(); }

  // Returns lowering decisions for the operation instance.
  DecisionsAttr GetDecisions();

  // Sets lowering decisions for the operation instance.
  void SetDecisions(DecisionsAttr decisions);

  // Emits an error at the location of the operation instance.
  mlir::InFlightDiagnostic EmitError();

 private:
  // Points to a ComputeOp if the operation is an actual operation. Point to a
  // ValueProducerOp if the operation is a copy.
  //
  // Internally, the pointer union is a tagged union. In practice, this means
  // that the union will return the type with which it was created, even if op_
  // is both a ComputeOp and a ValueProducerOp.
  llvm::PointerUnion<ComputeOp, ValueProducerOp> op_;
  int result_;
  int index_;
};

}  // namespace sair

// Allow using traits in pointer union.
namespace llvm {

template <>
struct llvm::PointerLikeTypeTraits<sair::ComputeOp> {
  static inline void *getAsVoidPointer(sair::ComputeOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline sair::ComputeOp getFromVoidPointer(void *p) {
    return sair::ComputeOp::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable =
      llvm::PointerLikeTypeTraits<Operation *>::NumLowBitsAvailable;
};

template <>
struct llvm::PointerLikeTypeTraits<sair::ValueProducerOp> {
  static inline void *getAsVoidPointer(sair::ValueProducerOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline sair::ValueProducerOp getFromVoidPointer(void *p) {
    return sair::ValueProducerOp::getFromOpaquePointer(p);
  }
  static constexpr int NumLowBitsAvailable =
      llvm::PointerLikeTypeTraits<Operation *>::NumLowBitsAvailable;
};

}  // namespace llvm

#endif  // SAIR_SAIR_OP_INTERFACES_H_
