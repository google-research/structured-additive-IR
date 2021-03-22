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
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_types.h"
#include "util.h"

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
  ValueOperand(mlir::OpOperand *operand, int index)
      : operand_(operand), index_(index) {}

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

class IterationSpaceAnalysis;

// Describes how a value is stored. Attributes may be null if the buffer is not
// yet specified.
class ValueStorage {
 public:
  ValueStorage() {}
  ValueStorage(mlir::StringAttr space, mlir::StringAttr buffer_name,
               MappingAttr layout)
      : space_(space), buffer_name_(buffer_name), layout_(layout) {}

  // Memory space the value is stored in. May be null if not yet specified.
  mlir::StringAttr space() const { return space_; }

  // Name of the buffer where the value is stored, if specified.
  mlir::StringAttr buffer_name() const { return buffer_name_; }

  // Mapping from the iteration space of the value to buffer dimensions.
  MappingAttr layout() const { return layout_; }

  // Returns the value storage as seen through an operand.
  ValueStorage Map(const ValueOperand &operand,
                   const IterationSpaceAnalysis &iteration_spaces) const;

 private:
  mlir::StringAttr space_;
  mlir::StringAttr buffer_name_;
  MappingAttr layout_;
};

bool operator==(const ValueStorage &lhs, const ValueStorage &rhs);
bool operator!=(const ValueStorage &lhs, const ValueStorage &rhs);

// Verifies a `SairOp`.
mlir::LogicalResult VerifySairOp(mlir::Operation *op);

// Verifies a `ComputeOp`.
mlir::LogicalResult VerifyComputeOp(mlir::Operation *op);

// Verifies a `RangeOp`.
mlir::LogicalResult VerifyRangeOp(mlir::Operation *op);

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

// Sets the mapping at the given position.
template <typename ConcreteType>
void SetMapping(ConcreteType op, int position, ::sair::MappingAttr mapping) {
  llvm::SmallVector<mlir::Attribute, 4> new_array =
      llvm::to_vector<4>(op.mapping_array());
  new_array[position] = mapping;
  mlir::ArrayAttr new_attr = mlir::ArrayAttr::get(op.getContext(), new_array);
  op->setAttr(::sair::SairDialect::kMappingAttrName, new_attr);
}

using namespace mlir;  // NOLINT
#include "sair_op_interfaces.h.inc"

}  // namespace sair

#endif  // SAIR_SAIR_OP_INTERFACES_H_
