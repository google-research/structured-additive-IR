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
class SairPlaceholderOp;

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

class SairProgramOp;
class SairDialect;
class ResultInstance;
class OperandInstance;
class ValueAccessInstance;
class ComputeOpInstance;

// A SairOp that will appear in the code after copies and operation instances
// are materialized.
class OpInstance {
 public:
  OpInstance() {}
  OpInstance(nullptr_t) : op_(nullptr) {}

  // Create an instance of a SairOp. The SairOp must have a single instance.
  // TODO(ulysse): support having multiple instance of non-compute operations.
  explicit OpInstance(SairOp op);

  // Creates an OpInstance representing the unique instance of `op`. Fails
  // if `op` has more than one instance. Unlike the function above, this is not
  // meant to evlove once we support multiple instance of non-compute
  // operations.
  static OpInstance Unique(SairOp op);

  // Indicates if the instance is the copy of a value.
  bool is_copy() const { return op_.is<ValueProducerOp>(); }
  // Indicates if the instance is a duplicate of compute op.
  bool is_duplicate() const { return op_.is<SairOp>(); }

  // OpInstance converts to true if it is != nullptr.
  operator bool() const;

  // Returns the duplicated SairOp. `is_duplicate` must be true.
  mlir::Operation *GetDuplicatedOp() const;

  // Returns the value copied by the operation. `is_copy` must be true.
  mlir::Value GetCopiedValue() const;

  // Returns the operation that defines the instance.
  mlir::Operation *getOperation() const;
  SairOp GetSairOp() const;

  // Provides a hash for the instance.
  unsigned HashValue() const;

  // Emits an error at the location of the operation instance.
  mlir::InFlightDiagnostic EmitError() const;

  // Attach a note at the location of the operation instance.
  mlir::Diagnostic &AttachNote(mlir::InFlightDiagnostic &diag) const;

  // Returns the location of the original operation. EmitError and AttachNote
  // should be prefered as they provide more precise error messages.
  mlir::Location getLoc() const;

  // Returns the underlying operation context. The instance must not be null.
  mlir::MLIRContext *context() const;

  // Returns the Sair program this operation belongs to.
  SairProgramOp program() const;

  // Returns a pointer to the Sair dialect.
  SairDialect *GetSairDialect() const;

  // Returns the Shape of the operation.
  DomainShapeAttr GetShape() const;

  // Returns the rank of the operation domain.
  int domain_size() const;

  // Returns the i-th dimension of the operation.
  ResultInstance domain(int i) const;

  // Returns the domain of the operation as an iterator of ResultInstance.
  auto domain() const;

  // Returns the domain of the operation as an iterator of ValueAccessInstance.
  auto DomainWithDependencies() const;

  // Returns the operand of the operation at the given position.
  OperandInstance Operand(int position) const;

  // Returns the operands of the operation as a range of OperandInstance.
  auto Operands() const;

  // Number of results for the operation as a range of ResultInstance.
  int num_results() const;

  // Returns the result of the operation at the given position.
  ResultInstance Result(int result) const;

  // Returns the results of the operation.
  auto Results() const;

  // Size of each sub-domain of the operation.
  llvm::SmallVector<int> SubDomains() const;

  // Returns a mask of the dimensions that must exit before using the result.
  llvm::SmallBitVector ResultsDimDependencies() const;

  // LLVM-style RTTI infrastructure.
  static bool classof(const OpInstance &op) { return true; }

  template <typename U>
  bool isa() const {
    assert(!op_.isNull() && "isa<> used on a null OpInstance");
    return U::classof(*this);
  }

  template <typename U>
  U dyn_cast() const {
    return isa<U>() ? U(*this) : U();
  }

  template <typename U>
  U dyn_cast_or_null() const {
    return op_.isNull() ? U() : dyn_cast<U>();
  }

  template <typename U>
  U cast() const {
    assert(isa<U>());
    return U(*this);
  }

 protected:
  friend ResultInstance;

  OpInstance(llvm::PointerUnion<SairOp, ValueProducerOp> op, int result,
             int index)
      : op_(op), index_(index), result_(result) {}

  // If the instance is a copy of a value, returns the operation producing the
  // value.
  ValueProducerOp GetValueProducer() const;

  // Index of the instance or index of the copy.
  int index() const { return index_; }

  // In the case where the instance is a copy, index of the result the copy
  // applies to.
  int result() const { return result_; }

 private:
  friend bool operator==(const OpInstance &lhs, const OpInstance &rhs);
  friend llvm::DenseMapInfo<ComputeOpInstance>;

  // Returns the operation domain as MLIR values.
  ValueRange GetDomainValues() const;

  // An instance is either a duplicate of an existing SairOp or sair.copy
  // operation that will be introduced to copy a value produced by a
  // ValueProducerOp.
  //
  // Internally, the pointer union is a tagged union. In practice, this means
  // that the union will return the type with which it was created, even if op_
  // is both a ComputeOp and a ValueProducerOp.
  llvm::PointerUnion<SairOp, ValueProducerOp> op_;

  int index_ = 0;
  int result_ = 0;
};

bool operator==(const OpInstance &lhs, const OpInstance &rhs);
bool operator!=(const OpInstance &lhs, const OpInstance &rhs);

// Abstraction around either a ComputeOp or a copy of a Sair value specified by
// the `copies` attribute of a ValueProducerOp.
class ComputeOpInstance : public OpInstance {
 public:
  // Converts `op` into a ComputeOpInstance.
  ComputeOpInstance(const OpInstance &op);
  // Creates an null instance.
  ComputeOpInstance() {}
  // Creates an operation instance that points to a ComputeOp.
  ComputeOpInstance(ComputeOp op, int index)
      : OpInstance(llvm::cast<SairOp>(op.getOperation()), 0, index) {}
  // Creates an operation instance that points to a value copy.
  ComputeOpInstance(ValueProducerOp op, int result, int index)
      : OpInstance(op, result, index) {
    assert(op != nullptr);
  }

  // Creates a ComputeOpInstance representing the unique instance of `op`. Fails
  // if `op` has more than one instance.
  static ComputeOpInstance Unique(ComputeOp op);

  // Creates a ComputeOpInstance that does not represent any operation and that
  // will only be equal to other markers created with the same id.
  //
  // This is a static method rather than a public constructor in order to make
  // clear that this create a special instance of the class, that should not be
  // used to represent an operation but only when a marker is needed, such as in
  // llvm::DenseMapInfo.
  static ComputeOpInstance Marker(int id) { return ComputeOpInstance(id); }

  // Returns lowering decisions for the operation instance.
  DecisionsAttr GetDecisions() const;

  // Sets lowering decisions for the operation instance.
  void SetDecisions(DecisionsAttr decisions);

  // Returns the loop nest of the operation. Returns an empty array if the loop
  // nest is unspecified.
  llvm::ArrayRef<mlir::Attribute> Loops() const;

  // Sets the loop_nest field of decisions.
  void SetLoopNest(mlir::ArrayAttr loop_nest);

  // Returns storage information for the i-th result of the operation.
  BufferAttr Storage(int result) const;

  // Set storage information for the given result.
  void SetStorage(int result, BufferAttr storage);

  // LLVM-style RTTI infrastructure.
  static bool classof(const OpInstance &op) {
    return op.is_copy() || llvm::isa<ComputeOp>(op.getOperation());
  }

 private:
  // Internal constructor for this::Marker.
  ComputeOpInstance(int id) : OpInstance(nullptr, 0, id) {}

  // Returns the ComputeOp pointed to by the instance. Fails if the instance is
  // a copy.
  ComputeOp GetComputeOp() const;
};

// An value produced by an instance of a Sair operation. This can either be a
// dimension or a Sair value.
class ResultInstance {
 public:
  ResultInstance(OpInstance op, int result) : op_(op), result_(result) {}

  // Create a ResultInstance representing the unique instance of value. Fails if
  // value has zero or more than one instance.
  static ResultInstance Unique(mlir::Value value);

  // Returns the instance of the operation that defines the result.
  const OpInstance &defining_op() const { return op_; }

  // Returns the position of the result.
  int result_number() const { return result_; }

  // Returns the value type.
  ShapedType GetType() const;

  // Returns the original value.
  mlir::Value GetValue() const;

  // Returns operations that use the instance, along with the position of the
  // operand that use the instance. The position is the MLIR position, not the
  // position in Sair value operands. It includes domain operands.
  llvm::SmallVector<std::pair<OpInstance, int>> GetUses() const;

  // Provides a hash for the instance.
  unsigned HashValue() const;

 private:
  friend bool operator==(const ResultInstance &lhs, const ResultInstance &rhs);

  OpInstance op_;
  int result_;
};

bool operator==(const ResultInstance &lhs, const ResultInstance &rhs);
bool operator!=(const ResultInstance &lhs, const ResultInstance &rhs);

// An instance of a Sair value accessed by a mapping.
struct ValueAccessInstance {
  ResultInstance value;
  MappingAttr mapping;
};

// A !sair.value operand of an OpInstance.
class OperandInstance {
 public:
  // Creates the operand of `op` at the given position. The position ignore
  // non-!sair.value operands of the operation (such as the domain of the
  // operation).
  OperandInstance(OpInstance op, int operand_position)
      : op_(op), operand_position_(operand_position) {}

  // Creates the operand of `op` that corresponds to `operand` in the
  // original operation.
  OperandInstance(ValueOperand operand, OpInstance op)
      : OperandInstance(op, operand.position()) {}

  // Returns the operation that owns the operand.
  OpInstance owner() const { return op_; }

  // Returns the accessed value. Returns std::nullopt if the result instance is
  // not yet specified.
  std::optional<ResultInstance> GetValue() const;

  // Returns the mapping used to access the value.
  MappingAttr Mapping() const;

  // Returns the value access performed by the operand.
  std::optional<ValueAccessInstance> Get() const;

  // Returns a mask of dimensions that must execute after the operand is
  // computed.
  llvm::SmallBitVector DependingDims() const;

  // Indicates if the operand can be used before it is defined.
  bool AllowUseBeforeDef() const;

  // If the operand is a loop-carried dependency, indicates along which
  // dimensions it is carried.
  llvm::SmallBitVector CarryingDims() const;

 private:
  OpInstance op_;
  int operand_position_;

  // Returns the original operand. Fails if op_ is a copy of a value.
  ValueOperand GetOriginalOperand() const;
};

inline auto OpInstance::domain() const {
  return llvm::map_range(GetDomainValues(), [](mlir::Value v) {
    OpInstance dim_op = OpInstance(llvm::cast<SairOp>(v.getDefiningOp()));
    return ResultInstance(dim_op, v.cast<OpResult>().getResultNumber());
  });
}

inline auto OpInstance::DomainWithDependencies() const {
  DomainShapeAttr shape = GetShape();
  return llvm::map_range(llvm::enumerate(GetDomainValues()), [=](auto p) {
    auto value = p.value().template cast<mlir::OpResult>();
    OpInstance dim_op = OpInstance(llvm::cast<SairOp>(value.getOwner()));
    ValueAccessInstance access = {
        .value = ResultInstance(dim_op, value.getResultNumber()),
        .mapping = shape.Dimension(p.index()).dependency_mapping()};
    return access;
  });
}

inline auto OpInstance::Operands() const {
  int num_operands =
      is_copy() ? 1
                : llvm::cast<SairOp>(GetDuplicatedOp()).ValueOperands().size();
  return llvm::map_range(llvm::seq<int>(0, num_operands),
                         [this](int i) { return OperandInstance(*this, i); });
}

inline auto OpInstance::Results() const {
  return llvm::map_range(llvm::seq<int>(0, num_results()),
                         [this](int i) { return ResultInstance(*this, i); });
}

}  // namespace sair

// Allow using traits in pointer union.
namespace llvm {

template <>
struct llvm::PointerLikeTypeTraits<sair::SairOp> {
  static inline void *getAsVoidPointer(sair::SairOp val) {
    return const_cast<void *>(val.getAsOpaquePointer());
  }
  static inline sair::SairOp getFromVoidPointer(void *p) {
    return sair::SairOp::getFromOpaquePointer(p);
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

// Allow using ComputeOpInstance in llvm::DenseMap.
template <>
struct DenseMapInfo<sair::ComputeOpInstance> {
  static sair::ComputeOpInstance getEmptyKey() {
    return sair::ComputeOpInstance::Marker(1);
  }

  static sair::ComputeOpInstance getTombstoneKey() {
    return sair::ComputeOpInstance::Marker(2);
  }

  static unsigned getHashValue(const sair::ComputeOpInstance &op) {
    return op.HashValue();
  }

  static bool isEqual(const sair::ComputeOpInstance &lhs,
                      const sair::ComputeOpInstance &rhs) {
    return lhs == rhs;
  }
};

// Allow using OpInstance in llvm::DenseMap.
template <>
struct DenseMapInfo<sair::OpInstance> {
  static sair::OpInstance getEmptyKey() {
    return DenseMapInfo<sair::ComputeOpInstance>::getEmptyKey();
  }

  static sair::OpInstance getTombstoneKey() {
    return DenseMapInfo<sair::ComputeOpInstance>::getTombstoneKey();
  }

  static unsigned getHashValue(const sair::OpInstance &op) {
    return op.HashValue();
  }

  static bool isEqual(const sair::OpInstance &lhs,
                      const sair::OpInstance &rhs) {
    return lhs == rhs;
  }
};

// Allow using ResultInstance in llvm::DenseMap.
template <>
struct DenseMapInfo<sair::ResultInstance> {
  static sair::ResultInstance getEmptyKey() {
    auto op = DenseMapInfo<sair::OpInstance>::getEmptyKey();
    return sair::ResultInstance(op, 0);
  }

  static sair::ResultInstance getTombstoneKey() {
    auto op = DenseMapInfo<sair::OpInstance>::getTombstoneKey();
    return sair::ResultInstance(op, 0);
  }

  static unsigned getHashValue(const sair::ResultInstance &value) {
    return value.HashValue();
  }

  static bool isEqual(const sair::ResultInstance &lhs,
                      const sair::ResultInstance &rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // SAIR_SAIR_OP_INTERFACES_H_
