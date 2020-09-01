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

#ifndef SAIR_SAIR_TRAITS_H_
#define SAIR_SAIR_TRAITS_H_

#include <algorithm>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
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
  mlir::Value get() const { return operand_->get(); }
  // Returns the type of the value referenced by the operand.
  ValueType GetType() const {
    return operand_->get().getType().cast<ValueType>();
  }
  // Returns the operation owning the operand.
  mlir::Operation *getOwner() { return operand_->getOwner(); }
  // Returns the access pattern associated to the operand.
  AccessPatternAttr AccessPattern();

  // Sets the value referenced by the operand.
  void set_value(mlir::Value value) { operand_->set(value); }
  // Sets the access pattern associated to the operand.
  void SetAccessPattern(AccessPatternAttr access_pattern);

  // Returns a mask of dimensions that must execute after the operand is
  // computed.
  llvm::SmallBitVector DimsDependingOnOperand() const;

  // Indicates if the operand can be used before it is defined.
  bool AllowUseBeforeDef() const;

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

// Verifies that the given Sair operation has allowed ancestors.
mlir::LogicalResult VerifySairOpParent(mlir::Operation *operation);

}  // namespace sair

// MLIR requires trait to be defined in the mlir::OpTrait namespace.
namespace mlir {
namespace OpTrait {
namespace sair {

// Checks the invariants of Sair operations:
// - That domain is compatible with the shape of the operation.
// - That the operands are composed of first the domain, second the Sair value
//   operands accessed through access patterns and finaly the other operands.
// - The the type of operands is compatible with the access patterns.
template <typename ConcreteType>
class SairOpTrait : public OpTrait::TraitBase<ConcreteType, SairOpTrait> {
 public:
  // Verifies that the invariants of Sair operations are satisfied.
  static mlir::LogicalResult verifyTrait(Operation *op) {
    ConcreteType sair_op(op);
    // Check that the domain has the right shape.
    if (llvm::size(sair_op.domain()) != sair_op.shape().NumDimensions()) {
      return sair_op.emitError("unexpected number of dimensions");
    }
    for (auto pair :
         llvm::zip(sair_op.domain(), sair_op.shape().Dimensions())) {
      if (std::get<0>(pair).getType() != std::get<1>(pair).type()) {
        return sair_op.emitError("unexpected dimension type");
      }
    }
    // Check that the domain is defined locally.
    for (mlir::Value dimension : sair_op.domain()) {
      mlir::Operation *defining_op = dimension.getDefiningOp();
      if (!defining_op ||
          defining_op->getParentRegion() != op->getParentRegion()) {
        return op->emitError()
               << "sair dimensions must be defined in the region they are used";
      }
      if (!defining_op->isBeforeInBlock(op)) {
        return (op->emitError() << "dimension used before its definition")
                   .attachNote(defining_op->getLoc())
               << "definition here";
      }
    }
    // Check that operands start with the domain.
    if (!sair_op.domain().empty() &&
        sair_op.domain().begin() != op->operand_begin()) {
      return sair_op.emitError()
             << "expected operands to start with the domain";
    }
    // Check that there is enough operands.
    int min_num_operands =
        sair_op.shape().NumDimensions() + sair_op.access_pattern_array().size();
    if (op->getNumOperands() < min_num_operands) {
      return sair_op.emitError() << "unexpected number of operands";
    }

    if (!sair_op.ValueOperands().empty()) {
      // Verify that the "access_pattern_array" attribute exists.
      if (!op->getAttr(::sair::SairDialect::kAccessPatternAttrName)) {
        return mlir::emitError(op->getLoc())
               << "missing " << ::sair::SairDialect::kAccessPatternAttrName
               << " attribute";
      }
      for (mlir::Attribute pattern : sair_op.access_pattern_array()) {
        if (pattern.cast<::sair::AccessPatternAttr>().DependsOnDimension(
                ::sair::AccessPatternAttr::kNoDimension)) {
          return mlir::emitError(op->getLoc())
                 << "all dimensions of the accessed domain must be mapped";
        }
      }
    }

    // Check !sair.value operands.
    for (::sair::ValueOperand v : sair_op.ValueOperands()) {
      auto value_type =
          v.get().getType().template dyn_cast<::sair::ValueType>();
      if (!value_type) {
        return mlir::emitError(v.get().getLoc())
               << "expected a !sair.value operand";
      }
      if (v.AccessPattern().UseDomainSize() != sair_op.domain().size()) {
        return mlir::emitError(op->getLoc()) << "invalid use domain size";
      }
      ::sair::DomainShapeAttr expected_shape =
          sair_op.shape().Inverse(v.AccessPattern());
      if (expected_shape != value_type.Shape()) {
        return mlir::emitError(v.get().getLoc())
               << "access pattern incompatible with the operand shape";
      }
      mlir::Operation *defining_op = v.get().getDefiningOp();
      if (!defining_op ||
          defining_op->getParentRegion() != op->getParentRegion()) {
        return op->emitError()
               << "sair values must be defined in the region they are used";
      }
      if (!defining_op->isBeforeInBlock(op) && !v.AllowUseBeforeDef()) {
        return (op->emitError() << "operand used before its definition")
                   .attachNote(defining_op->getLoc())
               << "definition here";
      }

      llvm::SmallBitVector dependency_mask = v.AccessPattern().DependencyMask();
      if (dependency_mask.anyCommon(v.DimsDependingOnOperand())) {
        return op->emitError() << "an operand access pattern references a "
                                  "dimension that depends on the operand";
      }
    }

    // Check that returned Sair values have the right shape.
    ::sair::DomainShapeAttr results_shape =
        sair_op.shape().Prefix(sair_op.results_rank());
    for (mlir::Value result : op->getResults()) {
      ::sair::SairShapedType type =
          result.getType().dyn_cast<::sair::SairShapedType>();
      if (type == nullptr) continue;
      if (type.Shape() != results_shape) {
        return op->emitError() << "unexpected shape: expected " << results_shape
                               << ", got " << type.Shape();
      }
    }

    return ::sair::VerifySairOpParent(sair_op);
  }

  // Returns the access pattern of the !sair.value operand at the given
  // position.
  ::sair::AccessPatternAttr AccessPattern(int pos) {
    ConcreteType op(this->getOperation());
    return op.access_pattern_array()
        .getValue()[pos]
        .template cast<::sair::AccessPatternAttr>();
  }

  // Returns the Sair value accessed by the operation, along with the
  // corresponding access patterns.
  ::sair::ValueOperandRange ValueOperands() {
    ConcreteType op(this->getOperation());
    auto operands = this->getOperation()
                        ->getOpOperands()
                        .drop_front(op.domain().size())
                        .take_front(op.access_pattern_array().size());
    return ::sair::ValueOperandRange(operands);
  }

  // Sets the access pattern at the given position.
  void SetAccessPattern(int position,
                        ::sair::AccessPatternAttr access_pattern) {
    ConcreteType op(this->getOperation());
    llvm::SmallVector<mlir::Attribute, 4> new_array =
        llvm::to_vector<4>(op.access_pattern_array());
    new_array[position] = access_pattern;
    mlir::ArrayAttr new_attr = mlir::ArrayAttr::get(new_array, op.getContext());
    op.setAttr(::sair::SairDialect::kAccessPatternAttrName, new_attr);
  }
};

// Trait for Sair operations that return a single Sair value or iteration
// dimension.
template <typename ConcreteType>
class OneResult : public OpTrait::TraitBase<ConcreteType, OneResult> {
 public:
  // Verifies that the operation has a single result, of type !sair.value. This
  // is a hook for the Mlir trait system.
  static mlir::LogicalResult verifyTrait(Operation *op) {
    if (op->getNumResults() != 1) {
      return op->emitError() << "requires one result";
    }
    if (!op->getResult(0).getType().isa<::sair::SairShapedType>()) {
      return op->emitError() << "requires a !sair.value return type";
    }
    return success();
  }

  // Returns the shape of the operation.
  ::sair::DomainShapeAttr shape() {
    return this->getOperation()
        ->getResult(0)
        .getType()
        .template cast<::sair::SairShapedType>()
        .Shape();
  }
};

}  // namespace sair
}  // namespace OpTrait
}  // namespace mlir

#endif  // SAIR_SAIR_TRAITS_H_
