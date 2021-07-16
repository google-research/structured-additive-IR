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

#include "sair_op_interfaces.h"

#include <iterator>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "loop_nest.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "storage.h"

namespace sair {

mlir::Type ValueAccess::ElementType() const {
  return value.getType().cast<ValueType>().ElementType();
}

bool operator==(const ValueAccess &lhs, const ValueAccess &rhs) {
  return lhs.value == rhs.value && lhs.mapping == rhs.mapping;
}

bool operator!=(const ValueAccess &lhs, const ValueAccess &rhs) {
  return !(lhs == rhs);
}

ValueOperand::ValueOperand(mlir::OpOperand *operand) : operand_(operand) {
  auto owner = cast<SairOp>(operand->getOwner());
  index_ = operand->getOperandNumber() - owner.domain().size();
  assert(index_ >= 0 && "expected domain operands before value operands");
}

MappingAttr ValueOperand::Mapping() const {
  return cast<SairOp>(operand_->getOwner())
      .mapping_array()
      .getValue()[index_]
      .template cast<::sair::MappingAttr>();
}

void ValueOperand::SubstituteValue(ValueAccess new_value) {
  set_value(new_value.value);
  SetMapping(Mapping().Compose(new_value.mapping));
}

void ValueOperand::SetMapping(MappingAttr mapping) {
  SairOp op = cast<SairOp>(operand_->getOwner());
  op.SetMapping(index_, mapping);
}

ValueOperandRange::ValueOperandRange()
    : RangeBaseT(std::make_pair(nullptr, 0), 0) {}

ValueOperandRange::ValueOperandRange(
    llvm::MutableArrayRef<mlir::OpOperand> operands)
    : RangeBaseT(std::make_pair(operands.data(), 0), operands.size()) {}

ValueOperandRange::PtrPair ValueOperandRange::offset_base(PtrPair base_ptr,
                                                          ptrdiff_t offset) {
  base_ptr.first += offset;
  base_ptr.second += offset;
  return base_ptr;
}

ValueOperand ValueOperandRange::dereference_iterator(PtrPair base_ptr,
                                                     ptrdiff_t offset) {
  return ValueOperand(base_ptr.first + offset);
}

llvm::SmallBitVector ValueOperand::DependingDims() const {
  return cast<SairOp>(operand_->getOwner()).DimsDependingOnOperand(index_);
}

bool ValueOperand::AllowUseBeforeDef() const {
  return cast<SairOp>(operand_->getOwner()).AllowUseBeforeDef(index_);
}

llvm::SmallBitVector ValueOperand::CarryingDims() const {
  return cast<SairOp>(operand_->getOwner()).CarryingDimensions(index_);
}

void UpdateValueUses(mlir::Value value, ValueAccess new_value) {
  for (OpOperand &operand : llvm::make_early_inc_range(value.getUses())) {
    ValueOperand(&operand).SubstituteValue(new_value);
  }
}

ValueOrConstant ValueOrConstant::Map(MappingAttr mapping) const {
  if (is_constant()) return *this;
  ValueAccess value_access = value();
  value_access.mapping = mapping.Compose(value_access.mapping);
  return value_access;
}

mlir::LogicalResult VerifySairOp(Operation *op) {
  SairOp sair_op = cast<SairOp>(op);

  // Sair operations are only allowed inside a SairProgramOp.
  auto program = dyn_cast<SairProgramOp>(op->getParentOp());
  if (program == nullptr) {
    return op->emitOpError() << "expected to be immediately contained in a '"
                             << SairProgramOp::getOperationName() << "'";
  }

  // Assert that the domain has the right shape.
  assert(llvm::size(sair_op.domain()) == sair_op.shape().NumDimensions());
#ifndef NDEBUG
  for (auto pair :
       llvm::zip(sair_op.domain(), sair_op.shape().Dimensions())) {
    assert(std::get<0>(pair).getType() == std::get<1>(pair).type());
  }
#endif

  // Assert that operands start with the domain.
  assert(sair_op.domain().empty() ||
         sair_op.domain().begin() == op->operand_begin());

  // Check that the domain is defined locally.
  for (mlir::Value dimension : sair_op.domain()) {
    mlir::Operation *defining_op = dimension.getDefiningOp();
    if (defining_op == nullptr || defining_op->getParentOp() != program) {
      return op->emitError()
             << "sair dimensions must be defined in the region they are used";
    }
  }

  if (!sair_op.ValueOperands().empty()) {
    // Verify that the "mapping_array" attribute exists.
    if (!op->getAttr(SairOp::kMappingAttrName)) {
      return mlir::emitError(op->getLoc())
             << "missing " << SairOp::kMappingAttrName << " attribute";
    }
    for (mlir::Attribute attr : sair_op.mapping_array()) {
      MappingAttr mapping = attr.cast<MappingAttr>();
      if (mapping.HasNoneExprs() || mapping.HasUnknownExprs()) {
        return mlir::emitError(op->getLoc())
               << "all dimensions of the accessed domain must be mapped";
      }
    }
  }

  // Check !sair.value operands.
  for (::sair::ValueOperand v : sair_op.ValueOperands()) {

    // Verify operands of Sair operands are defined in the same program.
    mlir::Operation *defining_op = v.value().getDefiningOp();
    if (defining_op == nullptr || defining_op->getParentOp() != program) {
      return op->emitError()
             << "sair values must be defined in the region they are used";
    }

    if (v.Mapping().UseDomainSize() != sair_op.domain().size()) {
      return mlir::emitError(op->getLoc()) << "invalid use domain size";
    }

    AttrLocation mapping_loc(op->getLoc(), "operand mapping");
    if (mlir::failed(
            VerifyMappingShape(mapping_loc, v.Mapping(), sair_op.shape()))) {
      return mlir::failure();
    }

    auto expected_shape = sair_op.shape().AccessedShape(v.Mapping());
    auto given_shape =
        v.value().getType().template cast<::sair::ValueType>().Shape();
    if (expected_shape != given_shape) {
      return op->emitError() << "invalid operand shape: expected "
                             << expected_shape << ", got " << given_shape;
    }

    llvm::SmallBitVector dependency_mask = v.Mapping().DependencyMask();
    if (dependency_mask.anyCommon(v.DependingDims())) {
      return op->emitError() << "an operand mapping references a "
                                "dimension that depends on the operand";
    }
  }

  // Check that returned Sair values have the right shape.
  ::sair::DomainShapeAttr results_shape =
      sair_op.shape().Prefix(sair_op.results_rank());
  for (mlir::Value result : op->getResults()) {
    auto type = result.getType().cast<ShapedType>();
    if (type.Shape() != results_shape) {
      return op->emitError() << "unexpected shape: expected " << results_shape
                             << ", got " << type.Shape();
    }
  }

  if (!isa<ComputeOp>(sair_op.getOperation())) {
    if (sair_op->hasAttr(ComputeOp::kDecisionsAttrName)) {
      return op->emitError() << "only compute Sair ops can have the '"
                             << ComputeOp::kDecisionsAttrName << "' attribute";
    }
  }

  return ::mlir::success();
}

// Verifies that `decisions` is well formed when used for an operation with the
// given shape and result types.
static mlir::LogicalResult VerifyDecisionsWellFormed(mlir::Location loc,
                                                     DomainShapeAttr shape,
                                                     TypeRange result_types,
                                                     DecisionsAttr decisions) {
  auto *sair_dialect = shape.getContext()->getLoadedDialect<SairDialect>();

  // Check loop nest.
  mlir::ArrayAttr loop_nest = decisions.loop_nest();
  llvm::DenseSet<mlir::Attribute> loop_names;
  if (loop_nest != nullptr) {
    if (mlir::failed(
            VerifyLoopNestWellFormed(loc, shape, loop_nest.getValue()))) {
      return mlir::failure();
    }
    loop_names.reserve(loop_nest.size());
    for (mlir::Attribute attr : loop_nest.getValue()) {
      loop_names.insert(attr.cast<LoopAttr>().name());
    }
  }

  // Check storage.
  mlir::ArrayAttr storage = decisions.storage();
  if (storage != nullptr &&
      mlir::failed(VerifyStorageAttrWellFormed(
          loc, sair_dialect, result_types, loop_names, storage.getValue()))) {
    return mlir::failure();
  }

  // Check expansion pattern.
  mlir::StringAttr pattern_name = decisions.expansion();
  if (pattern_name == nullptr) return mlir::success();
  const ExpansionPattern *pattern =
      sair_dialect->GetExpansionPattern(pattern_name.getValue());
  if (pattern == nullptr) {
    return mlir::emitError(loc)
           << "invalid expansion pattern name " << pattern_name;
  }
  return mlir::success();
}

mlir::LogicalResult VerifyComputeOp(mlir::Operation *operation) {
  auto op = llvm::cast<ComputeOp>(operation);
  auto sair_op = llvm::cast<SairOp>(operation);
  return VerifyDecisionsWellFormed(op.getLoc(), sair_op.shape(),
                                   op->getResultTypes(), op.GetDecisions());
}

mlir::LogicalResult VerifyValueProducerOp(mlir::Operation *operation) {
  ValueProducerOp op(operation);
  SairOp sair_op(operation);
  auto copies = operation->getAttrOfType<mlir::ArrayAttr>(
      ValueProducerOp::kCopiesAttrName);
  if (copies == nullptr) return mlir::success();
  if (copies.size() != operation->getNumResults()) {
    return op.emitError()
           << "the `copies` attribute must have one entry per operation result";
  }

  DomainShapeAttr shape = sair_op.shape().Prefix(sair_op.results_rank());
  for (int i = 0, e = op->getNumResults(); i < e; ++i) {
    for (mlir::Attribute attr : op.GetCopies(i)) {
      auto decisions = attr.cast<DecisionsAttr>();
      if (mlir::failed(VerifyDecisionsWellFormed(
              op.getLoc(), shape, {op->getResultTypes()[i]}, decisions))) {
        return mlir::failure();
      }
    }
  }
  return mlir::success();
}

void SetMapping(SairOp op, int position, ::sair::MappingAttr mapping) {
  llvm::SmallVector<mlir::Attribute, 4> new_array =
      llvm::to_vector<4>(op.mapping_array());
  new_array[position] = mapping;
  mlir::ArrayAttr new_attr = mlir::ArrayAttr::get(op.getContext(), new_array);
  op->setAttr(SairOp::kMappingAttrName, new_attr);
}

#include "sair_op_interfaces.cc.inc"

}  // namespace sair
