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
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "loop_nest.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_ops.h"
#include "sair_traits.h"
#include "sair_types.h"
#include "utils.h"

namespace sair {

AccessPatternAttr ValueOperand::AccessPattern() {
  return cast<SairOp>(operand_->getOwner()).access_pattern_array()
      .getValue()[index_]
      .template cast<::sair::AccessPatternAttr>();
}

void ValueOperand::SetAccessPattern(AccessPatternAttr access_pattern) {
  SairOp op = cast<SairOp>(operand_->getOwner());
  op.SetAccessPattern(index_, access_pattern);
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
  return ValueOperand(base_ptr.first + offset, base_ptr.second + offset);
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

// Sair operations are only allowed inside a SairProgramOp.
mlir::LogicalResult VerifySairOpParent(mlir::Operation *operation) {
  if (isa<SairProgramOp>(operation->getParentOp())) {
    return mlir::success();
  }

  return operation->emitOpError()
         << "expected to be immediately contained in a '"
         << SairProgramOp::getOperationName() << "'";
}

llvm::Optional<int> GetMemorySpace(int result, mlir::Operation *op) {
  llvm::Optional<mlir::ArrayAttr> array =
      cast<ValueProducerOp>(op).memory_space();
  if (!array.hasValue()) return llvm::None;
  mlir::IntegerAttr space = array.getValue()[result].dyn_cast<IntegerAttr>();
  if (space == nullptr) return llvm::None;
  return space.getInt();
}

llvm::Optional<int> GetMemorySpace(mlir::Value value) {
  assert(value.getType().isa<ValueType>());
  // Sair requires !sair.value operands to be defined by an operation in the
  // same block, ensuring that value.getDefiningOp() is well defined.
  ValueProducerOp producer = cast<ValueProducerOp>(value.getDefiningOp());
  for (int i = 0, e = producer.getOperation()->getNumResults(); i < e; ++i) {
    if (producer.getOperation()->getResult(i) == value) {
      return producer.GetMemorySpace(i);
    }
  }
  llvm_unreachable("value not found in the defining operation");
}

void SetMemorySpace(int result, llvm::Optional<int> memory_space,
                    mlir::Operation *op) {
  auto old_attribute =
      op->getAttrOfType<mlir::ArrayAttr>(ValueProducerOp::kMemorySpaceAttrName);
  llvm::SmallVector<mlir::Attribute, 4> memory_spaces;
  if (old_attribute == nullptr) {
    auto unit_attr = mlir::UnitAttr::get(op->getContext());
    memory_spaces.resize(op->getNumResults(), unit_attr);
  } else {
    appendRange(memory_spaces, old_attribute.getValue());
  }
  if (memory_space.hasValue()) {
    memory_spaces[result] = mlir::IntegerAttr::get(
        mlir::IntegerType::get(64, op->getContext()), memory_space.getValue());
  } else {
    memory_spaces[result] = mlir::UnitAttr::get(op->getContext());
  }
  auto new_attribute = mlir::ArrayAttr::get(memory_spaces, op->getContext());
  op->setAttr(ValueProducerOp::kMemorySpaceAttrName, new_attribute);
}

mlir::LogicalResult VerifySairOp(Operation *op) {
  SairOp sair_op = cast<SairOp>(op);
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
      if (!pattern.cast<::sair::AccessPatternAttr>().IsFullySpecified()) {
        return mlir::emitError(op->getLoc())
               << "all dimensions of the accessed domain must be mapped";
      }
    }
  }

  // Check !sair.value operands.
  for (::sair::ValueOperand v : sair_op.ValueOperands()) {
    auto value_type = v.GetType().template dyn_cast<::sair::ValueType>();
    if (!value_type) {
      return mlir::emitError(v.value().getLoc())
             << "expected a !sair.value operand";
    }
    if (v.AccessPattern().UseDomainSize() != sair_op.domain().size()) {
      return mlir::emitError(op->getLoc()) << "invalid use domain size";
    }
    ::sair::DomainShapeAttr expected_shape =
        sair_op.shape().Inverse(v.AccessPattern());
    if (expected_shape != value_type.Shape()) {
      return mlir::emitError(v.value().getLoc())
             << "access pattern incompatible with the operand shape";
    }
    mlir::Operation *defining_op = v.value().getDefiningOp();
    if (!defining_op->isBeforeInBlock(op) && !v.AllowUseBeforeDef()) {
      return (op->emitError() << "operand used before its definition")
                 .attachNote(defining_op->getLoc())
             << "definition here";
    }

    llvm::SmallBitVector dependency_mask = v.AccessPattern().DependencyMask();
    if (dependency_mask.anyCommon(v.DependingDims())) {
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

// Returns the first loop of the loop_nest attribute of the operation, if any.
static LoopAttr FirstLoopOrNull(mlir::Operation *op) {
  ComputeOp compute_op = dyn_cast_or_null<ComputeOp>(op);
  if (compute_op == nullptr) return nullptr;
  if (!compute_op.loop_nest().hasValue()) return nullptr;
  if (compute_op.LoopNestLoops().empty()) return nullptr;
  return compute_op.LoopNestLoops().front().dyn_cast<LoopAttr>();
}


mlir::LogicalResult VerifyValueProducerOp(mlir::Operation *operation) {
  ValueProducerOp op = cast<ValueProducerOp>(operation);
  // All results must be Sair values. This is not a user-facing error. It should
  // be verified by operations implementing `SairValueProducerOp`.
  assert(llvm::all_of(operation->getResultTypes(),
                      [](mlir::Type type) { return type.isa<ValueType>(); }));
  llvm::Optional<mlir::ArrayAttr> memory_space_attr = op.memory_space();
  if (!memory_space_attr.hasValue()) return mlir::success();
  llvm::ArrayRef<mlir::Attribute> memory_spaces =
      memory_space_attr.getValue().getValue();

  if (memory_spaces.size() != operation->getNumResults()) {
    return op.emitError()
           << "wrong number of entries for the memory_space attribute";
  }

  bool needs_allocation = false;
  for (int i = 0, e = memory_spaces.size(); i < e; ++i) {
    mlir::Attribute attr = memory_spaces[i];
    if (attr.isa<mlir::UnitAttr>()) continue;

    int memory_space = attr.cast<mlir::IntegerAttr>().getInt();
    ValueType type = operation->getResult(i).getType().cast<ValueType>();
    switch (memory_space) {
      case ValueProducerOp::kMemory:
        // TODO(ulysse): support lowering index values to memory.
        if (type.ElementType().isa<mlir::IndexType>()) {
          return op.emitError()
                 << "index variables cannot be allocated in memory";
        }
        needs_allocation = true;
        break;
      case ValueProducerOp::kRegister:
        if (!type.Shape().Is0d()) {
          // TODO(ulysse): consider the dimensionality of the layout instead,
          // once layout attributes are implemented.
          return op.emitError() << "only 0D values may be stored in registers";
        }
        break;
      default:
        return op.emitError() << "unexpected memory space";
    }
  }

  // Ensure that we can introduce the malloc between the producer of dimension
  // sizes and the current op.
  // TODO(ulysse): can we fold this in the generic interface for exposing
  // dependencies?
  LoopAttr first_loop = FirstLoopOrNull(op);
  if (!needs_allocation || first_loop == nullptr) return mlir::success();

  for (mlir::Value dimension : cast<SairOp>(operation).domain()) {
    SairDynRangeOp defining_op =
        dyn_cast<SairDynRangeOp>(dimension.getDefiningOp());
    if (defining_op == nullptr) continue;
    auto is_producer_fused = [&](mlir::Value value) {
      if (value == nullptr) return false;
      LoopAttr loop = FirstLoopOrNull(value.getDefiningOp());
      if (loop == nullptr) return false;
      return first_loop.name() == loop.name();
    };
    if (is_producer_fused(defining_op.lower_bound()) ||
        is_producer_fused(defining_op.upper_bound())) {
      return op.emitError()
             << "operation cannot be nested in loop " << first_loop.name()
             << ": dimension sizes must be defined before entering the loop "
                "nest";
    }
  }

  return mlir::success();
}

mlir::LogicalResult VerifyRangeOp(mlir::Operation *op) {
  RangeOp range_op = cast<RangeOp>(op);
  if (!range_op.step().isStrictlyPositive()) {
    return range_op.emitError() << "step must be strictly positive";
  }
  return mlir::success();
}

#include "sair_op_interfaces.cc.inc"

}  // namespace sair
