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
  for (auto pair :
       llvm::zip(sair_op.domain(), sair_op.shape().Dimensions())) {
    assert(std::get<0>(pair).getType() == std::get<1>(pair).type());
    (void)pair;
  }

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
    if (!defining_op->isBeforeInBlock(op)) {
      return (op->emitError() << "dimension used before its definition")
                 .attachNote(defining_op->getLoc())
             << "definition here";
    }
  }

  if (!sair_op.ValueOperands().empty()) {
    // Verify that the "mapping_array" attribute exists.
    if (!op->getAttr(::sair::SairDialect::kMappingAttrName)) {
      return mlir::emitError(op->getLoc())
             << "missing " << ::sair::SairDialect::kMappingAttrName
             << " attribute";
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

    assert(sair_op.shape().AccessedShape(v.Mapping()) ==
           v.value().getType().template cast<::sair::ValueType>().Shape());

    if (!defining_op->isBeforeInBlock(op) && !v.AllowUseBeforeDef()) {
      return (op->emitError() << "operand used before its definition")
                 .attachNote(defining_op->getLoc())
             << "definition here";
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
    if (sair_op->hasAttr(ComputeOp::kLoopNestAttrName)) {
      return op->emitError() << "only compute Sair ops can have the '"
                             << ComputeOp::kLoopNestAttrName << "' attribute";
    }
    if (sair_op->hasAttr(ComputeOp::kSequenceAttrName)) {
      return op->emitOpError() << "unexpected '" << ComputeOp::kSequenceAttrName
                               << "' attribute on a non-compute op";
    }
  }

  return ::mlir::success();
}

namespace {
// Simple RAII object that maintains a stack of MLIR Locations.
class ExtraLocationSaver {
 public:
  ExtraLocationSaver(llvm::SmallVectorImpl<mlir::Location> &vector,
                     mlir::Location location)
      : vector_(vector) {
    vector_.push_back(location);
  }

  ~ExtraLocationSaver() { vector_.pop_back(); }

 private:
  llvm::SmallVectorImpl<mlir::Location> &vector_;
};
}  // namespace

static mlir::LogicalResult VerifyDomainOperandSequenced(
    Value operand, int64_t owner_sequence_number,
    llvm::SmallVectorImpl<mlir::Location> &locations);

// Verifies that the operations producing the domain operands of `sair_op` are
// sequenced before `owner_sequence_number`. Reports errors using the
// `locations` stack otherwise.
static mlir::LogicalResult VerifyDomainOperandsSequenced(
    SairOp sair_op, int64_t owner_sequence_number,
    llvm::SmallVectorImpl<mlir::Location> &locations) {
  for (mlir::Value domain_operand : sair_op.domain()) {
    if (mlir::failed(VerifyDomainOperandSequenced(
            domain_operand, owner_sequence_number, locations))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

// Verifies that the operation producing Sair value `operand` is sequenced
// before `owner_sequence_number`. Reports errors using the `locations` stack
// otherwise.
static mlir::LogicalResult VerifyOperandSequenced(
    mlir::Value operand, int64_t owner_sequence_number,
    llvm::SmallVectorImpl<mlir::Location> &locations,
    mlir::Operation *allow_same_with, bool seen_fby_then = false) {
  mlir::Operation *defining_op = operand.getDefiningOp();
  assert(defining_op && "expected Sair values to have a defining op");

  // If the operand is defined by a compute op, that op may have the sequence
  // attribute. Use that for verification.
  if (auto defining_compute_op = dyn_cast<ComputeOp>(defining_op)) {
    llvm::Optional<int64_t> operand_sequence_number =
        defining_compute_op.Sequence();
    if (!operand_sequence_number ||
        *operand_sequence_number < owner_sequence_number ||
        (*operand_sequence_number == owner_sequence_number && seen_fby_then &&
         allow_same_with == defining_compute_op)) {
      return mlir::success();
    }

    mlir::InFlightDiagnostic diag =
        mlir::emitError(locations.front())
        << "value use sequenced before its definition";
    for (mlir::Location loc : llvm::drop_begin(locations)) {
      diag.attachNote(loc)
          << "transitive through this implicitly sequenced operation";
    }
    diag.attachNote(defining_op->getLoc()) << "sequenced value definition";
    return diag;
  }

  // Otherwise, we need to walk up the use-def chains until we find a compute
  // operation. Keep track of operations we are traversing for eventual error
  // reporting.
  ExtraLocationSaver raii(locations, defining_op->getLoc());

  // First, check the domain in case it has any dependent ranges. This wasn't
  // necessary for compute operations as they have their domain checked
  // separately (for better error reporting).
  auto defining_sair_op = cast<SairOp>(defining_op);
  if (mlir::failed(VerifyDomainOperandsSequenced(
          defining_sair_op, owner_sequence_number, locations))) {
    return mlir::failure();
  }

  // Recursively check Sair value operands of the operations that produce the
  // current operand.
  for (ValueOperand operand : defining_sair_op.ValueOperands()) {
    // "fby" can be used to create definition loops that must be sequenced at
    // the same point, and is the only operation that has AllowUseBeforeDef.
    if (mlir::failed(VerifyOperandSequenced(
            operand.value(), owner_sequence_number, locations, allow_same_with,
            operand.AllowUseBeforeDef()))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Verifies that the value that appears in the domain of a Sair operation is
// produced by an operation sequenced before `owner_sequence_number`. Reports
// errors using the `locations` stack otherwise.
static mlir::LogicalResult VerifyDomainOperandSequenced(
    Value operand, int64_t owner_sequence_number,
    llvm::SmallVectorImpl<mlir::Location> &locations) {
  mlir::Operation *defining_op = operand.getDefiningOp();
  assert(defining_op && "expected Sair range values to have a defining op");

  // Static ranges are sequenced relative to their users and don't have operands
  // that are explicitly sequenced.
  if (auto defining_range_op = dyn_cast<SairStaticRangeOp>(defining_op)) {
    return mlir::success();
  }

  ExtraLocationSaver raii(locations, defining_op->getLoc());

  // Dynamic ranges may be using values, which must be sequenced before the user
  // of dynamic range.
  if (auto defining_dyn_range_op = dyn_cast<SairDynRangeOp>(defining_op)) {
    if (defining_dyn_range_op.LowerBound().is_value() &&
        mlir::failed(VerifyOperandSequenced(defining_dyn_range_op.lower_bound(),
                                            owner_sequence_number, locations,
                                            nullptr))) {
      return mlir::failure();
    }
    if (mlir::failed(VerifyOperandSequenced(defining_dyn_range_op.upper_bound(),
                                            owner_sequence_number, locations,
                                            nullptr))) {
      return mlir::failure();
    }
  }

  // Sair ops may have dynamic domain dimensions that need further verification.
  return VerifyDomainOperandsSequenced(cast<SairOp>(defining_op),
                                       owner_sequence_number, locations);
}

// Verifies that the operands of the given compute operation are defined by
// operations sequenced before (or at the same time as, in case of fby) this
// operation.
static mlir::LogicalResult VerifyOperandsSequenced(ComputeOp op) {
  llvm::Optional<int64_t> sequence_number = op.Sequence();
  if (!sequence_number) return mlir::success();

  llvm::SmallVector<mlir::Location> locations;
  locations.push_back(op.getLoc());
  auto sair_op = cast<SairOp>(op.getOperation());
  for (ValueOperand operand : sair_op.ValueOperands()) {
    if (mlir::failed(VerifyOperandSequenced(operand.value(), *sequence_number,
                                            locations, op))) {
      return mlir::failure();
    }
  }
  assert(locations.size() == 1);
  return VerifyDomainOperandsSequenced(sair_op, *sequence_number, locations);
}

mlir::LogicalResult VerifyComputeOp(mlir::Operation *operation) {
  ComputeOp op(operation);
  if (op.loop_nest().hasValue() &&
      mlir::failed(VerifyLoopNestWellFormed(op, op.LoopNestLoops()))) {
    return mlir::failure();
  }
  return VerifyOperandsSequenced(op);
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
