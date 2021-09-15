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
    if (sair_op->hasAttr(ComputeOp::kInstancesAttrName)) {
      return op->emitError() << "only compute Sair ops can have the '"
                             << ComputeOp::kInstancesAttrName << "' attribute";
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
  // Ensure that the instances attribute has the right type. Otherwise, return
  // success and let the op verifier raise an error. The op verifier will raise
  // an error as the attribute type is specified in the op definition.
  auto instances =
      operation->getAttrOfType<mlir::ArrayAttr>(ComputeOp::kInstancesAttrName);
  if (instances == nullptr) return mlir::success();
  if (llvm::any_of(instances.getValue(), [](mlir::Attribute attr) {
        return !attr.isa<DecisionsAttr>();
      })) {
    return mlir::success();
  }

  for (int i = 0, e = op.NumInstances(); i < e; ++i) {
    if (mlir::failed(VerifyDecisionsWellFormed(op.getLoc(), sair_op.shape(),
                                               op->getResultTypes(),
                                               op.GetDecisions(i)))) {
      return mlir::failure();
    }
  }
  return mlir::success();
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

bool HasExactlyOneInstance(SairOp op) {
  auto compute_op = dyn_cast<ComputeOp>(op.getOperation());
  if (compute_op != nullptr && compute_op.NumInstances() != 1) return false;
  auto value_producer = dyn_cast<ValueProducerOp>(op.getOperation());
  if (value_producer != nullptr && value_producer.HasCopies()) return false;
  return true;
}

OpInstance::OpInstance(SairOp op) : OpInstance(op, 0, 0) {
  assert(!llvm::isa<ComputeOp>(op.getOperation()));
}

OpInstance OpInstance::Unique(SairOp op) {
  assert(op.HasExactlyOneInstance());
  return OpInstance(op, 0, 0);
}

OpInstance::operator bool() const { return *this != nullptr; }

mlir::Operation *OpInstance::GetDuplicatedOp() const {
  return op_.get<SairOp>().getOperation();
}

mlir::Value OpInstance::GetCopiedValue() const {
  ValueProducerOp op = GetValueProducer();
  return op->getResult(result_);
}

ValueProducerOp OpInstance::GetValueProducer() const {
  return op_.get<ValueProducerOp>();
}

mlir::Operation *OpInstance::getOperation() const {
  if (is_copy()) return op_.get<ValueProducerOp>().getOperation();
  return op_.get<SairOp>().getOperation();
}

unsigned OpInstance::HashValue() const {
  intptr_t key = reinterpret_cast<intptr_t>(op_.getOpaqueValue());
  return llvm::hash_combine(llvm::hash_value(key), llvm::hash_value(result_),
                            llvm::hash_value(index_));
}

mlir::InFlightDiagnostic OpInstance::EmitError() const {
  if (auto compute_op = op_.dyn_cast<SairOp>()) {
    return compute_op.emitError() << "in instance " << index_;
  }
  auto value_producer = op_.get<ValueProducerOp>();
  return value_producer.emitError()
         << "in copy " << index_ << " of result " << result_ << ": ";
}

mlir::Diagnostic &OpInstance::AttachNote(mlir::InFlightDiagnostic &diag) const {
  if (auto compute_op = op_.dyn_cast<SairOp>()) {
    return diag.attachNote(compute_op->getLoc()) << "in instance " << index_;
  }
  auto value_producer = op_.get<ValueProducerOp>();
  return diag.attachNote(value_producer->getLoc())
         << "in copy " << index_ << " of result " << result_ << ": ";
}

mlir::Location OpInstance::getLoc() const { return getOperation()->getLoc(); }

mlir::MLIRContext *OpInstance::context() const {
  return getOperation()->getContext();
}

SairProgramOp OpInstance::program() const {
  return llvm::cast<SairProgramOp>(getOperation()->getParentOp());
}

SairDialect *OpInstance::GetSairDialect() const {
  return static_cast<SairDialect *>(getOperation()->getDialect());
}

DomainShapeAttr OpInstance::GetShape() const {
  if (is_copy()) {
    auto sair_op =
        llvm::cast<SairOp>(op_.get<ValueProducerOp>().getOperation());
    return sair_op.shape().Prefix(sair_op.results_rank());
  } else {
    return op_.get<SairOp>().shape();
  }
}

int OpInstance::domain_size() const {
  if (is_copy()) {
    auto sair_op =
        llvm::cast<SairOp>(op_.get<ValueProducerOp>().getOperation());
    return sair_op.results_rank();
  } else {
    return op_.get<SairOp>().domain().size();
  }
}

ResultInstance OpInstance::domain(int i) const {
  SairOp op;
  if (is_copy()) {
    op = llvm::cast<SairOp>(GetCopiedValue().getDefiningOp());
  } else {
    op = llvm::cast<SairOp>(GetDuplicatedOp());
  }
  mlir::Value dim = op.domain()[i];
  OpInstance dim_op(llvm::cast<SairOp>(dim.getDefiningOp()));
  return ResultInstance(dim_op, dim.cast<OpResult>().getResultNumber());
}

ValueRange OpInstance::GetDomainValues() const {
  if (is_copy()) {
    auto op = llvm::cast<SairOp>(GetCopiedValue().getDefiningOp());
    return op.domain().take_front(domain_size());
  } else {
    return llvm::cast<SairOp>(GetDuplicatedOp()).domain();
  }
}

OperandInstance OpInstance::Operand(int position) const {
  return OperandInstance(*this, position);
}

int OpInstance::num_results() const {
  if (is_copy()) return 1;
  return GetDuplicatedOp()->getNumResults();
}

ResultInstance OpInstance::Result(int result) const {
  return ResultInstance(*this, result);
}

llvm::SmallVector<int> OpInstance::SubDomains() const {
  if (is_copy()) {
    return {domain_size()};
  } else {
    return llvm::cast<SairOp>(GetDuplicatedOp()).SubDomains();
  }
}

llvm::SmallBitVector OpInstance::ResultsDimDependencies() const {
  if (is_copy()) {
    return llvm::SmallBitVector(domain_size());
  } else {
    return llvm::cast<SairOp>(GetDuplicatedOp()).ResultsDimDependencies();
  }
}

bool operator==(const OpInstance &lhs, const OpInstance &rhs) {
  return lhs.op_ == rhs.op_ && lhs.index_ == rhs.index_ &&
         lhs.result_ == rhs.result_;
}

bool operator!=(const OpInstance &lhs, const OpInstance &rhs) {
  return !(lhs == rhs);
}

ComputeOpInstance::ComputeOpInstance(const OpInstance &op) : OpInstance(op) {
  assert(op.isa<ComputeOpInstance>());
}

ComputeOpInstance ComputeOpInstance::Unique(ComputeOp op) {
  assert(llvm::cast<SairOp>(op.getOperation()).HasExactlyOneInstance());
  return ComputeOpInstance(op, 0);
}

DecisionsAttr ComputeOpInstance::GetDecisions() const {
  if (is_duplicate()) {
    return GetComputeOp().GetDecisions(index());
  }
  llvm::ArrayRef<mlir::Attribute> copies =
      GetValueProducer().GetCopies(result());
  return copies[index()].cast<DecisionsAttr>();
}

void ComputeOpInstance::SetDecisions(DecisionsAttr decisions) {
  assert(decisions != nullptr);
  if (is_duplicate()) {
    GetComputeOp().SetDecisions(index(), decisions);
  } else {
    GetValueProducer().SetCopy(result(), index(), decisions);
  }
}

llvm::ArrayRef<mlir::Attribute> ComputeOpInstance::Loops() const {
  DecisionsAttr decisions = GetDecisions();
  if (decisions.loop_nest() == nullptr) return {};
  return decisions.loop_nest().getValue();
}

void ComputeOpInstance::SetLoopNest(mlir::ArrayAttr loop_nest) {
  DecisionsAttr new_decisions =
      MapLoopNest([=](mlir::ArrayAttr) { return loop_nest; })(GetDecisions());
  SetDecisions(new_decisions);
}

BufferAttr ComputeOpInstance::Storage(int result) const {
  DecisionsAttr decisions = GetDecisions();
  if (decisions.storage() == nullptr ||
      decisions.storage()[result].isa<mlir::UnitAttr>()) {
    return nullptr;
  }
  return decisions.storage()[result].cast<BufferAttr>();
}

void ComputeOpInstance::SetStorage(int result, BufferAttr storage) {
  DecisionsAttr decisions = GetDecisions();
  auto unit_attr = mlir::UnitAttr::get(context());
  llvm::SmallVector<mlir::Attribute> array(num_results(), unit_attr);
  if (decisions.storage() != nullptr) {
    for (int i = 0; i < num_results(); ++i) {
      array[i] = decisions.storage()[i];
    }
  }
  array[result] = storage;
  mlir::ArrayAttr array_attr = mlir::ArrayAttr::get(context(), array);

  DecisionsAttr new_decisions =
      MapStorage([=](mlir::ArrayAttr) { return array_attr; })(decisions);
  SetDecisions(new_decisions);
}

ComputeOp ComputeOpInstance::GetComputeOp() const {
  return llvm::cast<ComputeOp>(GetDuplicatedOp());
}

ResultInstance ResultInstance::Unique(mlir::Value value) {
  OpResult result = value.cast<OpResult>();
  OpInstance producer = OpInstance::Unique(cast<SairOp>(result.getOwner()));
  return ResultInstance(producer, result.getResultNumber());
}

ShapedType ResultInstance::GetType() const {
  return GetValue().getType().cast<ShapedType>();
}

mlir::Value ResultInstance::GetValue() const {
  if (op_.is_copy()) {
    return op_.GetCopiedValue();
  } else {
    return op_.GetDuplicatedOp()->getResult(result_);
  }
}

unsigned ResultInstance::HashValue() const {
  return llvm::hash_combine(op_.HashValue(), llvm::hash_value(result_));
}

llvm::SmallVector<std::pair<OpInstance, int>> ResultInstance::GetUses() const {
  llvm::SmallVector<std::pair<OpInstance, int>> uses;
  // TODO(ulysse): allow specifying the instance used for operands. For now, we
  // always use the first instance.
  if (op_.is_copy() || op_.index() != 0) return {};
  mlir::Operation *def_op = op_.GetDuplicatedOp();

  // Register copy uses.
  if (auto value_producer = dyn_cast<ValueProducerOp>(def_op)) {
    int num_copies = value_producer.GetCopies(op_.result()).size();
    // The operand number must account for domain dimensions.
    int operand_number = GetType().Shape().NumDimensions();
    for (int i = 0; i < num_copies; ++i) {
      ComputeOpInstance instance(value_producer, op_.result(), i);
      uses.emplace_back(instance, operand_number);
    }
  }

  // Register non-copy uses.
  mlir::Value value = def_op->getResult(result_);
  for (OpOperand &use : value.getUses()) {
    mlir::Operation *user = use.getOwner();
    int operand_number = use.getOperandNumber();
    if (auto compute_op = dyn_cast<ComputeOp>(user)) {
      for (int i = 0, e = compute_op.NumInstances(); i < e; ++i) {
        uses.emplace_back(ComputeOpInstance(compute_op, i), operand_number);
      }
    } else {
      uses.emplace_back(OpInstance(cast<SairOp>(user)), operand_number);
    }
  }
  return uses;
}

bool operator==(const ResultInstance &lhs, const ResultInstance &rhs) {
  return lhs.op_ == rhs.op_ && lhs.result_ == rhs.result_;
}

bool operator!=(const ResultInstance &lhs, const ResultInstance &rhs) {
  return !(lhs == rhs);
}

std::optional<ResultInstance> OperandInstance::GetValue() const {
  // Retrieve the MLIR value.
  mlir::Value value;
  if (op_.is_copy()) {
    value = op_.GetCopiedValue();
  } else {
    auto owner = cast<SairOp>(op_.GetDuplicatedOp());
    value = owner.ValueOperands()[operand_position_].value();
  }

  auto result = value.cast<OpResult>();
  mlir::Operation *defining_op = result.getOwner();

  // TODO(ulysse): allow specifying the instance use in operands. For now, we
  // always use the first instance.
  OpInstance def_op;
  if (auto compute_op = dyn_cast<ComputeOp>(defining_op)) {
    if (compute_op.NumInstances() < 1) return std::nullopt;
    def_op = ComputeOpInstance(compute_op, 0);
  } else {
    def_op = OpInstance(cast<SairOp>(defining_op));
  }

  return ResultInstance(def_op, result.getResultNumber());
}

std::optional<ValueAccessInstance> OperandInstance::Get() const {
  auto value = GetValue();
  if (value.has_value()) return ValueAccessInstance({*value, Mapping()});
  return std::nullopt;
}

MappingAttr OperandInstance::Mapping() const {
  if (op_.is_copy()) {
    return MappingAttr::GetIdentity(op_.context(), op_.domain_size());
  } else {
    return GetOriginalOperand().Mapping();
  }
}

llvm::SmallBitVector OperandInstance::DependingDims() const {
  if (op_.is_copy()) {
    return llvm::SmallBitVector(op_.domain_size());
  } else {
    return GetOriginalOperand().DependingDims();
  }
}

bool OperandInstance::AllowUseBeforeDef() const {
  if (op_.is_copy()) return false;
  return GetOriginalOperand().AllowUseBeforeDef();
}

llvm::SmallBitVector OperandInstance::CarryingDims() const {
  if (op_.is_copy()) {
    return llvm::SmallBitVector(op_.domain_size());
  } else {
    return GetOriginalOperand().CarryingDims();
  }
}

ValueOperand OperandInstance::GetOriginalOperand() const {
  auto sair_op = cast<SairOp>(op_.GetDuplicatedOp());
  return sair_op.ValueOperands()[operand_position_];
}

#include "sair_op_interfaces.cc.inc"

}  // namespace sair
