// Copyright 2021 Google LLC
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

#include <memory>
#include <tuple>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "transforms/lowering.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Traverses `result_copies` attributes (expects DecisionsAttr to be the
// underlying type for all attributes) starting from `position` where each copy
// may refer to another copy and finds the instance that is being copied.
llvm::Optional<unsigned> FindCopiedInstance(
    llvm::ArrayRef<mlir::Attribute> result_copies, unsigned position,
    mlir::Location location) {
  auto decisions = result_copies[position].cast<DecisionsAttr>();
  if (decisions.copy_of() == nullptr || decisions.copy_of().isa<UnitAttr>()) {
    mlir::emitError(location) << "expected the source of copy to be specified";
    return llvm::None;
  }
  if (auto instance = decisions.copy_of().dyn_cast<InstanceAttr>()) {
    return instance.getValue();
  }
  return FindCopiedInstance(
      result_copies, decisions.copy_of().cast<CopyAttr>().getValue(), location);
}

// In the given container operation, clones operations that have multiple
// instances and creates copies of their results if requested by `instances`
// and `copies` attributes, respectively.
mlir::LogicalResult CreateInstancesAndCopies(Operation *container) {
  // Before starting, check that instances are indeed present when used. Drop
  // operations with empty instance lists and no uses.
  llvm::SmallVector<SairOp> ops;
  auto result = container->walk([&](SairOp sair_op) -> mlir::WalkResult {
    if (!sair_op.instances().has_value()) {
      sair_op->emitError() << "expected ops to have instances";
      return mlir::WalkResult::interrupt();
    }
    // Ops with no instances can be removed. They shouldn't have any users.
    if (sair_op.NumInstances() == 0) {
      // Exit must not be removed, or processed further if it has no instnaces
      if (isa<SairExitOp>(sair_op.getOperation())) {
        return mlir::WalkResult::advance();
      }

      for (Operation *user : sair_op->getUsers()) {
        auto sair_user = cast<SairOp>(user);
        if (sair_user.instances().has_value() &&
            sair_user.NumInstances() != 0) {
          continue;
        }
        auto diag =
            sair_op->emitError()
            << "operation has zero instances but its results are in use";
        diag.attachNote(user->getLoc()) << "user found";
        return diag;
      }
      sair_op.erase();
      return mlir::WalkResult::advance();
    }
    ops.push_back(sair_op);
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Stage 1: create clones of the original operation for each instance and
  // introduce copies of required. Populates the mapping from the original
  // operation results and operand attributes identifying them (CopyAttr or
  // InstanceAttr) to the results of the cloned operations or introduced copies.
  // Does not immediately update new operations since there is no guarantee of
  // defining operations being processed before users in graph regions.
  mlir::MLIRContext *context = container->getContext();
  auto instance_zero = InstanceAttr::get(context, 0);
  llvm::SmallVector<SairOp> new_ops;
  llvm::DenseMap<std::pair<mlir::Value, mlir::Attribute>, mlir::Value> mapping;
  for (SairOp sair_op : ops) {
    OpBuilder builder(sair_op);
    for (int i = 0, e = sair_op.NumInstances(); i < e; ++i) {
      Operation *clone = builder.clone(*sair_op.getOperation());
      for (mlir::Value result : sair_op->getResults()) {
        mapping.try_emplace(
            std::make_pair(result, InstanceAttr::get(context, i)),
            clone->getResult(result.cast<OpResult>().getResultNumber()));
      }

      // Keep only one instance in the cloned op and drop `copy_of`.
      clone->removeAttr(SairOp::kInstancesAttrName);
      auto clone_sair_op = cast<SairOp>(clone);
      clone_sair_op.AddInstance(sair_op.GetDecisions(i));
      clone_sair_op->removeAttr(ValueProducerOp::kCopiesAttrName);
      new_ops.push_back(clone_sair_op);
    }

    auto value_producer = dyn_cast<ValueProducerOp>(sair_op.getOperation());
    if (!value_producer) continue;

    // Value producer operations may also request copies of their results to be
    // produced. Create such copies.
    for (int i = 0, e = value_producer->getNumResults(); i < e; ++i) {
      for (auto en : llvm::enumerate(value_producer.GetCopies(i))) {
        DecisionsAttr decisions = en.value().cast<DecisionsAttr>();
        mlir::Value source = value_producer->getResult(i);
        unsigned rank =
            source.getType().cast<ValueType>().Shape().NumDimensions();
        llvm::Optional<unsigned> copied_instance = FindCopiedInstance(
            value_producer.GetCopies(i), en.index(), value_producer->getLoc());
        if (!copied_instance.has_value()) return mlir::failure();
        mlir::ArrayAttr instance_operand_attr =
            sair_op.GetDecisions(*copied_instance).operands();
        SmallVector<mlir::Attribute> operand_attrs;
        if (instance_operand_attr == nullptr) {
          operand_attrs.append(rank, instance_zero);
        } else {
          llvm::append_range(operand_attrs,
                             instance_operand_attr.getValue().take_front(rank));
        }
        operand_attrs.push_back(decisions.copy_of() == nullptr
                                    ? instance_zero
                                    : decisions.copy_of());
        auto copy_decisions = DecisionsAttr::get(
            decisions.sequence(), decisions.loop_nest(), decisions.storage(),
            decisions.expansion(), /*copy_of=*/nullptr,
            mlir::ArrayAttr::get(context, operand_attrs), context);
        mlir::Value copy = builder.create<SairCopyOp>(
            value_producer->getLoc(), source.getType(),
            sair_op.domain().take_front(rank),
            builder.getArrayAttr(MappingAttr::GetIdentity(context, rank)),
            source, builder.getArrayAttr(copy_decisions),
            /*copies=*/nullptr);
        mapping.try_emplace(
            std::make_pair(source, CopyAttr::get(context, en.index())), copy);
        new_ops.push_back(copy.getDefiningOp<SairOp>());
      }
    }
  }

  // Stage 2: in clones and copies, update operands to point to values produced
  // by other clones and copies using the mapping.
  for (SairOp sair_op : new_ops) {
    Operation *op = sair_op;
    DecisionsAttr decisions = sair_op.GetDecisions(0);
    if (decisions.operands() == nullptr) {
      return op->emitError()
             << "expected 'operands' field of '" << SairOp::kInstancesAttrName
             << "' to be specified";
    }
    llvm::SmallVector<mlir::Attribute> operand_attrs;
    llvm::append_range(operand_attrs, decisions.operands().getValue());
    for (int i = 0, e = op->getNumOperands(); i < e; ++i) {
      mlir::Value value = op->getOperand(i);
      mlir::Attribute key_attr = operand_attrs[i];
      if (key_attr.isa<UnitAttr>()) {
        return op->emitError()
               << "expceted concerete instance or copy as operand #" << i;
      }
      mlir::Value replacement = mapping.lookup(std::make_pair(value, key_attr));
      if (replacement == nullptr) {
        // Only external values produced by non-Sair ops can be missing a
        // replacement value.
        assert(value.getDefiningOp<SairOp>() == nullptr &&
               "could not materialize copy or instance");
        continue;
      }
      op->getOpOperand(i).set(replacement);
      operand_attrs[i] = instance_zero;
    }
    sair_op.SetDecisions(0, UpdateOperands(decisions, operand_attrs));
  }

  // Stage 3: remove the original operations. They may form use-def cycles, so
  // drop the uses first. The results of these ops should not be used by newly
  // introduced ops and they are therefore safe to be removed.
  for (SairOp sair_op : ops) {
    sair_op->dropAllUses();
    sair_op->erase();
  }

  return mlir::success();
}

// Pass that creates clones of an operation or copies of its results as
// requested by the respective attributes.
class MaterializeInstancesPass
    : public MaterializeInstancesPassBase<MaterializeInstancesPass> {
 public:
  void runOnOperation() override {
    if (mlir::failed(CreateInstancesAndCopies(getOperation()))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateMaterializeInstancesPass() {
  return std::make_unique<MaterializeInstancesPass>();
}
}  // namespace sair
