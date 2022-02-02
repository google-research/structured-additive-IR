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

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Checks if the given Sair Op is trivial and can be simplified. Sair Ops are
// trivial if they have a body region with 0D shape and all their operands are
// 0D Sair values constructed from known scalars.
bool IsTrivialSairMap(SairMapOp op) {
  if (!op.shape().Is0d()) return false;
  for (const ValueOperand operand : op.ValueOperands()) {
    assert(operand.GetType().Shape().Is0d());

    mlir::Operation *defining_operation = operand.value().getDefiningOp();
    if (!isa_and_nonnull<SairFromScalarOp>(defining_operation)) {
      return false;
    }
  }
  return true;
}

// Replaces the first trivial Sair Op in the function with the contents of its
// body. The results of the trivial Op are wrapped into a 0D Sair value. This
// processes only one trivial Op since the rewrite it performs can make other
// Ops trivial, and since it can invalidate the pointers to the operations if we
// stored them during a pre-pass. Returns "true" if it modified the input
// function, "false" otherwise.
bool InlineTrivialSairOp(mlir::FuncOp function) {
  // Find the first map or map_reduce operation with 0d domain and 0d inputs.
  SairMapOp trivial_op;
  mlir::WalkResult walk_result = function.walk([&trivial_op](SairMapOp op) {
    if (IsTrivialSairMap(op)) {
      trivial_op = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!walk_result.wasInterrupted()) {
    return false;
  }
  assert(trivial_op);

  // Collect the source non-Sair scalars that are used to construct Sair values.
  llvm::SmallVector<mlir::Value, 8> source_values;
  auto trivial_op_operands = trivial_op.ValueOperands();
  source_values.reserve(trivial_op_operands.size());
  for (const ValueOperand operand : trivial_op_operands) {
    mlir::Operation *defining_operation = operand.value().getDefiningOp();
    source_values.push_back(cast<SairFromScalarOp>(defining_operation).value());
  }

  // Use the sources of the 0d value directly.
  mlir::Operation *trivial_operation = trivial_op.getOperation();
  assert(trivial_op_operands.size() == trivial_op.block().getNumArguments());
  for (auto pair :
       llvm::zip(source_values, trivial_op.block().getArguments())) {
    mlir::Value source = std::get<0>(pair);
    mlir::Value block_argument = std::get<1>(pair);
    mlir::replaceAllUsesInRegionWith(block_argument, source, trivial_op.body());
  }

  // Move the body contents immediately before the Sair program.
  auto sair_program = trivial_op->getParentOfType<SairProgramOp>();
  mlir::Operation *sair_program_operation = sair_program.getOperation();
  mlir::Block *program_block = sair_program_operation->getBlock();
  mlir::Operation *terminator = trivial_op.block().getTerminator();
  program_block->getOperations().splice(
      mlir::Block::iterator(sair_program_operation),
      trivial_op.block().getOperations());

  // Wrap the terminator operands into Sair values so they are compatible with
  // the remaining users.
  llvm::SmallVector<mlir::Value, 8> result_values;
  mlir::OpBuilder builder(trivial_op);
  result_values.reserve(terminator->getNumOperands());
  for (mlir::Value operand : terminator->getOperands()) {
    mlir::Type type = ValueType::get(operand.getType());
    mlir::Value value =
        builder.create<SairFromScalarOp>(trivial_op.getLoc(), type, operand);
    result_values.push_back(value);
  }
  trivial_operation->replaceAllUsesWith(result_values);
  terminator->erase();

  // Erase "from_value" if we are about to erase its only user. Explicitly drop
  // the uses of defined values first to break use-def chains.
  for (const ValueOperand operand : trivial_op_operands) {
    mlir::Operation *defining_operation = operand.value().getDefiningOp();
    if (defining_operation->hasOneUse()) {
      defining_operation->dropAllDefinedValueUses();
      defining_operation->erase();
    }
  }
  trivial_operation->erase();
  return true;
}

// MLIR pass that replaces trivial Sair ops with the content of their body.
class InlineTrivialSairOpsPass
    : public InlineTrivialSairOpsPassBase<InlineTrivialSairOpsPass> {
  void runOnOperation() override {
    mlir::FuncOp function = getOperation();
    // Iteratively find the first trivial Sair Op and inline it, which may
    // create new trivial Ops. The iteration stops when no trivial Ops are
    // present in the function.
    while (InlineTrivialSairOp(function)) {
      // Intentionally empty.
    }

    // Inline trivial sair.program operations. A sair.program operation is
    // trivial if it only contains sair.from_scalar operations apart from its
    // terminator.
    function.walk([](SairProgramOp op) {
      for (mlir::Operation &sair_op : op.body().front().without_terminator()) {
        if (!isa<SairFromScalarOp>(&sair_op)) return;
      }
      mlir::Operation *exit_op = op.body().front().getTerminator();
      for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        // The defining operation of sair.exit operands are defined in the same
        // block than the sair.exit operation. This is checked by SairOpTrait.
        SairFromScalarOp from_scalar_op = llvm::cast<SairFromScalarOp>(
            exit_op->getOperand(i).getDefiningOp());
        op.getResult(i).replaceAllUsesWith(from_scalar_op.value());
      }
      op.erase();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateInlineTrivialOpsPass() {
  return std::make_unique<InlineTrivialSairOpsPass>();
}

}  // namespace sair
