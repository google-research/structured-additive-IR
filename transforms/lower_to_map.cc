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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "sair_ops.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Lowers `op` into a sair.map operation that takes a single value as argument
// and returns it immediately.
void RewriteCopyToMap(SairCopyOp op, mlir::OpBuilder &builder) {
  mlir::OperandRange inputs = op.getOperands().slice(op.domain().size(), 1);
  SairMapOp new_op = builder.create<SairMapOp>(
      op.getLoc(), op.getType(), op.domain(), op.access_pattern_array(), inputs,
      op.shape(), op.loop_nestAttr(), op.memory_spaceAttr());

  // Set the block of code in the sair.map operation to return its argument
  // unchanged.
  builder.setInsertionPointToStart(&new_op.block());
  builder.create<SairReturnOp>(op.getLoc(), new_op.block_inputs().front());
  op.result().replaceAllUsesWith(new_op.getResult(0));
  op.erase();
}

class CopyToMap : public CopyToMapPassBase<CopyToMap> {
  // Converts sair.copy operations into sair.map operations. This is a hook for
  // the MLIR pass infrastructure.
  void runOnFunction() override {
    mlir::MLIRContext *context = &getContext();
    getFunction().walk([context](SairCopyOp op) {
      mlir::OpBuilder builder(context);
      builder.setInsertionPoint(op);
      RewriteCopyToMap(op, builder);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateCopyToMapPass() {
  return std::make_unique<CopyToMap>();
}

}  // namespace sair
