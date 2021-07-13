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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_ops.h"
#include "storage.h"
#include "transforms/domain_utils.h"
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

class LowerToMap : public LowerToMapPassBase<LowerToMap> {
  // Converts sair.copy operations into sair.map operations. This is a hook for
  // the MLIR pass infrastructure.
  void runOnFunction() override {
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);

    getFunction().walk([&](ComputeOp op) -> mlir::WalkResult {
      auto *sair_dialect = static_cast<SairDialect *>(op->getDialect());
      if (!op.expansion().hasValue()) {
        signalPassFailure();
        return op.emitError() << "no target expansion pattern specified";
      }

      auto sair_op = cast<SairOp>(op.getOperation());
      MapBodyBuilder map_body(sair_op.domain().size(), op->getContext());
      builder.setInsertionPointToStart(&map_body.block());
      for (ValueOperand operand : sair_op.ValueOperands()) {
        map_body.AddOperand(operand.Get());
      }

      const ExpansionPattern &pattern =
          *sair_dialect->GetExpansionPattern(*op.expansion());
      llvm::SmallVector<mlir::Value> results =
          pattern.Emit(op, map_body, builder);
      builder.create<SairReturnOp>(op.getLoc(), results);

      builder.setInsertionPoint(op);
      DecisionsAttr decisions = op.GetDecisions();
      auto new_decisions = DecisionsAttr::get(
          decisions.sequence(), decisions.loop_nest(), decisions.storage(),
          builder.getStringAttr(kMapExpansionPattern), context);
      SairMapOp map_op = builder.create<SairMapOp>(
          op.getLoc(), op->getResultTypes(), sair_op.domain(),
          map_body.sair_values(), sair_op.shape(), new_decisions);
      map_op.body().takeBody(map_body.region());
      ForwardAttributes(op, map_op);

      op->replaceAllUsesWith(map_op);
      op->erase();
      return mlir::success();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerToMapPass() {
  return std::make_unique<LowerToMap>();
}

}  // namespace sair
