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

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "sair_ops.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Returns a value containing the size of the given !sair.range dimension,
// available in the body of 'op'. Adds an argument to 'op' to pass the size if
// needed. Use 'builder' to insert operations in the body of 'op' if needed.
//
// Expects the dimension to be defined in the same sair.program. Returns nullptr
// if the dimension depends on any other dimension.
mlir::Value RetrieveRangeSize(SairOpWithBody op, mlir::Value range_dimension,
                              mlir::OpBuilder &builder) {
  mlir::Operation *defining_op = range_dimension.getDefiningOp();
  assert(defining_op && "sair computations must be contained in a block");
  if (auto static_range = llvm::dyn_cast<SairStaticRangeOp>(defining_op)) {
    return builder.create<mlir::ConstantOp>(range_dimension.getLoc(),
                                            static_range.sizeAttr());
  }

  auto range = llvm::cast<SairRangeOp>(defining_op);
  SairOp sair_op = cast<SairOp>(op.getOperation());
  if (!range.domain().empty()) return nullptr;
  return op.AddValueOperand(
      range.size(), AccessPatternAttr::get(builder.getContext(),
                                           sair_op.domain().size(), {}));
}

// Replaces the innermost dimension of the domain by a loop. Returns a failure
// if the operation has an empty domain, if an operand depends on the innermost
// dimension, if the dimension depends on other dimensions or if dimensions are
// not defined in the same sair.program. Passes the given loop carried variables
// to the generated loop and returns the result of the loop from the operation
// body.
mlir::LogicalResult IntroduceLoop(SairOpWithBody op,
                                  mlir::ValueRange loop_carried_variables,
                                  mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);

  SairOp sair_op = SairOp(op);
  if (sair_op.domain().empty()) return mlir::failure();
  int dim_position = sair_op.domain().size() - 1;
  // Check that no input depend on the dimension that we are removing.
  for (ValueOperand operand : sair_op.ValueOperands()) {
    if (operand.AccessPattern().DependsOnDimension(dim_position)) {
      return mlir::failure();
    }
  }

  // Build the loop.
  builder.setInsertionPointToStart(&op.block());
  mlir::Value upper_bound = RetrieveRangeSize(
      SairOpWithBody(op), sair_op.domain()[dim_position], builder);
  if (upper_bound == nullptr) return mlir::failure();
  auto zero = builder.create<mlir::ConstantIndexOp>(op.getLoc(), 0);
  auto one = builder.create<mlir::ConstantIndexOp>(op.getLoc(), 1);
  mlir::scf::ForOp for_op = builder.create<mlir::scf::ForOp>(
      op.getLoc(), zero, upper_bound, one, loop_carried_variables);
  op.block()
      .getArgument(dim_position)
      .replaceAllUsesWith(for_op.getInductionVar());

  // Move the body of the Sair operation in the for while keeping the
  // sair.return operation.
  mlir::Block::OpListType &for_body = for_op.getBody()->getOperations();
  for_body.splice(for_body.begin(), op.block().getOperations(),
                  mlir::Block::iterator(for_op.getOperation()->getNextNode()),
                  op.block().without_terminator().end());
  builder.setInsertionPointToEnd(for_op.getBody());
  // The sair.yield operation is automatically created by the scf::ForOp
  // builder if there are no induction variables.
  if (!loop_carried_variables.empty()) {
    builder.create<mlir::scf::YieldOp>(
        for_op.getLoc(), op.block().getTerminator()->getOperands());
  }

  // Rename loop carried variables inside the loop.
  for (std::tuple<mlir::Value, mlir::Value> p :
       llvm::zip(loop_carried_variables, for_op.getRegionIterArgs())) {
    mlir::replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p),
                                     for_op.getRegion());
  }
  op.block().getTerminator()->setOperands(for_op.getResults());
  op.RemoveInnermostDimension();
  return mlir::success();
}

// Replaces the innermost dimension of the domain by a loop. Returns a failure
// if the operation has an empty domain, if an operand depends on the innermost
// dimension, if the dimension depends on other dimensions or if the dimensions
// are not defined in the same sair.program. Also fails if the operation has any
// output as changing the number of dimensions would change the shape of the
// outputs.
mlir::LogicalResult IntroduceLoop(SairMapOp op, mlir::OpBuilder &builder) {
  if (op.getNumResults() > 0) return mlir::failure();
  return IntroduceLoop(SairOpWithBody(op), {}, builder);
}

// Replaces the innermost reduction dimension of the operation. Returns a
// failure if the reduction domain is empty, if any input depends on the
// reduction dimension, if the dimension depends on other dimensions or if
// dimensions are not defined in the same sair.program.
mlir::LogicalResult IntroduceLoop(SairMapReduceOp op,
                                  mlir::OpBuilder &builder) {
  if (op.reduction_domain().empty()) return mlir::failure();
  auto loop_carried_variables =
      op.block().getArguments().slice(op.domain().size(), op.inits().size());
  return IntroduceLoop(SairOpWithBody(op), loop_carried_variables, builder);
}

// Converts a sair.map_reduce operation with no iteration dimensions into a
// sair.map operation.
void LowerMapReduceToMap(SairMapReduceOp op, mlir::OpBuilder &builder) {
  assert(op.reduction_domain().empty());
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  builder.setInsertionPoint(op);
  OperandRange operands = op.getOperands().drop_front(op.domain().size());
  SairMapOp new_op = builder.create<SairMapOp>(
      op.getLoc(), op.getResultTypes(), op.parallel_domain(),
      op.access_pattern_array(), operands, op.shape(), op.loop_nestAttr(),
      op.memory_spaceAttr());
  new_op.body().takeBody(op.body());
  op.replaceAllUsesWith(new_op.results());
  op.erase();
}

// Replaces iteration dimensions in sair.map and sair.map_reduce operation by
// loops, converting sair.map_reduce operation into sair.map operations in the
// process. Fails if operations operand depend on any dimension,  if operations
// have results with more than 1 dimension or if dimensions are not defined in
// the same sair.program.
class IntroduceLoops : public IntroduceLoopsPassBase<IntroduceLoops> {
  void runOnFunction() override {
    mlir::OpBuilder builder(&getContext());
    // Lower sair.map_reduce operations into sair.map operations.
    getFunction().walk([&](SairMapReduceOp op) {
      for (int i = 0, e = op.reduction_domain().size(); i < e; ++i) {
        if (mlir::failed(IntroduceLoop(op, builder))) {
          op.emitError() << "failed to introduce loops";
          signalPassFailure();
        }
      }
      LowerMapReduceToMap(op, builder);
    });
    // Lower sair.map operations.
    getFunction().walk([&](SairMapOp op) {
      for (int i = 0, e = op.domain().size(); i < e; ++i) {
        if (mlir::failed(IntroduceLoop(op, builder))) {
          op.emitError() << "failed to introduce loops";
          signalPassFailure();
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateIntroduceLoopsPass() {
  return std::make_unique<IntroduceLoops>();
}

}  // namespace sair
