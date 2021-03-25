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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "loop_nest.h"
#include "sair_ops.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

class LowerProjAny : public LowerProjAnyPassBase<LowerProjAny> {
  // Eliminates proj_any operations or lowers them to proj_last operations.
  void runOnFunction() override {
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);
    auto result = getFunction().walk([&](SairProjAnyOp op) -> mlir::WalkResult {
      builder.setInsertionPoint(op);
      const auto &iteration_spaces =
          getChildAnalysis<IterationSpaceAnalysis>(op->getParentOp());

      auto source = cast<SairOp>(op.value().getDefiningOp());
      const IterationSpace &source_space = iteration_spaces.Get(source);
      if (!source_space.mapping().IsIdentity()) {
        return source.emitError() << "operation iteration space not normalized";
      }

      int domain_size = op.domain().size();
      for (OpOperand &use : llvm::make_early_inc_range(op.result().getUses())) {
        SairOp user = cast<SairOp>(use.getOwner());
        ValueOperand operand(&use);
        const IterationSpace &user_space = iteration_spaces.Get(user);
        if (!user_space.mapping().IsIdentity()) {
          return use.getOwner()->emitError()
                 << "operation iteration space not normalized";
        }

        // Compute number of common loops between source and user.
        llvm::ArrayRef<mlir::StringAttr> source_loops =
            source_space.loop_names();
        llvm::ArrayRef<mlir::StringAttr> user_loops = user_space.loop_names();
        auto it_pair = std::mismatch(source_loops.begin(), source_loops.end(),
                                     user_loops.begin(), user_loops.end());
        int num_common_loops =
            std::distance(source_loops.begin(), it_pair.first);

        auto loops_mapping =
            MappingAttr::GetIdentity(context, num_common_loops);
        auto value_mapping =
            operand.Mapping().Resize(domain_size).Compose(op.Value().Mapping());
        MappingAttr mapping = loops_mapping.Resize(value_mapping.size())
                                  .ResizeUseDomain(user.domain().size())
                                  .Unify(value_mapping);

        // Projection can be fully eliminated.
        if (num_common_loops == mapping.size()) {
          operand.set_value(op.value());
          operand.SetMapping(mapping.Compose(op.Value().Mapping()));
          continue;
        }

        // We expect a mapping that is identity on first `num_common_loops`
        // dimensions and `none` afterward. First `num_common_loops` are
        // guaranteed to be identity by verifiers.
        for (MappingExpr expr :
             mapping.Dimensions().drop_front(num_common_loops)) {
          if (!expr.isa<MappingNoneExpr>()) {
            return op.emitError()
                   << "cannot lower operation to proj_last on scalars";
          }
        }

        // Create a proj_last operation.
        auto proj_shape = source.shape().Prefix(mapping.size());
        auto proj_type = ValueType::get(proj_shape.Prefix(num_common_loops),
                                        op.Value().GetType().ElementType());
        auto parallel_domain = source.domain().take_front(num_common_loops);
        auto projection_domain = source.domain().slice(
            num_common_loops, mapping.size() - num_common_loops);
        auto proj_mapping = MappingAttr::GetIdentity(context, mapping.size());

        auto proj_last = builder.create<SairProjLastOp>(
            op.getLoc(), proj_type, parallel_domain, projection_domain,
            builder.getArrayAttr({proj_mapping}), op.value(), proj_shape);

        operand.set_value(proj_last);
        int user_domain_size = operand.Mapping().UseDomainSize();
        operand.SetMapping(MappingAttr::GetIdentity(context, num_common_loops)
                               .ResizeUseDomain(user_domain_size));
      }

      op.erase();
      return mlir::success();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerProjAnyPass() {
  return std::make_unique<LowerProjAny>();
}

}  // namespace sair
