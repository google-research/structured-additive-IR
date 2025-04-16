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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "loop_nest.h"
#include "sair_dialect.h"
#include "sair_ops.h"

namespace sair {

#define GEN_PASS_DEF_LOWERPROJANYPASS
#include "transforms/lowering.h.inc"

namespace {

class LowerProjAny : public impl::LowerProjAnyPassBase<LowerProjAny> {
  // Eliminates proj_any operations or lowers them to proj_last operations.
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);
    auto result = getOperation().walk([&](SairProjAnyOp op)
                                          -> mlir::WalkResult {
      builder.setInsertionPoint(op);
      const auto &iteration_spaces =
          getChildAnalysis<IterationSpaceAnalysis>(op->getParentOp());

      auto source = cast<SairOp>(op.getValue().getDefiningOp());
      if (op.HasCopies()) {
        return source.emitError() << "copies must be materialized before "
                                     "lowering proj_any operations";
      }

      if (!source.HasExactlyOneInstance()) {
        return source.emitError() << "instances must be materialized before "
                                     "lowering proj_any operations";
      }
      const IterationSpace &source_space =
          iteration_spaces.Get(OpInstance::Unique(source));
      if (!source_space.mapping().IsIdentity()) {
        return source.emitError() << "operation iteration space not normalized";
      }

      int domain_size = op.getDomain().size();
      for (OpOperand &use :
           llvm::make_early_inc_range(op.getResult().getUses())) {
        SairOp user = cast<SairOp>(use.getOwner());
        ValueOperand operand(&use);
        const IterationSpace &user_space =
            iteration_spaces.Get(OpInstance::Unique(user));
        if (!user_space.mapping().IsIdentity()) {
          return use.getOwner()->emitError()
                 << "operation iteration space not normalized";
        }

        int num_common_loops = source_space.NumCommonLoops(user_space);
        auto loops_mapping =
            MappingAttr::GetIdentity(context, num_common_loops);
        auto value_mapping =
            operand.Mapping().Resize(domain_size).Compose(op.Value().Mapping());
        MappingAttr mapping = loops_mapping.Resize(value_mapping.size())
                                  .ResizeUseDomain(user.getDomain().size())
                                  .Unify(value_mapping);

        // Projection can be fully eliminated.
        if (num_common_loops == mapping.size()) {
          operand.set_value(op.getValue());
          operand.SetMapping(mapping.Compose(op.Value().Mapping()));
          continue;
        }

        // We expect a mapping that is identity on first `num_common_loops`
        // dimensions and `none` afterward. First `num_common_loops` are
        // guaranteed to be identity by verifiers.
        for (MappingExpr expr :
             mapping.Dimensions().drop_front(num_common_loops)) {
          if (!mlir::isa<MappingNoneExpr>(expr)) {
            return op.emitError()
                   << "cannot lower operation to proj_last on scalars";
          }
        }

        // Create a proj_last operation.
        auto proj_shape = source.getShape().Prefix(mapping.size());
        auto proj_type = ValueType::get(proj_shape.Prefix(num_common_loops),
                                        op.Value().GetType().ElementType());
        auto parallel_domain = source.getDomain().take_front(num_common_loops);
        auto projection_domain = source.getDomain().slice(
            num_common_loops, mapping.size() - num_common_loops);
        auto proj_mapping = MappingAttr::GetIdentity(context, mapping.size());

        auto proj_last = builder.create<SairProjLastOp>(
            op.getLoc(), proj_type, parallel_domain, projection_domain,
            builder.getArrayAttr({proj_mapping}), op.getValue(), proj_shape,
            /*instances=*/op.getInstancesAttr(),
            /*copies=*/nullptr);

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

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLowerProjAnyPass() {
  return std::make_unique<LowerProjAny>();
}

}  // namespace sair
