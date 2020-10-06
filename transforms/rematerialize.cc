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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Contains the loop bounds in the form of a variable range and constant step.
struct LoopBounds {
  LoopBounds(mlir::Value range, int step, bool is_dependent)
      : range(range), step(step), is_dependent(is_dependent) {}

  mlir::Value range;
  int step;

  // Set if the range is dependent on another value. This should be eventually
  // replaced by the dependence description, but for now only serves to abort
  // rematerialization in such cases.
  bool is_dependent;
};

// Creates a new sair.copy operation that is intended to replace `op`. Takes the
// domain and loop nest attribute supplied as arguments, extracts the value
// being copied and the access pattern from `op` and constructs the type based
// on the existing elemental type and the domain shape.
SairCopyOp RecreateOp(SairCopyOp op, mlir::ValueRange domain,
                      llvm::ArrayRef<mlir::Attribute> loop_nest,
                      DomainShapeAttr domain_shape, mlir::OpBuilder &builder) {
  auto origial_type = op.getResult().getType().cast<ValueType>();
  auto new_type =
      ValueType::get(op.getContext(), domain_shape, origial_type.ElementType());
  auto loop_nest_attr = builder.getArrayAttr(loop_nest);
  return builder.create<SairCopyOp>(op.getLoc(), new_type, domain,
                                    op.access_pattern_array(), op.value(),
                                    loop_nest_attr, op.memory_spaceAttr());
}

// Replaces `op` by the same op with actual dimensions in the domain instead of
// rematerialization tags. Effectively introduces as many trailing domain
// operands as `loops` and extends the shape of the result accordingly. Expects
// `loops` to contain indices of dimensions tagged for rematerialization in the
// loop nest attribute. The `main_loops` map should contain the loop bounds for
// all dimensions to rematerialize.
mlir::LogicalResult Rematerialize(
    ComputeOp op, ArrayRef<size_t> loops,
    const llvm::DenseMap<mlir::Attribute, LoopBounds> &main_loops) {
  MLIRContext *ctx = op.getContext();
  auto sair_op = cast<SairOp>(op.getOperation());

  // Keep the original domain and store the operand position to use for new
  // domain dimensions about to be inserted.
  auto domain = llvm::to_vector<6>(sair_op.domain());
  size_t position = domain.size();
  domain.reserve(domain.size() + loops.size());

  // For each loop to rematerialize, add the range as the last domain argument
  // and update the loop nest attribute accordingly.
  auto loop_nest_array = llvm::to_vector<4>(op.LoopNestLoops());
  for (size_t loop_index : loops) {
    auto loop = op.LoopNestLoops()[loop_index].cast<LoopAttr>();
    auto bounds_iterator = main_loops.find(loop.name());
    assert(bounds_iterator != main_loops.end() &&
           "invalid loop_nest attribute");
    const LoopBounds &bounds = bounds_iterator->getSecond();
    domain.push_back(bounds.range);

    // TODO: attempt to move the upward slice of the range before its first use.
    mlir::Operation *range_def = bounds.range.getDefiningOp();
    assert(range_def && "unexpected !sair.range as block argument");
    if (!range_def->isBeforeInBlock(op.getOperation())) {
      return (range_def->emitOpError()
              << "range value definition would not precede its use after "
                 "rematerialization")
                 .attachNote(op.getLoc())
             << "to be used here";
    }

    if (bounds.is_dependent) {
      return op.emitOpError()
             << "rematerialization is not supported for dependent dimensions";
    }

    loop_nest_array[loop_index] = LoopAttr::get(
        loop.name(), IteratorAttr::get(ctx, position++, bounds.step), ctx);
  }

  // Expand the shape accordingly.
  // TODO: this assumes we can only rematerialize independent dimensions. In the
  // future, we should also pull the dimensions it depends on.
  auto extra_domain =
      DomainShapeAttr::HyperRectangular(op.getContext(), loops.size());
  auto domain_shape = sair_op.shape().Product(extra_domain);

  OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  if (auto copy_op = dyn_cast<SairCopyOp>(op.getOperation())) {
    SairCopyOp new_op =
        RecreateOp(copy_op, domain, loop_nest_array, domain_shape, builder);
    // The parallel domain contains all the original dimensions.
    mlir::ValueRange parallel_domain = copy_op.domain();

    // The reduction domain contains the dimensions introduced for
    // rematerialization.
    auto reduction_domain = mlir::ValueRange(domain).take_back(loops.size());

    // Project out the rematerialized dimensions. Use the identity access
    // pattern here since defs and uses conserved their patterns.
    mlir::Value new_result = builder.create<SairProjAnyOp>(
        copy_op.getLoc(), copy_op.getType(), parallel_domain, reduction_domain,
        builder.getArrayAttr(
            AccessPatternAttr::GetIdentity(op.getContext(), domain.size())),
        new_op.getResult(), domain_shape, copy_op.memory_spaceAttr());
    copy_op.replaceAllUsesWith(new_result);
    copy_op.erase();
  }

  return mlir::success();
}

// Rematerializes loops in all compute operations in the given program.
mlir::LogicalResult RematerializeInProgram(SairProgramOp op) {
  llvm::DenseMap<mlir::Attribute, LoopBounds> main_loops;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<size_t, 2>>
      pending_rematerializations;

  // Perform a single walk across the program to collect both the information
  // about actual loop bounds and the information about dimensions that require
  // rematerialization.
  op.walk([&main_loops, &pending_rematerializations](ComputeOp comp) {
    if (!comp.loop_nest()) return;

    llvm::ArrayRef<mlir::Attribute> loopAttrRange = comp.LoopNestLoops();
    for (size_t i = 0, e = loopAttrRange.size(); i < e; ++i) {
      auto loop = loopAttrRange[i].cast<LoopAttr>();
      if (loop.iter().Rematerialize()) {
        pending_rematerializations[comp.getOperation()].push_back(i);
        continue;
      }

      int dimension = loop.iter().Dimension();
      auto sair_op = cast<SairOp>(comp.getOperation());
      Value range = sair_op.domain()[dimension];
      bool is_dependent =
          sair_op.shape().Dimensions()[dimension].DependencyMask().any();
      main_loops.try_emplace(loop.name(), range, loop.iter().Step(),
                             is_dependent);
    }
  });

  // Rematrialize dimensions in each op where it is necessary. This operates on
  // all dimensions of an op simultaneously because the op is erased in the
  // process and we don't want to keep track of that.
  for (const auto &rematerialization : pending_rematerializations) {
    if (mlir::failed(
            Rematerialize(cast<ComputeOp>(rematerialization.getFirst()),
                          rematerialization.getSecond(), main_loops))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Pass that exercises rematerialization on Sair programs.
class RematerializePass : public RematerializePassBase<RematerializePass> {
 public:
  void runOnFunction() override {
    getFunction().walk([this](SairProgramOp program) {
      if (mlir::failed(RematerializeInProgram(program)))
        return signalPassFailure();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateRematerializePass() {
  return std::make_unique<RematerializePass>();
}

}  // namespace sair
