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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "transforms/lowering_pass_classes.h"
#include "utils.h"

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

// Creates Sair value types with the same elemental types as those in the given
// range, and with the given shape. Appends these new types to result.
void AdaptTypesToShape(mlir::TypeRange types, DomainShapeAttr shape,
                       llvm::SmallVectorImpl<mlir::Type> &result) {
  auto range = llvm::map_range(types, [shape](mlir::Type type) -> mlir::Type {
    return ValueType::get(shape.getContext(), shape,
                          type.cast<ValueType>().ElementType());
  });
  result.append(range.begin(), range.end());
}

// Creates a new access pattern array by shifting all the accessed dimensions
// starting from `insert_pos` right by `num_dims`. This reflects `num_dims`
// dimensions being inserted at `insert_pos` into the domain.
mlir::ArrayAttr AdaptAccessPatterns(mlir::ArrayAttr access_pattern_array,
                                    size_t insert_pos, size_t num_dims) {
  llvm::SmallVector<mlir::Attribute, 4> new_access_patterns;
  new_access_patterns.reserve(access_pattern_array.size());
  for (auto access_pattern :
       access_pattern_array.getAsRange<AccessPatternAttr>()) {
    new_access_patterns.push_back(
        access_pattern.ShiftRight(num_dims, insert_pos));
  }
  return mlir::ArrayAttr::get(new_access_patterns,
                              access_pattern_array.getContext());
}

// Moves the body of the source operation to the target operation and inserts
// `num` block arguments of the given type at `pos`.
template <typename OpTy>
OpTy TakeBodyAdjustArguments(OpTy target, OpTy source, int pos, int num,
                             mlir::Type type) {
  target.body().takeBody(source.body());
  for (size_t i = 0; i < num; ++i) {
    target.body().front().insertArgument(pos, type);
  }
  return target;
}

// Creates a new sair.copy operation that is intended to replace `op`. Takes the
// additional domain dimensions, the updated result type and loop nest attribute
// supplied as arguments, extracts the value being copied and the access pattern
// from `op`,
SairCopyOp RecreateOp(SairCopyOp op, mlir::TypeRange result_types,
                      mlir::ValueRange extra_domain,
                      mlir::ArrayAttr loop_nest_attr,
                      DomainShapeAttr domain_shape, mlir::OpBuilder &builder) {
  assert(result_types.size() == 1);
  auto domain = llvm::to_vector<8>(op.domain());
  appendRange(domain, extra_domain);
  return builder.create<SairCopyOp>(op.getLoc(), result_types[0], domain,
                                    op.access_pattern_array(), op.value(),
                                    loop_nest_attr, op.memory_spaceAttr());
}

// Creates a new sair.map operation that is intended to replace `op`. Takes the
// additional domain dimensions, the updated result types and the loop nest
// attribute supplied as arguments; moves the body and copies the access
// patterns from `op`.
SairMapOp RecreateOp(SairMapOp op, mlir::TypeRange result_types,
                     mlir::ValueRange extra_domain,
                     mlir::ArrayAttr loop_nest_attr,
                     DomainShapeAttr domain_shape, mlir::OpBuilder &builder) {
  auto domain = llvm::to_vector<8>(op.domain());
  appendRange(domain, extra_domain);
  auto new_op = builder.create<SairMapOp>(
      op.getLoc(), result_types, domain, op.access_pattern_array(), op.inputs(),
      domain_shape, loop_nest_attr, op.memory_spaceAttr());

  return TakeBodyAdjustArguments(new_op, op, op.domain().size(),
                                 extra_domain.size(), builder.getIndexType());
}

// Creates a new sair.map_reduce operation that is intended to replace `op`.
// Takes the additional parallel domain dimensions, the updated result types and
// the loop nest attribute supplied as arguments; moves the body and copies the
// reduction domain from `op`; takes the access patterns from `op` and changes
// them to account for the inserted parallel dimensions.
SairMapReduceOp RecreateOp(SairMapReduceOp op, mlir::TypeRange result_types,
                           mlir::ValueRange extra_domain,
                           mlir::ArrayAttr loop_nest_attr,
                           DomainShapeAttr domain_shape,
                           mlir::OpBuilder &builder) {
  auto parallel_domain = llvm::to_vector<8>(op.parallel_domain());
  mlir::ArrayAttr access_pattern_attr = AdaptAccessPatterns(
      op.access_pattern_array(), parallel_domain.size(), extra_domain.size());
  appendRange(parallel_domain, extra_domain);

  auto new_op = builder.create<SairMapReduceOp>(
      op.getLoc(), result_types, parallel_domain, op.reduction_domain(),
      access_pattern_attr, op.inits(), op.inputs(), domain_shape,
      loop_nest_attr, op.memory_spaceAttr());

  return TakeBodyAdjustArguments(new_op, op, op.parallel_domain().size(),
                                 extra_domain.size(), builder.getIndexType());
}

// Returns the operand range containing parallel domain dimensions.
mlir::Operation::operand_range ParallelDomain(SairOp op) {
  if (isa<SairCopyOp, SairMapOp>(op.getOperation())) {
    return op.domain();
  } else if (auto map_reduce = dyn_cast<SairMapReduceOp>(op.getOperation())) {
    return map_reduce.parallel_domain();
  }
  llvm_unreachable("unsupported sair op");
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

  // Keep the parallel domain and store the operand position to use for new
  // domain dimensions about to be inserted.
  auto parallel_domain = ParallelDomain(sair_op);
  size_t position = parallel_domain.size();

  // Rebuild the loop nest attribute and populate the list of extra domain
  // dimensions.
  auto loop_nest_array = llvm::to_vector<4>(op.LoopNestLoops());
  llvm::SmallVector<mlir::Value, 4> extra_domain;
  extra_domain.reserve(loops.size());
  for (size_t i = 0, e = loop_nest_array.size(); i < e; ++i) {
    // If we are inserting domain dimensions in the middle of the dimension
    // list, update the indices of trailing dimensions.
    auto loop = loop_nest_array[i].cast<LoopAttr>();
    if (!loop.iter().Rematerialize()) {
      if (loop.iter().Dimension() >= parallel_domain.size()) {
        loop_nest_array[i] = LoopAttr::get(
            loop.name(),
            IteratorAttr::get(ctx, loop.iter().Dimension() + loops.size(),
                              loop.iter().Step()),
            ctx);
      }
      continue;
    }
    if (!llvm::is_contained(loops, i)) continue;

    // For each loop to rematerialize, add the range as the last domain argument
    // and update the loop nest attribute accordingly.
    auto bounds_iterator = main_loops.find(loop.name());
    assert(bounds_iterator != main_loops.end() &&
           "invalid loop_nest attribute");
    const LoopBounds &bounds = bounds_iterator->getSecond();
    extra_domain.push_back(bounds.range);

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

    loop_nest_array[i] = LoopAttr::get(
        loop.name(), IteratorAttr::get(ctx, position++, bounds.step), ctx);
  }

  // Expand the shape accordingly.
  // TODO: this assumes we can only rematerialize independent dimensions. In the
  // future, we should also pull the dimensions it depends on.
  auto extra_domain_shape =
      DomainShapeAttr::HyperRectangular(op.getContext(), loops.size());
  DomainShapeAttr orig_op_shape = sair_op.shape();
  DomainShapeAttr domain_shape =
      orig_op_shape.ProductAt(parallel_domain.size(), extra_domain_shape);
  DomainShapeAttr result_shape =
      orig_op_shape.Prefix(parallel_domain.size()).Product(extra_domain_shape);

  mlir::Operation *orig_operation = op.getOperation();
  llvm::SmallVector<mlir::Type, 2> new_types;
  new_types.reserve(orig_operation->getNumResults());
  AdaptTypesToShape(orig_operation->getResultTypes(), result_shape, new_types);

  OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  mlir::Operation *new_operation =
      llvm::TypeSwitch<mlir::Operation *, mlir::Operation *>(orig_operation)
          .Case<SairCopyOp, SairMapOp, SairMapReduceOp>([&](auto orig_op) {
            auto new_op = RecreateOp(orig_op, new_types, extra_domain,
                                     builder.getArrayAttr(loop_nest_array),
                                     domain_shape, builder);
            return new_op.getOperation();
          })
          .Default([](mlir::Operation *op) { return nullptr; });
  if (!new_operation) return mlir::failure();

  // Project out the rematerialized dimensions from all results. Use the
  // identity access pattern here since defs and uses conserved their
  // patterns.
  auto value_producer = cast<ValueProducerOp>(orig_operation);
  for (unsigned i = 0, e = new_types.size(); i < e; ++i) {
    mlir::Value orig_result = orig_operation->getResult(i);
    mlir::Value remat_result = new_operation->getResult(i);

    // Construct the new shape from that of the original result rather
    // than the operation shape to avoid including reduction dimensions.
    DomainShapeAttr orig_result_shape =
        orig_result.getType().cast<ValueType>().Shape();
    DomainShapeAttr shape = orig_result_shape.Product(extra_domain_shape);

    auto proj_op = builder.create<SairProjAnyOp>(
        op.getLoc(), orig_result.getType(), parallel_domain, extra_domain,
        builder.getArrayAttr(AccessPatternAttr::GetIdentity(
            op.getContext(), parallel_domain.size() + loops.size())),
        remat_result, shape, /*memory_space=*/nullptr);
    if (llvm::Optional<int> memory_space = value_producer.GetMemorySpace(i)) {
      proj_op.SetMemorySpace(i, memory_space);
    }
    orig_result.replaceAllUsesWith(proj_op.getResult());
  }

  op.erase();

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

    llvm::ArrayRef<mlir::Attribute> loop_attr_range = comp.LoopNestLoops();
    for (size_t i = 0, e = loop_attr_range.size(); i < e; ++i) {
      auto loop = loop_attr_range[i].cast<LoopAttr>();
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
