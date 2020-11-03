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

#include <list>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "transforms/lowering_pass_classes.h"
#include "utils.h"

namespace sair {
namespace {

// Adds canonicalization patterns from Ops to `list.
template <typename... Ops>
void getAllPatterns(mlir::OwningRewritePatternList &list, mlir::MLIRContext *ctx) {
  (Ops::getCanonicalizationPatterns(list, ctx), ...);
}

// Keeps track of the operations to process to introduce loops. Maintains two
// work lists, one for sair operations to canonicalize and one for sair.map
// operations to lower.
class Driver : public mlir::PatternRewriter {
 public:
  Driver(mlir::MLIRContext *ctx) : PatternRewriter(ctx) {
    getAllPatterns<
#define GET_OP_LIST
#include "sair_ops.cc.inc"
        >(canonicalization_patterns_, ctx);
  }

  // Applies canonicalization and dead-code elimination to operations that
  // changed since the last call to simplify.
  void Simplify() {
    while (!simplify_work_list_.empty()) {
      mlir::Operation *operation = *simplify_work_list_.begin();
      simplify_work_list_.erase(simplify_work_list_.begin());
      if (mlir::isOpTriviallyDead(operation)) {
        notifyOperationRemoved(operation);
        operation->erase();
        continue;
      }

      for (const auto &pattern : canonicalization_patterns_) {
        if (pattern->getRootKind().hasValue() &&
            pattern->getRootKind() != operation->getName()) {
          continue;
        }
        if (mlir::succeeded(pattern->matchAndRewrite(operation, *this))) {
          break;
        }
      }
    }
  }

  // Registers an operation to process.
  void AddOperation(mlir::Operation *operation) {
    if (!isa<SairOp>(operation)) return;
    simplify_work_list_.insert(operation);

    if (auto map_op = dyn_cast<SairMapOp>(operation)) {
      map_ops_work_list_.insert(operation);
    }
  }

  // Returns a sair.map operation to process.
  SairMapOp PopMapOp() {
    if (map_ops_work_list_.empty()) return nullptr;
    SairMapOp map_op = cast<SairMapOp>(*map_ops_work_list_.begin());
    map_ops_work_list_.erase(map_ops_work_list_.begin());
    return map_op;
  }

 private:
  // Hook called when a new operation is created.
  void notifyOperationInserted(mlir::Operation *op) override {
    AddOperation(op);
  }

  // Hook called when an opertion is erased. Removes the operation from work
  // lists and adds its operands instead.
  void notifyOperationRemoved(mlir::Operation *op) override {
    for (mlir::Value operand : op->getOperands()) {
      mlir::Operation *defining_op = operand.getDefiningOp();
      if (defining_op == nullptr) continue;
      AddOperation(defining_op);
    }

    simplify_work_list_.remove(op);
    map_ops_work_list_.remove(op);
    pending_updates_.erase(op);

    for (auto &[_, dependencies] : pending_updates_) {
      for (int i = 0; i < dependencies.size(); ++i) {
        if (dependencies[i] != op) continue;
        std::swap(dependencies[i], dependencies.back());
        dependencies.pop_back();
      }
    }
  }

  // Hook called when an operation is being replaced by an other. Adds users of
  // the operation to work lists.
  void notifyRootReplaced(mlir::Operation *op) override {
    for (mlir::Value result : op->getResults()) {
      for (mlir::Operation *user : result.getUsers()) {
        AddOperation(user);
      }
    }
  }

  // Hook called before an operation is updated in place. Saves the operations
  // that it depends on to add them to the work-list after the operation is
  // updated.
  void startRootUpdate(mlir::Operation *op) override {
    // Gather ops defining the operands of `op` and store them until the update
    // is finalized. Duplicates will be eliminated when operations are actually
    // added to the work list.
    llvm::SmallVector<mlir::Operation *, 4> dependencies;
    for (mlir::Value v : op->getOperands()) {
      mlir::Operation *defining_op = v.getDefiningOp();
      if (defining_op == nullptr) continue;
      dependencies.push_back(defining_op);
    }

    auto res = pending_updates_.insert({op, std::move(dependencies)});
    assert(res.second);
    (void)res;  // Avoid variable unused errors in release build.
  }

  // Hook called after an operation is update in place. Adds its previous
  // operands to the work list.
  void finalizeRootUpdate(mlir::Operation *op) override {
    auto it = pending_updates_.find(op);
    assert(it != pending_updates_.end());
    AddOperation(op);
    for (mlir::Operation *dependency : it->second) {
      AddOperation(dependency);
    }
    pending_updates_.erase(it);
  }

  // Hook called when an in-place update that was announced by `startRootUpdate`
  // is cancelled.
  void cancelRootUpdate(mlir::Operation *op) override {
    pending_updates_.erase(op);
  }

  mlir::OwningRewritePatternList canonicalization_patterns_;
  llvm::SetVector<mlir::Operation *> simplify_work_list_;
  llvm::SetVector<mlir::Operation *> map_ops_work_list_;

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      pending_updates_;
};

// Returns the first sair.map operation before `op`, if any.
SairMapOp PrevMapOp(SairMapOp op) {
  mlir::Operation *operation = op.getOperation()->getPrevNode();

  while (operation != nullptr) {
    SairMapOp map_op = dyn_cast<SairMapOp>(operation);
    if (map_op != nullptr) return map_op;
    operation = operation->getPrevNode();
  }
  return nullptr;
}

// Returns the first sair.map operation after `op`, if any.
SairMapOp NextMapOp(SairMapOp op) {
  mlir::Operation *operation = op.getOperation()->getNextNode();

  while (operation != nullptr) {
    SairMapOp map_op = dyn_cast<SairMapOp>(operation);
    if (map_op != nullptr) return map_op;
    operation = operation->getNextNode();
  }
  return nullptr;
}

// Indicates if `prefix` is a prefix of `array`. Returns `false` if any of the
// arrays is a null array.
bool IsPrefix(mlir::ArrayAttr prefix, mlir::ArrayAttr array) {
  if (prefix == nullptr || array == nullptr) return false;
  if (prefix == array) return true;
  if (prefix.size() >= array.size()) return false;

  for (int i = 0, e = prefix.size(); i < e; ++i) {
    if (prefix[i] != array[i]) return false;
  }

  return true;
}

// Registers operations of the sair.program in the driver. Checks that
// ComputeOps have been lowered to sair.map, that all sair.map have a
// `loop_nest` attribute set and that sair.proj_any operations are eliminated.
mlir::LogicalResult RegisterOperations(SairProgramOp program, Driver &driver) {
  mlir::Block &block = program.body().front();
  for (mlir::Operation &operation : block) {
    if (isa<SairProjAnyOp>(operation)) {
      return operation.emitError() << "sair.proj_any operations must be "
                                      "eliminated before introducing loops";
    }

    driver.AddOperation(&operation);
    if (!isa<ComputeOp>(operation)) continue;

    SairMapOp map_op = dyn_cast<SairMapOp>(operation);
    if (map_op == nullptr) {
      return operation.emitError() << "operation must be lowered to sair.map";
    }

    if (!map_op.loop_nest().hasValue()) {
      return map_op.emitError() << "missing loop_nest attribute";
    }

    for (mlir::Attribute attr : map_op.LoopNestLoops()) {
      LoopAttr loop = attr.cast<LoopAttr>();
      if (loop.iter().Rematerialize() || loop.iter().Step() != 1) {
        return map_op.emitError()
               << "loop must not rematerialize or be strip-mined";
      }
    }
  }

  return mlir::success();
}

// Helper function that erases a value from a range of values.
llvm::SmallVector<mlir::Value, 4> EraseValue(mlir::ValueRange range,
                                             int position) {
  llvm::SmallVector<mlir::Value, 4> new_range;
  new_range.reserve(range.size() - 1);
  appendRange(new_range, range.take_front(position));
  appendRange(new_range, range.drop_front(position + 1));
  return new_range;
}

// Erases a dimension from the use domain of the access pattern. If the
// dimension is mapped to a dimension of the def domain, the dimension from the
// def domain is also removed.
AccessPatternAttr EraseDimension(AccessPatternAttr access_pattern,
                                 int dimension) {
  mlir::SmallVector<int, 4> dimensions;
  for (int d : access_pattern) {
    if (d < dimension) {
      dimensions.push_back(d);
    } else if (d > dimension) {
      dimensions.push_back(d - 1);
    }
  }
  return AccessPatternAttr::get(access_pattern.getContext(),
                                access_pattern.UseDomainSize() - 1, dimensions);
}

// Erases a dimension for a shape attribute. Remaining dimensions must not
// depend on the removed dimension.
DomainShapeAttr EraseDimension(DomainShapeAttr shape, int dimension) {
  llvm::SmallVector<DomainShapeDim, 4> shape_dimensions;
  shape_dimensions.reserve(shape.NumDimensions() - 1);
  appendRange(shape_dimensions, shape.Dimensions().take_front(dimension));

  for (auto shape_dim : shape.Dimensions().drop_front(dimension + 1)) {
    assert(!shape_dim.DependencyMask().test(dimension));
    shape_dimensions.emplace_back(
        shape_dim.type(),
        EraseDimension(shape_dim.dependency_pattern(), dimension));
  }
  return DomainShapeAttr::get(shape.getContext(), shape_dimensions);
}

// Erases the given dimension from the shape of each type of a range of
// ValueType.
llvm::SmallVector<mlir::Type, 4> EraseDimension(mlir::TypeRange types,
                                                int dimension) {
  llvm::SmallVector<mlir::Type, 4> res;
  res.reserve(types.size());
  for (mlir::Type type : types) {
    ValueType value_type = type.cast<ValueType>();
    DomainShapeAttr shape = EraseDimension(value_type.Shape(), dimension);
    res.push_back(
        ValueType::get(type.getContext(), shape, value_type.ElementType()));
  }
  return res;
}

// Renames dimensions in the loop nest attribute to account for the fact that
// `dimension` was removed.
mlir::ArrayAttr EraseDimensionFromLoopNest(
    llvm::ArrayRef<mlir::Attribute> loop_nest, int dimension,
    mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::Attribute, 4> new_loop_nest;
  new_loop_nest.reserve(loop_nest.size());
  for (mlir::Attribute attr : loop_nest) {
    LoopAttr loop = attr.cast<LoopAttr>();
    if (loop.iter().Rematerialize() || loop.iter().Dimension() < dimension) {
      new_loop_nest.push_back(attr);
      continue;
    }

    assert(loop.iter().Dimension() != dimension);
    IteratorAttr iter = IteratorAttr::get(context, loop.iter().Dimension() - 1,
                                          loop.iter().Step());
    new_loop_nest.push_back(LoopAttr::get(loop.name(), iter, context));
  }
  return mlir::ArrayAttr::get(new_loop_nest, context);
}

// Erases a projection dimension from a proj_last operation and replaces
// `op.value()` by `new_value`.
void EraseDimension(SairProjLastOp op, int dimension, mlir::Value new_value,
                    Driver &driver) {
  mlir::OpBuilder::InsertionGuard guard(driver);
  assert(dimension >= op.parallel_domain().size());

  int dim_pos = dimension - op.parallel_domain().size();
  AccessPatternAttr access_pattern =
      EraseDimension(op.Value().AccessPattern(), dimension);

  driver.setInsertionPoint(op);
  driver.replaceOpWithNewOp<SairProjLastOp>(
      op, /*result_type=*/op.getType(),
      /*parallel_domain=*/op.parallel_domain(),
      /*projection_domain=*/EraseValue(op.projection_domain(), dim_pos),
      /*access_pattern_array*/ driver.getArrayAttr({access_pattern}),
      /*value=*/new_value,
      /*shape=*/EraseDimension(op.shape(), dimension),
      /*memory_space=*/op.memory_spaceAttr());
}

// Erases a sequential dimension from a sair.fby operation and replaces
// `op.value()` by `new_value`.
void EraseDimension(SairFbyOp op, int dimension, mlir::Value new_value,
                    Driver &driver) {
  mlir::OpBuilder::InsertionGuard guard(driver);
  assert(dimension >= op.parallel_domain().size());

  ValueType type = op.getType().cast<ValueType>();
  mlir::Type element_type = type.ElementType();
  DomainShapeAttr shape = EraseDimension(type.Shape(), dimension);
  int dim_pos = dimension - op.parallel_domain().size();
  AccessPatternAttr access_pattern =
      EraseDimension(op.Value().AccessPattern(), dimension);

  driver.setInsertionPoint(op);
  driver.replaceOpWithNewOp<SairFbyOp>(
      op,
      /*result_type=*/ValueType::get(op.getContext(), shape, element_type),
      /*parallel_domain=*/op.parallel_domain(),
      /*sequential_domain=*/EraseValue(op.sequential_domain(), dim_pos),
      /*access_pattern_array*/
      driver.getArrayAttr({op.Init().AccessPattern(), access_pattern}),
      /*init=*/op.init(), /*value=*/new_value,
      /*memory_space=*/op.memory_spaceAttr());
}

// Creates a for operation of size `size` at the current insertion point of
// `driver` and nests the rest of the current block, except the terminator, in
// the loop. Replaces `old_index` by the index of the loop. Replaces the result
// of the block at position `i` by the result of the scf.for operation at
// position `results_pos[i]`.
mlir::scf::ForOp CreateForOp(mlir::Location loc, mlir::Value lower_bound,
                             mlir::Value upper_bound, llvm::APInt step,
                             mlir::Value old_index,
                             mlir::ValueRange iter_args_init,
                             mlir::ValueRange iter_args,
                             mlir::ValueRange iter_args_result,
                             llvm::ArrayRef<int> results_pos, Driver &driver) {
  mlir::OpBuilder::InsertionGuard guard(driver);
  auto step_value =
      driver.create<mlir::ConstantIndexOp>(loc, step.getSExtValue());
  mlir::scf::ForOp for_op = driver.create<mlir::scf::ForOp>(
      loc, lower_bound, upper_bound, step_value, iter_args_init);

  // Replace the sair.map results.
  mlir::Block &block = *driver.getBlock();
  mlir::Operation *terminator = block.getTerminator();
  for (int i = 0, e = results_pos.size(); i < e; ++i) {
    terminator->setOperand(i, for_op.getResult(results_pos[i]));
  }

  // Move the loop body.
  mlir::Block::OpListType &for_body = for_op.getBody()->getOperations();
  for_body.splice(for_body.begin(), block.getOperations(),
                  mlir::Block::iterator(for_op.getOperation()->getNextNode()),
                  block.without_terminator().end());

  // The sair.yield operation is automatically created by the scf::ForOp
  // builder if there are no induction variables.
  if (!iter_args_result.empty()) {
    driver.setInsertionPointToEnd(for_op.getBody());
    driver.create<mlir::scf::YieldOp>(for_op.getLoc(), iter_args_result);
  }

  old_index.replaceAllUsesWith(for_op.getInductionVar());
  for (auto [old_value, new_value] :
       llvm::zip(iter_args, for_op.getRegionIterArgs())) {
    if (old_value == nullptr) continue;
    mlir::replaceAllUsesInRegionWith(old_value, new_value, for_op.getRegion());
  }

  return for_op;
}

// Use builder to create a variable of the given type. The variable value will
// not be used. Returns nullptr if the type is not an integer or float type.
mlir::Value GetValueOfType(mlir::Location loc, mlir::Type type,
                           Driver &driver) {
  mlir::Attribute value = driver.getZeroAttr(type);
  if (value == nullptr) {
    // TODO(b/169314813): use sair.undef instead.
    mlir::emitError(loc) << "unable to create a default value of type " << type;
    return nullptr;
  }
  return driver.create<mlir::ConstantOp>(loc, type, value);
}

// Updates users of a value after introducing a loop in the sair.map operation
// producing the value.
mlir::LogicalResult UpdateLoopUser(SairMapOp old_op, SairMapOp new_op,
                                   mlir::Value old_value, mlir::Value new_value,
                                   int dimension, Driver &driver) {
  // Use an early-inc iterator to avoid invalidating the iterator when
  // destroying uses.
  auto uses = llvm::make_early_inc_range(old_value.getUses());
  for (mlir::OpOperand &use : uses) {
    SairOp user = cast<SairOp>(use.getOwner());
    int operand_position = use.getOperandNumber() - user.domain().size();

    AccessPatternAttr access_pattern =
        user.ValueOperands()[operand_position].AccessPattern();
    int user_dimension = access_pattern.Dimension(dimension);

    if (auto proj_last = dyn_cast<SairProjLastOp>(use.getOwner())) {
      EraseDimension(proj_last, user_dimension, new_value, driver);
      continue;
    }

    SairFbyOp fby_op = cast<SairFbyOp>(use.getOwner());
    assert(fby_op.value() == old_value);

    for (mlir::Operation *fby_user : fby_op.getResult().getUsers()) {
      if (fby_user == old_op || fby_user == new_op) continue;
      // TODO(ulysse): update the insert copies pass to introduce copies
      return fby_user->emitError()
             << "insert copies between sair.fby and users located after "
                "producing loops before calling loop introduction";
    }

    EraseDimension(fby_op, user_dimension, new_value, driver);
  }

  return mlir::success();
}

// Replaces the innermost dimension of the domain by a loop.
mlir::LogicalResult IntroduceLoop(SairMapOp op, Driver &driver) {
  llvm::ArrayRef<mlir::Attribute> loop_nest = op.LoopNestLoops();
  LoopAttr loop = loop_nest.back().cast<LoopAttr>();

  int dimension = loop.iter().Dimension();
  RangeOp range = cast<RangeOp>(op.domain()[dimension].getDefiningOp());
  if (op.shape().Dimensions()[dimension].DependencyMask().any()) {
    return op.emitError()
           << "lowering dependent dimensions is not supported yet";
  }

  // Get the inputs of the new operation.
  llvm::SmallVector<mlir::Value, 4> inputs = op.inputs();
  llvm::SmallVector<mlir::Attribute, 4> access_patterns;
  access_patterns.reserve(access_patterns.size());
  for (mlir::Attribute attr : op.access_pattern_array()) {
    AccessPatternAttr access_pattern = attr.cast<AccessPatternAttr>();
    access_patterns.push_back(EraseDimension(access_pattern, dimension));
  }

  // Retrieve the loop size.
  driver.setInsertionPointToStart(&op.block());
  auto materialize_bound = [&](const ValueOrConstant &bound) -> mlir::Value {
    if (bound.is_constant()) {
      return driver.create<ConstantOp>(op.getLoc(), bound.constant());
    }
    inputs.push_back(bound.value());
    assert(bound.access_pattern().UseDomainSize() == 0);
    access_patterns.push_back(
        AccessPatternAttr::get(op.getContext(), op.domain().size() - 1, {}));
    return op.block().addArgument(driver.getIndexType());
  };

  mlir::Value upper_bound = materialize_bound(range.UpperBound());
  mlir::Value lower_bound = materialize_bound(range.LowerBound());
  llvm::APInt step = range.step();
  mlir::Block::iterator for_insertion_point = driver.getInsertionPoint();

  // Create the new sair.map operation.
  driver.setInsertionPoint(op);
  mlir::ArrayAttr new_loop_nest = EraseDimensionFromLoopNest(
      loop_nest.drop_back(), dimension, driver.getContext());
  SairMapOp new_op = driver.create<SairMapOp>(
      op.getLoc(),
      /*result_types=*/EraseDimension(op.getResultTypes(), dimension),
      /*domain=*/EraseValue(op.domain(), dimension),
      /*access_patterns_array=*/driver.getArrayAttr(access_patterns),
      /*inputs=*/inputs,
      /*shape=*/EraseDimension(op.shape(), dimension),
      /*loop_nest=*/new_loop_nest,
      /*memory_space=*/op.memory_spaceAttr());
  new_op.body().takeBody(op.body());

  // Position of the sair.map in the results of the scf.for operation.
  llvm::SmallVector<int, 4> results_pos(op.getNumResults(), -1);

  // Register sair.fby operands.
  llvm::SmallVector<mlir::Value, 4> iter_args_init;
  llvm::SmallVector<mlir::Value, 4> iter_args;
  llvm::SmallVector<mlir::Value, 4> iter_args_result;

  mlir::Operation *terminator = new_op.block().getTerminator();
  for (ValueOperand operand : op.ValueOperands()) {
    SairFbyOp fby = dyn_cast<SairFbyOp>(operand.value().getDefiningOp());
    if (fby == nullptr) continue;

    mlir::Value value =
        new_op.block().getArgument(operand.position() + op.domain().size());
    iter_args_init.push_back(value);
    iter_args.push_back(value);

    // Find the corresponding result.
    int result_pos = -1;
    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
      if (fby.value() != op.getResult(i)) continue;
      result_pos = i;
      break;
    }
    assert(result_pos >= 0);

    results_pos[result_pos] = iter_args_result.size();
    iter_args_result.push_back(terminator->getOperand(result_pos));
  }

  // Replace results.
  driver.setInsertionPoint(&new_op.block(), for_insertion_point);
  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    if (mlir::failed(UpdateLoopUser(op, new_op, op.getResult(i),
                                    new_op.getResult(i), dimension, driver))) {
      return mlir::failure();
    }
    // Use loop-carried values to project results out of the loop.
    if (results_pos[i] >= 0) continue;
    mlir::Type type = new_op.block().getTerminator()->getOperand(i).getType();
    mlir::Value init = GetValueOfType(op.getLoc(), type, driver);
    if (init == nullptr) return mlir::failure();

    mlir::Value result = new_op.block().getTerminator()->getOperand(i);
    iter_args_init.push_back(init);
    iter_args.push_back(nullptr);
    results_pos[i] = iter_args_result.size();
    iter_args_result.push_back(result);
  }

  // Create the scf.for operation.
  mlir::Value old_index = new_op.body().getArgument(dimension);
  CreateForOp(op.getLoc(), lower_bound, upper_bound, step, old_index,
              iter_args_init, iter_args, iter_args_result, results_pos, driver);
  new_op.body().eraseArgument(dimension);

  // Erase the old operation.
  driver.eraseOp(op);
  return mlir::success();
}

// Fuses two sair.map operations. They must have the same loops in the loop_nest
// attribute.
void Fuse(SairMapOp first_op, SairMapOp second_op, Driver &driver) {
  mlir::OpBuilder::InsertionGuard insertion_guard(driver);

  llvm::SmallVector<mlir::Value, 4> second_block_args;
  llvm::SmallVector<mlir::Value, 4> inputs;
  llvm::SmallVector<mlir::Attribute, 4> access_patterns;

  mlir::Operation *first_return = first_op.block().getTerminator();
  mlir::Operation *second_return = second_op.block().getTerminator();

  // Map loop indexes of second_op to loop indexes of first_op.
  llvm::SmallVector<int, 4> first_to_second_pattern;
  first_to_second_pattern.append(second_op.domain().size(),
                                 AccessPatternAttr::kNoDimension);
  second_block_args.append(second_op.domain().size(), nullptr);
  for (auto [first_attr, second_attr] :
       llvm::zip(first_op.LoopNestLoops(), second_op.LoopNestLoops())) {
    int first_dimension = first_attr.cast<LoopAttr>().iter().Dimension();
    int second_dimension = second_attr.cast<LoopAttr>().iter().Dimension();
    first_to_second_pattern[second_dimension] = first_dimension;
    second_block_args[second_dimension] =
        first_op.block().getArgument(first_dimension);
  }

  // Gather operands for the new operation.
  appendRange(inputs, first_op.inputs());
  AccessPatternAttr first_to_second_pattern_attr = AccessPatternAttr::get(
      driver.getContext(), first_op.domain().size(), first_to_second_pattern);

  appendRange(access_patterns, first_op.access_pattern_array());
  for (ValueOperand operand : second_op.ValueOperands()) {
    if (operand.value().getDefiningOp() == first_op) {
      auto it = llvm::find(first_op.getResults(), operand.value());
      int return_pos = std::distance(first_op.result_begin(), it);
      second_block_args.push_back(first_return->getOperand(return_pos));
      continue;
    }

    inputs.push_back(operand.value());
    access_patterns.push_back(
        first_to_second_pattern_attr.Compose(operand.AccessPattern()));
    mlir::Value block_argument =
        first_op.block().addArgument(operand.GetType().ElementType());
    second_block_args.push_back(block_argument);
  }

  // Create the new sair.return operation.
  int num_results = first_op.getNumResults() + second_op.getNumResults();
  llvm::SmallVector<mlir::Value, 4> returned_scalars;
  returned_scalars.reserve(num_results);
  appendRange(returned_scalars, first_return->getOperands());
  appendRange(returned_scalars, second_return->getOperands());
  driver.setInsertionPoint(second_return);
  driver.replaceOpWithNewOp<SairReturnOp>(second_return, returned_scalars);

  // Merge bodies.
  driver.eraseOp(first_return);
  driver.mergeBlocks(&second_op.block(), &first_op.block(), second_block_args);

  // Gather return types for the new sair.map operation.
  llvm::SmallVector<mlir::Type, 4> result_types;
  result_types.reserve(num_results);
  appendRange(result_types, first_op.getResultTypes());
  appendRange(result_types, second_op.getResultTypes());

  // Gather memory space attributes for the new sair.map operation.
  llvm::SmallVector<mlir::Attribute, 4> memory_spaces;
  memory_spaces.reserve(num_results);
  auto append_memory_spaces = [&](SairMapOp op) {
    if (op.memory_space().hasValue()) {
      appendRange(memory_spaces, op.memory_space().getValue().getValue());
    } else {
      memory_spaces.append(op.getNumResults(), driver.getUnitAttr());
    }
  };
  append_memory_spaces(first_op);
  append_memory_spaces(second_op);

  // Create the operation.
  driver.setInsertionPoint(second_op);
  SairMapOp new_op = driver.create<SairMapOp>(
      /*location=*/first_op.getLoc(),
      /*result_types=*/result_types,
      /*domain=*/first_op.domain(),
      /*access_patterns_array=*/driver.getArrayAttr(access_patterns),
      /*inputs=*/inputs,
      /*shape=*/first_op.shape(),
      /*loop_nest=*/first_op.loop_nestAttr(),
      /*memory_space=*/driver.getArrayAttr(memory_spaces));
  new_op.body().takeBody(first_op.body());
  driver.replaceOp(first_op,
                   new_op.getResults().take_front(first_op.getNumResults()));
  driver.replaceOp(second_op,
                   new_op.getResults().take_back(second_op.getNumResults()));
}

// Indicates if two loop nests can be fused.
bool CanFuse(mlir::ArrayAttr lhs, mlir::ArrayAttr rhs) {
  if (lhs == nullptr || rhs == nullptr) return false;
  if (lhs.empty() || rhs.empty()) return lhs == rhs;
  LoopAttr lhs_inner_loop = lhs.getValue().back().cast<LoopAttr>();
  LoopAttr rhs_inner_loop = rhs.getValue().back().cast<LoopAttr>();
  return lhs_inner_loop.name() == rhs_inner_loop.name();
}

// Introduces the innermost loop of `op` or fuse it with one of its immediate
// neigbors if possible.
mlir::LogicalResult IntroduceLoopOrFuse(SairMapOp op, Driver &driver) {
  SairMapOp prev_op = PrevMapOp(op);
  SairMapOp next_op = NextMapOp(op);
  mlir::ArrayAttr curr_loop_nest = op.loop_nest().getValue();
  mlir::ArrayAttr prev_loop_nest =
      prev_op == nullptr ? nullptr : prev_op.loop_nest().getValue();
  mlir::ArrayAttr next_loop_nest =
      next_op == nullptr ? nullptr : next_op.loop_nest().getValue();

  if (CanFuse(prev_loop_nest, curr_loop_nest)) {
    Fuse(prev_op, op, driver);
  } else if (CanFuse(curr_loop_nest, next_loop_nest)) {
    Fuse(op, next_op, driver);
  } else if (!curr_loop_nest.empty() &&
             !IsPrefix(curr_loop_nest, prev_loop_nest) &&
             !IsPrefix(curr_loop_nest, next_loop_nest)) {
    return IntroduceLoop(op, driver);
  }

  return mlir::success();
}

// Replaces iteration dimensions in sair.map and sair.map_reduce operation by
// loops, converting sair.map_reduce operation into sair.map operations in the
// process. Fails if operations operand depend on any dimension,  if operations
// have results with more than 1 dimension or if dimensions are not defined in
// the same sair.program.
class IntroduceLoops : public IntroduceLoopsPassBase<IntroduceLoops> {
  // Introduce loops for a sair.program operation.
  void IntroduceProgramLoops(SairProgramOp program) {
    Driver driver(&getContext());
    if (mlir::failed(RegisterOperations(program, driver))) {
      signalPassFailure();
      return;
    }

    driver.Simplify();

    while (SairMapOp op = driver.PopMapOp()) {
      if (mlir::failed(IntroduceLoopOrFuse(op, driver))) {
        signalPassFailure();
        return;
      }

      driver.Simplify();
    }
  }

  void runOnFunction() override {
    // Retreive a a sorted list of SairMap operations.
    getFunction().walk([&](SairProgramOp op) { IntroduceProgramLoops(op); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateIntroduceLoopsPass() {
  return std::make_unique<IntroduceLoops>();
}

}  // namespace sair
