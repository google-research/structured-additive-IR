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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
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
#include "sair_dialect.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "sequence.h"
#include "storage.h"

namespace sair {

#define GEN_PASS_DEF_INTRODUCELOOPSPASS
#include "transforms/lowering.h.inc"

namespace {

// Adds canonicalization patterns from Ops to `list.
template <typename... Ops>
void getAllPatterns(mlir::RewritePatternSet &list, mlir::MLIRContext *ctx) {
  (Ops::getCanonicalizationPatterns(list, ctx), ...);
}

// Keeps track of the operations to process to introduce loops. Maintains two
// work lists, one for sair operations to canonicalize and one for sair.map
// operations to lower. Keeps `sequence_analysis` updated with op additions and
// deletions using the pattern rewriter notification hooks: the sequence numbers
// of compute operations are updated in the analysis on every addition and
// deletion. Note that the op attributes are not updated until `AssignInferred`
// is called on `sequence_analysis`.
class Driver : public mlir::PatternRewriter {
 public:
  Driver(mlir::MLIRContext *ctx, SequenceAnalysis &sequence_analysis)
      : PatternRewriter(ctx),
        canonicalization_patterns_(ctx),
        sequence_analysis_(sequence_analysis) {
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

      for (const auto &pattern :
           canonicalization_patterns_.getNativePatterns()) {
        if (pattern->getRootKind().has_value() &&
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

    // This hook is called after the op has been inserted in the block so we can
    // obtain the "reference" operation as next in the block. It should always
    // exist since we are never inserting after the terminator, and it should be
    // a SairOp if the inserted operation is a SairOp because only such ops are
    // allowed in a SairProgram.
    if (auto compute_op = dyn_cast<ComputeOp>(op)) {
      SairOp next_op = cast<SairOp>(op->getNextNode());
      sequence_analysis_.Insert(ComputeOpInstance::Unique(compute_op),
                                OpInstance::Unique(next_op),
                                Direction::kBefore);
    }
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

    if (auto compute_op = dyn_cast<ComputeOp>(op)) {
      sequence_analysis_.Erase(ComputeOpInstance::Unique(compute_op));
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

  mlir::RewritePatternSet canonicalization_patterns_;
  llvm::SetVector<mlir::Operation *> simplify_work_list_;
  llvm::SetVector<mlir::Operation *> map_ops_work_list_;

  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>>
      pending_updates_;

  SequenceAnalysis &sequence_analysis_;
};

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
mlir::LogicalResult RegisterOperations(
    SairProgramOp program, const SequenceAnalysis &sequence_analysis,
    Driver &driver) {
  // Add non-compute ops. These will be canonicalized by the driver and their
  // relative order doesn't matter.
  for (mlir::Operation &operation : program.getBody().front()) {
    if (isa<SairProjAnyOp>(operation)) {
      return operation.emitError() << "sair.proj_any operations must be "
                                      "eliminated before introducing loops";
    }
    if (!SairOp(&operation).HasExactlyOneInstance()) {
      return operation.emitError() << "operations must have exactly one "
                                      "instance when introducing loops";
    }
    // Compute ops have been added above.
    if (isa<ComputeOp>(operation)) continue;
    driver.AddOperation(&operation);
  }

  // Then add compute operations is their sequence order. The order is
  // crucially important because Sair ops may be sequenced differently from
  // their "natural" order in the block, but the loops must be created in the
  // proper order in the block. The order of non-compute ops doesn't matter
  // because they don't result in any executable code at this stage.
  for (ComputeOpInstance op : sequence_analysis.Ops()) {
    mlir::Operation *operation = op.GetDuplicatedOp();
    driver.AddOperation(operation);
    SairMapOp map_op = dyn_cast<SairMapOp>(operation);
    if (map_op == nullptr) {
      return operation->emitError() << "operation must be lowered to sair.map";
    }

    DecisionsAttr decisions = op.GetDecisions();
    if (decisions.loop_nest() == nullptr) {
      return map_op.emitError() << "missing loop_nest attribute";
    }

    for (mlir::Attribute attr : op.Loops()) {
      LoopAttr loop = attr.cast<LoopAttr>();
      if (!loop.iter().isa<MappingDimExpr>()) {
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
  llvm::append_range(new_range, range.take_front(position));
  llvm::append_range(new_range, range.drop_front(position + 1));
  return new_range;
}

// Erases a dimension from the use domain of the mapping. If the
// dimension is mapped to a dimension of the def domain, the dimension from the
// def domain is also removed.
MappingAttr EraseDimension(MappingAttr mapping, int dimension) {
  mlir::SmallVector<MappingExpr, 4> dimensions;
  for (MappingExpr expr : mapping) {
    MappingDimExpr dim_expr = expr.cast<MappingDimExpr>();
    if (dim_expr.dimension() < dimension) {
      dimensions.push_back(expr);
      continue;
    }
    if (dim_expr.dimension() == dimension) continue;
    dimensions.push_back(
        MappingDimExpr::get(dim_expr.dimension() - 1, expr.getContext()));
  }
  return MappingAttr::get(mapping.getContext(), mapping.UseDomainSize() - 1,
                          dimensions);
}

// Erases a dimension for a shape attribute. Remaining dimensions must not
// depend on the removed dimension.
DomainShapeAttr EraseDimension(DomainShapeAttr shape, int dimension) {
  llvm::SmallVector<DomainShapeDim, 4> shape_dimensions;
  shape_dimensions.reserve(shape.NumDimensions() - 1);
  llvm::append_range(shape_dimensions,
                     shape.Dimensions().take_front(dimension));

  for (auto shape_dim : shape.Dimensions().drop_front(dimension + 1)) {
    assert(!shape_dim.DependencyMask().test(dimension));
    shape_dimensions.emplace_back(
        shape_dim.type(),
        EraseDimension(shape_dim.dependency_mapping(), dimension));
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
    res.push_back(ValueType::get(shape, value_type.ElementType()));
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
    // This cast is always legal as `RegisterOperations` checks that loop
    // iterators are MappingDimExprs.
    int old_dimension = loop.iter().cast<MappingDimExpr>().dimension();
    if (old_dimension < dimension) {
      new_loop_nest.push_back(loop);
      continue;
    }
    new_loop_nest.push_back(LoopAttr::get(
        loop.name(), MappingDimExpr::get(old_dimension - 1, context),
        loop.unroll(), context));
  }
  return mlir::ArrayAttr::get(context, new_loop_nest);
}

// Erases a projection dimension from a proj_last operation and replaces
// `op.value()` by `new_value`.
void EraseDimension(SairProjLastOp op, int dimension, mlir::Value new_value,
                    Driver &driver) {
  mlir::OpBuilder::InsertionGuard guard(driver);
  assert(dimension >= op.getParallelDomain().size());

  int dim_pos = dimension - op.getParallelDomain().size();
  MappingAttr mapping = EraseDimension(op.Value().Mapping(), dimension);

  driver.setInsertionPoint(op);
  driver.replaceOpWithNewOp<SairProjLastOp>(
      op, /*result_type=*/op.getType(),
      /*parallel_domain=*/op.getParallelDomain(),
      /*projection_domain=*/EraseValue(op.getProjectionDomain(), dim_pos),
      /*mapping_array*/ driver.getArrayAttr({mapping}),
      /*value=*/new_value,
      /*shape=*/EraseDimension(op.getShape(), dimension),
      /*instances=*/EraseOperandFromDecisions(op.getInstancesAttr(), dimension),
      /*copies=*/nullptr);
}

// Erases a sequential dimension from a sair.fby operation and replaces
// `op.value()` by `new_value`.
void EraseDimension(SairFbyOp op, int dimension, mlir::Value new_value,
                    Driver &driver) {
  mlir::OpBuilder::InsertionGuard guard(driver);
  assert(dimension >= op.getParallelDomain().size());

  ValueType type = op.getType().cast<ValueType>();
  mlir::Type element_type = type.ElementType();
  DomainShapeAttr shape = EraseDimension(type.Shape(), dimension);
  int dim_pos = dimension - op.getParallelDomain().size();
  MappingAttr mapping = EraseDimension(op.Value().Mapping(), dimension);

  driver.setInsertionPoint(op);
  driver.replaceOpWithNewOp<SairFbyOp>(
      op,
      /*result_type=*/ValueType::get(shape, element_type),
      /*parallel_domain=*/op.getParallelDomain(),
      /*sequential_domain=*/EraseValue(op.getSequentialDomain(), dim_pos),
      /*mapping_array*/
      driver.getArrayAttr({op.Init().Mapping(), mapping}),
      /*init=*/op.getInit(), /*value=*/new_value,
      /*instances=*/EraseOperandFromDecisions(op.getInstancesAttr(), dimension),
      /*copies=*/nullptr);
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
      driver.create<mlir::arith::ConstantIndexOp>(loc, step.getSExtValue());
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
  return driver.create<mlir::arith::ConstantOp>(loc, type, value);
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

    MappingAttr mapping = user.ValueOperands()[operand_position].Mapping();
    int user_dimension =
        mapping.Dimension(dimension).cast<MappingDimExpr>().dimension();

    if (auto proj_last = dyn_cast<SairProjLastOp>(use.getOwner())) {
      EraseDimension(proj_last, user_dimension, new_value, driver);
      continue;
    }

    SairFbyOp fby_op = cast<SairFbyOp>(use.getOwner());
    assert(fby_op.getValue() == old_value);

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
mlir::LogicalResult IntroduceLoop(SairMapOp op,
                                  const StorageAnalysis &storage_analysis,
                                  Driver &driver) {
  auto *sair_dialect = static_cast<SairDialect *>(op->getDialect());
  llvm::ArrayRef<mlir::Attribute> loop_nest =
      ComputeOpInstance::Unique(cast<ComputeOp>(op.getOperation())).Loops();
  LoopAttr loop = loop_nest.back().cast<LoopAttr>();

  int dimension = loop.iter().cast<MappingDimExpr>().dimension();
  mlir::Operation *dimension_op = op.getDomain()[dimension].getDefiningOp();
  if (isa<SairPlaceholderOp>(dimension_op)) {
    return dimension_op->emitError()
           << "placeholders must be replaced by actual dimensions before "
              "introducing loops";
  }

  RangeOp range = cast<RangeOp>(dimension_op);
  MappingAttr range_mapping =
      op.getShape().Dimension(dimension).dependency_mapping().ResizeUseDomain(
          op.getDomain().size() - 1);

  // Get the inputs of the new operation.
  llvm::SmallVector<mlir::Value, 4> inputs = op.getInputs();
  llvm::SmallVector<mlir::Attribute, 4> mappings;
  mappings.reserve(mappings.size());
  for (mlir::Attribute attr : op.getMappingArray()) {
    MappingAttr mapping = attr.cast<MappingAttr>();
    mappings.push_back(EraseDimension(mapping, dimension));
  }

  // Retrieve the loop size.
  driver.setInsertionPointToStart(&op.block());
  auto materialize_bound = [&](const ValueOrConstant &bound) -> mlir::Value {
    if (bound.is_constant()) {
      return driver.create<arith::ConstantOp>(op.getLoc(), bound.constant());
    }
    // Check that the value is stored in registers.
    auto bound_instance = ResultInstance::Unique(bound.value().value);
    if (storage_analysis.GetStorage(bound_instance).space() !=
        sair_dialect->register_attr()) {
      // TODO(b/174127497): ensure that value stored in registers are produced
      // in the same loop nest.
      range.emitError() << "range bounds must be stored in registers";
      return nullptr;
    }

    inputs.push_back(bound.value().value);
    mappings.push_back(range_mapping.Compose(bound.value().mapping));
    return op.block().addArgument(driver.getIndexType(), op.getLoc());
  };

  mlir::Value upper_bound = materialize_bound(range.UpperBound());
  mlir::Value lower_bound = materialize_bound(range.LowerBound());
  if (upper_bound == nullptr || lower_bound == nullptr) return mlir::failure();
  llvm::APInt step(64, range.Step());
  mlir::Block::iterator for_insertion_point = driver.getInsertionPoint();

  // Create the new sair.map operation.
  driver.setInsertionPoint(op);
  mlir::ArrayAttr new_loop_nest = EraseDimensionFromLoopNest(
      loop_nest.drop_back(), dimension, driver.getContext());
  DecisionsAttr decisions = op.GetDecisions(0);
  auto new_decisions = DecisionsAttr::get(
      /*sequence=*/decisions.sequence(),
      /*loop_nest=*/new_loop_nest,
      /*storage=*/decisions.storage(),
      /*expansion=*/decisions.expansion(),
      /*copy_of=*/nullptr,
      /*operands=*/EraseOperandFromArray(decisions.operands(), dimension),
      op.getContext());
  SairMapOp new_op = driver.create<SairMapOp>(
      op.getLoc(),
      /*result_types=*/EraseDimension(op.getResultTypes(), dimension),
      /*domain=*/EraseValue(op.getDomain(), dimension),
      /*mappings_array=*/driver.getArrayAttr(mappings),
      /*inputs=*/inputs,
      /*shape=*/EraseDimension(op.getShape(), dimension),
      /*instances=*/driver.getArrayAttr({new_decisions}),
      /*copies=*/nullptr);
  new_op.getBody().takeBody(op.getBody());

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
        new_op.block().getArgument(operand.position() + op.getDomain().size());
    iter_args_init.push_back(value);
    iter_args.push_back(value);

    // Find the corresponding result.
    int result_pos = -1;
    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
      if (fby.getValue() != op.getResult(i)) continue;
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
  mlir::Value old_index = new_op.getBody().getArgument(dimension);
  mlir::scf::ForOp for_op = CreateForOp(
      op.getLoc(), lower_bound, upper_bound, step, old_index, iter_args_init,
      iter_args, iter_args_result, results_pos, driver);
  if (loop.unroll()) {
    if (mlir::failed(mlir::loopUnrollByFactor(
            for_op, loop.unroll().getValue().getZExtValue())))
      return failure();
  }
  new_op.getBody().eraseArgument(dimension);

  // Erase the old operation.
  driver.eraseOp(op);
  return mlir::success();
}

// Fuses two sair.map operations. They must have the same loops in the loop_nest
// attribute.
void Fuse(SairMapOp first_op, llvm::ArrayRef<mlir::Attribute> first_loop_nest,
          SairMapOp second_op, llvm::ArrayRef<mlir::Attribute> second_loop_nest,
          Driver &driver) {
  mlir::OpBuilder::InsertionGuard insertion_guard(driver);
  mlir::MLIRContext *context = driver.getContext();

  llvm::SmallVector<mlir::Value, 4> second_block_args;
  llvm::SmallVector<mlir::Value, 4> inputs;
  llvm::SmallVector<mlir::Attribute, 4> mappings;

  mlir::Operation *first_return = first_op.block().getTerminator();
  mlir::Operation *second_return = second_op.block().getTerminator();

  // Map loop indexes of second_op to loop indexes of first_op.
  llvm::SmallVector<MappingExpr, 4> first_to_second_mapping;
  first_to_second_mapping.append(second_op.getDomain().size(),
                                 MappingNoneExpr::get(context));
  second_block_args.append(second_op.getDomain().size(), nullptr);
  for (auto [first_attr, second_attr] :
       llvm::zip(first_loop_nest, second_loop_nest)) {
    MappingExpr first_iter = first_attr.cast<LoopAttr>().iter();
    int first_dimension = first_iter.cast<MappingDimExpr>().dimension();
    int second_dimension =
        second_attr.cast<LoopAttr>().iter().cast<MappingDimExpr>().dimension();
    first_to_second_mapping[second_dimension] = first_iter;
    second_block_args[second_dimension] =
        first_op.block().getArgument(first_dimension);
  }

  // Gather operands for the new operation.
  llvm::append_range(inputs, first_op.getInputs());
  MappingAttr first_to_second_mapping_attr =
      MappingAttr::get(driver.getContext(), first_op.getDomain().size(),
                       first_to_second_mapping);

  llvm::append_range(mappings, first_op.getMappingArray());
  for (ValueOperand operand : second_op.ValueOperands()) {
    if (operand.value().getDefiningOp() == first_op) {
      auto it = llvm::find(first_op.getResults(), operand.value());
      int return_pos = std::distance(first_op.result_begin(), it);
      second_block_args.push_back(first_return->getOperand(return_pos));
      continue;
    }

    inputs.push_back(operand.value());
    mappings.push_back(first_to_second_mapping_attr.Compose(operand.Mapping()));
    mlir::Value block_argument = first_op.block().addArgument(
        operand.GetType().ElementType(), second_op.getLoc());
    second_block_args.push_back(block_argument);
  }

  // Create the new sair.return operation.
  int num_results = first_op.getNumResults() + second_op.getNumResults();
  llvm::SmallVector<mlir::Value, 4> returned_scalars;
  returned_scalars.reserve(num_results);
  llvm::append_range(returned_scalars, first_return->getOperands());
  llvm::append_range(returned_scalars, second_return->getOperands());
  driver.setInsertionPoint(second_return);
  driver.replaceOpWithNewOp<SairReturnOp>(second_return, returned_scalars);

  // Merge bodies.
  driver.eraseOp(first_return);
  driver.mergeBlocks(&second_op.block(), &first_op.block(), second_block_args);

  // Gather return types for the new sair.map operation.
  llvm::SmallVector<mlir::Type> result_types;
  result_types.reserve(num_results);
  llvm::append_range(result_types, first_op.getResultTypes());
  llvm::append_range(result_types, second_op.getResultTypes());

  // Gather memory space attributes for the new sair.map operation.
  llvm::SmallVector<mlir::Attribute> storages;
  storages.reserve(num_results);
  auto append_storages = [&](SairMapOp op) {
    DecisionsAttr decisions = op.GetDecisions(0);
    if (decisions.storage() == nullptr) {
      storages.append(op.getNumResults(), driver.getUnitAttr());
    } else {
      llvm::append_range(storages, decisions.storage().getValue());
    }
  };
  append_storages(first_op);
  append_storages(second_op);

  // Create the operation.
  driver.setInsertionPoint(second_op);
  DecisionsAttr first_decisions = first_op.GetDecisions(0);
  auto new_decisions = DecisionsAttr::get(
      /*sequence=*/first_decisions.sequence(),
      /*loop_nest=*/first_decisions.loop_nest(),
      /*storage=*/driver.getArrayAttr(storages),
      /*expansion=*/first_decisions.expansion(),
      /*copy_of=*/first_decisions.copy_of(),
      /*operands=*/
      GetInstanceZeroOperands(context,
                              first_op.getDomain().size() + inputs.size()),
      context);
  SairMapOp new_op = driver.create<SairMapOp>(
      /*location=*/first_op.getLoc(),
      /*result_types=*/result_types,
      /*domain=*/first_op.getDomain(),
      /*mappings_array=*/driver.getArrayAttr(mappings),
      /*inputs=*/inputs,
      /*shape=*/first_op.getShape(),
      /*instances=*/driver.getArrayAttr({new_decisions}),
      /*copies=*/nullptr);
  new_op.getBody().takeBody(first_op.getBody());
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
mlir::LogicalResult IntroduceLoopOrFuse(
    SairMapOp op, const StorageAnalysis &storage_analysis,
    const SequenceAnalysis &sequence_analysis, Driver &driver) {
  auto op_instance =
      ComputeOpInstance::Unique(cast<ComputeOp>(op.getOperation()));
  ComputeOpInstance prev_op = sequence_analysis.PrevOp(op_instance);
  ComputeOpInstance next_op = sequence_analysis.NextOp(op_instance);
  mlir::ArrayAttr curr_loop_nest = op_instance.GetDecisions().loop_nest();
  mlir::ArrayAttr prev_loop_nest =
      prev_op == nullptr ? nullptr : prev_op.GetDecisions().loop_nest();
  mlir::ArrayAttr next_loop_nest =
      next_op == nullptr ? nullptr : next_op.GetDecisions().loop_nest();

  if (CanFuse(prev_loop_nest, curr_loop_nest)) {
    auto prev_map_op = cast<SairMapOp>(prev_op.GetDuplicatedOp());
    Fuse(prev_map_op, prev_loop_nest.getValue(), op, curr_loop_nest.getValue(),
         driver);
  } else if (CanFuse(curr_loop_nest, next_loop_nest)) {
    auto next_map_op = cast<SairMapOp>(next_op.GetDuplicatedOp());
    Fuse(op, curr_loop_nest.getValue(), next_map_op, next_loop_nest.getValue(),
         driver);
  } else if (!curr_loop_nest.empty() &&
             !IsPrefix(curr_loop_nest, prev_loop_nest) &&
             !IsPrefix(curr_loop_nest, next_loop_nest)) {
    return IntroduceLoop(op, storage_analysis, driver);
  }

  return mlir::success();
}

// Replaces iteration dimensions in sair.map and sair.map_reduce operation by
// loops, converting sair.map_reduce operation into sair.map operations in the
// process. Fails if operations operand depend on any dimension,  if operations
// have results with more than 1 dimension or if dimensions are not defined in
// the same sair.program.
class IntroduceLoops : public impl::IntroduceLoopsPassBase<IntroduceLoops> {
  // Introduce loops for a sair.program operation.
  void IntroduceProgramLoops(SairProgramOp program) {
    auto &sequence_analysis = getChildAnalysis<SequenceAnalysis>(program);
    Driver driver(&getContext(), sequence_analysis);
    auto storage_analysis = getChildAnalysis<StorageAnalysis>(program);
    if (mlir::failed(RegisterOperations(program, sequence_analysis, driver))) {
      signalPassFailure();
      return;
    }

    driver.Simplify();

    while (SairMapOp op = driver.PopMapOp()) {
      if (mlir::failed(IntroduceLoopOrFuse(op, storage_analysis,
                                           sequence_analysis, driver))) {
        signalPassFailure();
        return;
      }

      driver.Simplify();
    }
  }

  void runOnOperation() override {
    // Retreive a a sorted list of SairMap operations.
    getOperation().walk([&](SairProgramOp op) { IntroduceProgramLoops(op); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIntroduceLoopsPass() {
  return std::make_unique<IntroduceLoops>();
}

}  // namespace sair
