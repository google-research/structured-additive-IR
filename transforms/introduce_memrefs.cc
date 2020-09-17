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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "transforms/default_lowering_attributes.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Returns the last operation using the value. Expects all users of the value to
// be in the same block that produces it. Returns nullptr if the value has no
// use.
mlir::Operation *GetLastUse(mlir::Value value) {
  auto it = value.getUsers().begin();
  if (it == value.getUsers().end()) return nullptr;
  mlir::Operation *last_use = *(it++);
  for (auto e = value.getUsers().end(); it != e; ++it) {
    if (last_use->isBeforeInBlock(*it)) {
      last_use = *it;
    }
  }
  return last_use;
}

// Position of an operation relative to another.
enum class Direction { kBefore, kAfter };

// Specifies where to insert an operation in the generated code. The operation
// is inserted before 'before', nested in 'loop_nest'.
struct InsertionPoint {
  mlir::Operation *operation;
  Direction direction;
  mlir::ArrayAttr loop_nest;

  // Sets the insertion point of the builder.
  void Set(mlir::OpBuilder &builder) const {
    if (direction == Direction::kAfter) {
      builder.setInsertionPointAfter(operation);
    } else {
      builder.setInsertionPoint(operation);
    }
  }
};

// Finds the closest insertion point of `point` for an operation with
// `num_dimensions` dimensions that is fused with the `fusion_level` first
// loops of `point`. `direction` indicates if the insertion point should be
// located before or after `point`.
//
// Leaves the loop nest blank in the case where the loop nest of `point` is
// blank.
InsertionPoint FindInsertionPoint(int num_dimensions, int fusion_level,
                                  ComputeOp point,
                                  Direction direction = Direction::kBefore) {
  SairProgramOp program_op = cast<SairProgramOp>(point.getParentOp());
  if (!point.loop_nest().hasValue()) return {point, direction, nullptr};

  mlir::ArrayAttr loop_nest = point.loop_nest().getValue();
  int current_fusion_level = loop_nest.size();

  mlir::Operation *current_op = point.getOperation();
  while (current_fusion_level > fusion_level) {
    current_op = direction == Direction::kAfter ? current_op->getNextNode()
                                                : current_op->getPrevNode();
    if (current_op == nullptr) break;

    ComputeOp compute_op = dyn_cast<ComputeOp>(current_op);
    if (compute_op == nullptr) continue;
    if (!compute_op.loop_nest().hasValue()) break;
    mlir::ArrayAttr new_loop_nest = compute_op.loop_nest().getValue();
    if (current_fusion_level > new_loop_nest.size()) {
      current_fusion_level = new_loop_nest.size();
    }
    for (; current_fusion_level > 0; --current_fusion_level) {
      mlir::StringAttr name =
          loop_nest[current_fusion_level - 1].cast<LoopAttr>().name();
      mlir::StringAttr new_name =
          new_loop_nest[current_fusion_level - 1].cast<LoopAttr>().name();
      if (name != new_name) break;
    }

    point = compute_op;
  }
  mlir::ArrayAttr new_loop_nest =
      GetDefaultLoopNest(program_op, num_dimensions,
                         loop_nest.getValue().take_front(fusion_level));
  return {point, direction, new_loop_nest};
}

// Materializes `operand` in `domain` by inserting a copy operation. Returns the
// copy operation inserted.
SairCopyOp MaterializeOperand(DomainShapeAttr shape, mlir::OperandRange domain,
                              ValueOperand &operand,
                              const InsertionPoint &insertion_point,
                              llvm::Optional<int> memory_space,
                              mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  insertion_point.Set(builder);

  // Build the copy operation.
  mlir::Type type =
      builder.getType<ValueType>(shape, operand.GetType().ElementType());
  auto access_patterns = builder.getArrayAttr(operand.AccessPattern());
  mlir::Location loc = operand.getOwner()->getLoc();
  mlir::ArrayAttr memory_space_attr;
  if (memory_space.hasValue()) {
    memory_space_attr = builder.getArrayAttr(
        {builder.getI64IntegerAttr(memory_space.getValue())});
  }
  SairCopyOp copy_op = builder.create<SairCopyOp>(
      loc, type, domain, access_patterns, operand.value(),
      /*loop_nest=*/insertion_point.loop_nest,
      /*memory_space=*/memory_space_attr);
  // Point the operand to the result of the copy operation.
  operand.set_value(copy_op.result());
  operand.SetAccessPattern(
      AccessPatternAttr::GetIdentity(builder.getContext(), domain.size()));
  return copy_op;
}

// MLIR pass that inserts copies before sair.to_memref and sair.map_reduce
// operations in order to ensure that they can operate in place. No copy is
// inserted if the operation can already execute in place.
class InsertCopies : public InsertCopiesPassBase<InsertCopies> {
  // Inserts sair.copy operations in order to ensure that the access pattern of
  // sair.to_memref operations are invertible and that sair.to_memref operations
  // are not using a value produced by a sair.from_memeref operation.
  void runOnFunction() override {
    mlir::OpBuilder builder(&getContext());

    // Insert copies before sair.to_memref if the value is produced by a
    // sair.from_memref operation or if the access pattern is not invertible.
    getFunction().walk([&builder](SairToMemRefOp op) {
      mlir::Operation *defining_op = op.value().getDefiningOp();
      if (!isa<SairFromMemRefOp>(defining_op) &&
          op.AccessPattern(0).InverseAffineMap()) {
        return;
      }
      ValueOperand operand = op.ValueOperands()[0];
      SairProgramOp program_op = cast<SairProgramOp>(op.getParentOp());
      // Move the operation at the end of the program so that we can generate a
      // new loop nest without causing interference with existing fusion
      // constraints.
      // TODO(ulysse): allow choosing the insertion point with an attribute.
      op.getOperation()->moveBefore(program_op.body().front().getTerminator());
      InsertionPoint insertion_point;
      insertion_point.operation = op;
      insertion_point.direction = Direction::kBefore;
      insertion_point.loop_nest =
          GetDefaultLoopNest(program_op, op.shape().NumDimensions());
      MaterializeOperand(op.shape(), op.domain(), operand, insertion_point,
                         ValueProducerOp::kMemory, builder);
    });

    // Copy initializing operands of sair.map_reduce operations if they have
    // another use later in the Sair program or if the access pattern is not
    // injective.
    getFunction().walk([&builder](SairMapReduceOp op) {
      for (int i = 0, e = op.inits().size(); i < e; ++i) {
        ValueOperand operand = op.ValueOperands()[i];
        bool is_access_injective =
            operand.AccessPattern().IsInjective(op.parallel_domain().size());
        bool memory_space_match =
            GetMemorySpace(operand.value()) == op.GetMemorySpace(i);
        if (is_access_injective && GetLastUse(operand.value()) == op &&
            memory_space_match) {
          continue;
        }

        int first_reduce_loop = 0;
        int parallel_domain_size = op.parallel_domain().size();
        if (op.loop_nest().hasValue()) {
          auto loop_range = op.loop_nest().getValue().getAsRange<LoopAttr>();
          auto it = llvm::find_if(loop_range, [&](LoopAttr loop) {
            return !loop.iter().Rematerialize() &&
                   loop.iter().Dimension() >= parallel_domain_size;
          });
          first_reduce_loop = std::distance(it, loop_range.end());
        }

        InsertionPoint insertion_point =
            FindInsertionPoint(op.parallel_domain().size(), first_reduce_loop,
                               cast<ComputeOp>(op.getOperation()));
        DomainShapeAttr copy_shape = op.shape().Prefix(parallel_domain_size);
        MaterializeOperand(copy_shape, op.parallel_domain(), operand,
                           insertion_point, op.GetMemorySpace(i), builder);
      }
    });
  }
};

// Stores the value 'result' produced by 'op' in 'memref'. 'store_pattern' is
// an affine map from the index space of the domain of 'op' to the index space
// of 'memref' that indicates where to store in the memref.
//
// In practice, this function adds a store instruction at the end of the body of
// 'op'. It does not remove the stored value from the results of 'op'.
void StoreResultInMemref(SairOpWithBody op, mlir::OpResult result,
                         mlir::Value memref, mlir::AffineMap store_pattern,
                         mlir::ConversionPatternRewriter &rewriter) {
  int domain_size = cast<SairOp>(op.getOperation()).domain().size();
  auto return_op = cast<SairReturnOp>(op.block().getTerminator());
  mlir::Value value_to_store = return_op.getOperand(result.getResultNumber());
  rewriter.setInsertionPoint(return_op);
  auto domain_indices = op.block().getArguments().take_front(domain_size);

  rewriter.create<mlir::AffineStoreOp>(return_op.getLoc(), value_to_store,
                                       memref, store_pattern, domain_indices);
}

// Replaces a sair.to_memref operations by a store in the operation producing
// the value argument of the ToMemRef operation.
//
// Fails if the producing operation is not a SairOperationWithRegion operation
// or if the access pattern of the sair.to_memref operation is not invertible.
class LowerToMemRefPattern : public mlir::OpConversionPattern<SairToMemRefOp> {
  using mlir::OpConversionPattern<SairToMemRefOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SairToMemRefOp op, ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    SairToMemRefOp::Adaptor adapted(operands);
    SairOpWithBody defining_op =
        dyn_cast_or_null<SairOpWithBody>(adapted.value().getDefiningOp());
    if (defining_op == nullptr) return failure();
    mlir::AffineMap store_pattern = op.AccessPattern(0).InverseAffineMap();
    if (store_pattern == mlir::AffineMap(nullptr)) return failure();
    mlir::OpResult producer_result;

    // Retrieve the OpResult corresponding to value in the producer results.
    for (const auto &result : defining_op.getOperation()->getOpResults()) {
      if (result == adapted.value()) {
        producer_result = result;
        break;
      }
    }
    assert(producer_result);

    StoreResultInMemref(defining_op, producer_result, adapted.memref(),
                        store_pattern, rewriter);
    rewriter.eraseOp(op);
    return success();
  }
};

// MLIR pass that lowers all the sair.to_memref operations in a function. It
// replaces each sair.to_memref operation by store operations in the body of the
// operation producing the stored value.
//
// Fails if the producing operation is not a SairOperationWithRegion or if the
// access pattern of the sair.to_memref operation is not invertible.
class LowerToMemRef : public LowerToMemRefPassBase<LowerToMemRef> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    patterns.insert<LowerToMemRefPattern>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<SairDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalOp<mlir::FuncOp>();
    target.addIllegalOp<SairToMemRefOp>();

    if (failed(mlir::applyFullConversion(getFunction(), target, patterns))) {
      signalPassFailure();
    }
  }
};

// Lowers a sair.from_memref operation into a sair.from_scalar operation that
// returns a 0-dimensional sair value encapsulating the whole memref.
class MaterializeFromMemRef
    : public mlir::OpConversionPattern<SairFromMemRefOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      SairFromMemRefOp op, llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    if (op.domain().empty()) return mlir::failure();
    SairFromMemRefOp::Adaptor adapted(operands);
    rewriter.replaceOpWithNewOp<SairFromScalarOp>(op, adapted.memref());
    return mlir::success();
  }
};

// Updates the use of a !sair.value to load from a memref wrapped in a
// !sair.value operation instead. Use 'memref_layout' to map elements of 'use'
// to elements of the 'memref'.
void UpdateUseAfterMaterialization(mlir::Value memref_value,
                                   mlir::AffineMap memref_layout,
                                   mlir::Type memref_type, OpOperand &use,
                                   mlir::OpBuilder &builder) {
  auto sair_op = llvm::cast<SairOp>(use.getOwner());
  mlir::Block *body = &llvm::cast<SairOpWithBody>(use.getOwner()).block();

  const int operand_number = use.getOperandNumber();
  const int domain_size = sair_op.domain().size();
  const int input_position = operand_number - domain_size;
  AccessPatternAttr old_access_pattern = sair_op.AccessPattern(input_position);
  // Set the new value and access pattern.
  auto new_access_pattern =
      AccessPatternAttr::get(builder.getContext(), domain_size, {});
  sair_op.SetAccessPattern(input_position, new_access_pattern);
  use.set(memref_value);
  // Update the body of the map.
  mlir::BlockArgument scalar_argument = body->getArgument(operand_number);
  mlir::BlockArgument memref_argument =
      body->insertArgument(operand_number, memref_type);
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::AffineMap load_map =
      memref_layout.compose(old_access_pattern.AsAffineMap());
  auto load_indices = body->getArguments().take_front(domain_size);
  auto load_op = builder.create<mlir::AffineLoadOp>(
      sair_op.getLoc(), memref_argument, load_map, load_indices);
  scalar_argument.replaceAllUsesWith(load_op.getResult());
  body->eraseArgument(scalar_argument.getArgNumber());
}

// Replaces 'op' by a new sair.map_reduce operation that reads 'init_operand'
// from the memref wrapped in a Sair value 'memref_value'. 'init_operand' must
// be an operand initializing one of the reduction variables of 'op'.
// The reduction variable is removed from the new operation and replaced by
// loads and stores from and to the memref. Use 'memref_layout' to map elements
// of 'init_operand' to elements of the memref.
void UpdateUseInPlaceAfterMaterialization(
    SairMapReduceOp op, mlir::Value memref_value, mlir::AffineMap memref_layout,
    mlir::Type memref_type, OpOperand &init_operand, mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  SairOp sair_op = llvm::cast<SairOp>(op.getOperation());
  builder.setInsertionPoint(op);

  int domain_size = sair_op.domain().size();
  int init_position =
      init_operand.getOperandNumber() - op.inits().getBeginOperandIndex();

  llvm::SmallVector<mlir::Type, 4> result_types;
  result_types.reserve(op.getNumResults() - 1);
  auto result_type_it = op.result_type_begin() + init_position;
  result_types.append(op.result_type_begin(), result_type_it);
  result_types.append(std::next(result_type_it), op.result_type_end());

  llvm::SmallVector<mlir::Value, 4> init_operands;
  init_operands.reserve(op.getNumResults() - 1);
  auto init_operand_it = op.inits().begin() + init_position;
  init_operands.append(op.inits().begin(), init_operand_it);
  init_operands.append(std::next(init_operand_it), op.inits().end());

  llvm::SmallVector<mlir::Value, 4> input_operands;
  input_operands.reserve(op.inputs().size() + 1);
  input_operands.append(op.inputs().begin(), op.inputs().end());
  input_operands.push_back(memref_value);

  llvm::SmallVector<mlir::Attribute, 8> access_pattern_array;
  access_pattern_array.reserve(op.access_pattern_array().size());
  auto access_pattern_it = op.access_pattern_array().begin() + init_position;
  access_pattern_array.append(op.access_pattern_array().begin(),
                              access_pattern_it);
  access_pattern_array.append(std::next(access_pattern_it),
                              op.access_pattern_array().end());
  access_pattern_array.push_back(
      AccessPatternAttr::get(op.getContext(), domain_size, {}));
  mlir::ArrayAttr access_pattern_attr =
      builder.getArrayAttr(access_pattern_array);

  llvm::SmallVector<mlir::Attribute, 4> memory_spaces;
  memory_spaces.reserve(op.getNumResults() - 1);
  auto memory_space_it = op.memory_space().getValue().begin() + init_position;
  memory_spaces.append(op.memory_space().getValue().begin(), memory_space_it);
  memory_spaces.append(memory_space_it + 1, op.memory_space().getValue().end());
  mlir::ArrayAttr memory_space_attr = builder.getArrayAttr(memory_spaces);

  SairMapReduceOp new_op = builder.create<SairMapReduceOp>(
      op.getLoc(), result_types, op.parallel_domain(), op.reduction_domain(),
      access_pattern_attr, init_operands, input_operands, op.shape(),
      op.loop_nestAttr(), memory_space_attr);

  // Forward attributes.

  // Setup the new body.
  new_op.body().getBlocks().splice(new_op.body().begin(),
                                   op.body().getBlocks());
  auto access_indices = new_op.block().getArguments().take_front(domain_size);
  mlir::BlockArgument scalar_argument =
      new_op.block().getArgument(init_operand.getOperandNumber());
  mlir::BlockArgument memref_argument = new_op.block().addArgument(memref_type);

  // Load from the memref.
  mlir::AffineMap access_map =
      memref_layout.compose(op.AccessPattern(init_position).AsAffineMap());
  builder.setInsertionPointToStart(&new_op.block());
  auto load_op = builder.create<mlir::AffineLoadOp>(
      op.getLoc(), memref_argument, access_map, access_indices);
  scalar_argument.replaceAllUsesWith(load_op.getResult());
  new_op.block().eraseArgument(scalar_argument.getArgNumber());

  // Store into the memref.
  mlir::Operation *sair_return = new_op.block().getTerminator();
  builder.setInsertionPoint(sair_return);
  builder.create<mlir::AffineStoreOp>(
      op.getLoc(), sair_return->getOperand(init_position), memref_argument,
      access_map, access_indices);
  sair_return->eraseOperand(init_position);
}

// Replaces uses of a Sair value by loads from a memref wrapped in a Sair value.
// - 'old_value' is the Sair value to replace,
// - 'new_value' is a 0-dimensional Sair value wrapping a memref
// - 'layout' is an affine map from the domain of the old Sair value to the
//   elements of the memref that indicates the layout of the memref.
//
// Fails if any use of 'old_value' does not implement the SairOpWithBody trait
// or if it is a sair.map_reduce operation that cannot update 'old_value' in
// place.
mlir::LogicalResult UpdateUsersAfterMaterialization(mlir::Value old_value,
                                                    mlir::Value new_value,
                                                    mlir::AffineMap layout,
                                                    mlir::OpBuilder &builder) {
  ValueType new_value_type = new_value.getType().cast<ValueType>();
  mlir::Type memref_type = new_value_type.ElementType();

  // A Sair value to replace by new_value, using the given affine map to access
  // the elements of the memref wrapped in new_value.
  struct MaterializedValue {
    mlir::Value old_value;
    mlir::AffineMap layout;
  };

  llvm::SmallVector<MaterializedValue, 4> work_list;
  llvm::SmallVector<mlir::Operation *, 4> ops_to_erase;
  work_list.push_back({old_value, layout});
  while (!work_list.empty()) {
    MaterializedValue materialized_value = work_list.pop_back_val();
    mlir::Value old_value = materialized_value.old_value;
    mlir::AffineMap layout = materialized_value.layout;
    mlir::Operation *last_use = GetLastUse(old_value);

    for (mlir::OpOperand &use :
         llvm::make_early_inc_range(old_value.getUses())) {
      if (!isa<SairOpWithBody>(use.getOwner())) {
        use.getOwner()
                ->emitError(
                    "can only materialize operands of sair.map and "
                    "sair.map_reduce operations")
                .attachNote(old_value.getLoc())
            << "while trying to materialize a value produced here";
        return mlir::failure();
      }

      auto reduce_op = llvm::dyn_cast<SairMapReduceOp>(use.getOwner());
      int operand_number = use.getOperandNumber();
      // Handle the common case.
      if (reduce_op == nullptr || !reduce_op.IsInitOperand(operand_number)) {
        UpdateUseAfterMaterialization(new_value, layout, memref_type, use,
                                      builder);
        continue;
      }

      // Handle the special case where the user updates the argument in place.
      int init_number =
          operand_number - reduce_op.inits().getBeginOperandIndex();
      AccessPatternAttr access_pattern = reduce_op.AccessPattern(init_number);
      mlir::AffineMap result_layout = layout.compose(
          access_pattern.ResizeUseDomain(reduce_op.parallel_domain().size())
              .AsAffineMap());

      // We can only perform an in-place update if the access pattern is
      // injective and if 'op' is the last use of the variable.
      if (!result_layout.isPermutation()) {
        reduce_op.emitError("layout incompatible with an in-place update")
                .attachNote(old_value.getLoc())
            << "while trying to materialize a value produced here";
        return mlir::failure();
      }
      if (last_use != reduce_op) {
        mlir::InFlightDiagnostic error =
            reduce_op.emitError()
            << "cannot update in-place a value that is still alive";
        error.attachNote(last_use->getLoc()) << "value used here";
        error.attachNote(old_value.getLoc())
            << "while trying to materialize a value produced here";
        return mlir::failure();
      }

      UpdateUseInPlaceAfterMaterialization(reduce_op, new_value, layout,
                                           memref_type, use, builder);
      // Push new work to update the result of 'op'. Remove 'op' only once
      // the update is done.
      work_list.push_back({reduce_op.getResult(init_number), result_layout});
      ops_to_erase.push_back(reduce_op);
    }
  }
  for (mlir::Operation *op : ops_to_erase) {
    op->erase();
  }

  return mlir::success();
}

// Converts a Sair iteration dimension into a memref dimension. Appends the size
// of `dimension` to `memref_shape` if it is statically known. Otherwise,
// appends mlir::MemRefType::kDynamicSize to `memref_shape` and appends a Sair
// value containing the size to `alloc_operands` as well as the access pattern
// for accessing the size to 'alloc_access_pattern'.
void GetMemRefDimension(
    mlir::Value dimension, llvm::SmallVectorImpl<int64_t> &memref_shape,
    llvm::SmallVectorImpl<mlir::Value> &alloc_operands,
    llvm::SmallVectorImpl<mlir::Attribute> &alloc_access_patterns) {
  mlir::Operation *defining_op = dimension.getDefiningOp();
  // Sair ensures dimensions are defined in the region they are used.
  assert(defining_op);

  if (auto static_range = llvm::dyn_cast<SairStaticRangeOp>(defining_op)) {
    assert(static_range.size().getBitWidth() <= 64);
    memref_shape.push_back(static_range.size().getLimitedValue());
    return;
  }

  auto range = llvm::cast<SairRangeOp>(defining_op);
  assert(range.domain().empty());
  int dynamic_size = mlir::MemRefType::kDynamicSize;
  memref_shape.push_back(dynamic_size);
  alloc_operands.push_back(range.size());
  alloc_access_patterns.push_back(
      AccessPatternAttr::get(defining_op->getContext(), 0, {}));
}

// Returns the last ComputeOp using 'value' or one of the variables aliasing
// with 'value'. Multiple Sair values can alias because sair.map_reduce updates
// its initialization operands in place when they are lowered into memrefs.
ComputeOp GetLastStorageUse(mlir::Value value) {
  ComputeOp last_use = nullptr;

  llvm::SmallVector<mlir::Value, 4> work_list;
  work_list.push_back(value);
  while (!work_list.empty()) {
    mlir::Value value = work_list.pop_back_val();
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *owner = use.getOwner();
      ComputeOp compute_op = dyn_cast<ComputeOp>(owner);
      if (compute_op != nullptr) {
        if (last_use == nullptr ||
            last_use.getOperation()->isBeforeInBlock(owner)) {
          last_use = compute_op;
        }
      }

      // Push the result of the operation to the work list if it performs an
      // in-place update.
      auto map_reduce = llvm::dyn_cast<SairMapReduceOp>(owner);
      if (map_reduce == nullptr) continue;
      int operand_number = use.getOperandNumber();
      if (!map_reduce.IsInitOperand(operand_number)) continue;
      mlir::Value result = map_reduce.getResult(
          operand_number - map_reduce.inits().getBeginOperandIndex());
      work_list.push_back(result);
    }
  }
  return last_use;
}

// Allocates and deallocates a memref to hold `value`. Appends a Sair value
// holding the memref to `memref_values` and appends the memref type to
// `memref_types`. `producer_domain` must contain the domain of the operation
// producing `value`.
//
// Expects 'value' to have a defining operation and to be used exclusively in
// the block where it is created.
void CreateMemRefForValue(mlir::Value value, SairMapOp producer,
                          llvm::SmallVectorImpl<mlir::Value> &memref_values,
                          llvm::SmallVectorImpl<mlir::Type> &memref_types,
                          mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  llvm::SmallVector<int64_t, 4> memref_shape;
  memref_shape.reserve(producer.domain().size());
  llvm::SmallVector<mlir::Value, 4> alloc_operands;
  llvm::SmallVector<mlir::Attribute, 4> alloc_access_patterns;
  for (mlir::Value dimension : producer.domain()) {
    GetMemRefDimension(dimension, memref_shape, alloc_operands,
                       alloc_access_patterns);
  }

  // Compute the memref type.
  mlir::Type element_type = value.getType().cast<ValueType>().ElementType();
  DomainShapeAttr alloc_shape = DomainShapeAttr::get(builder.getContext());
  mlir::MemRefType memref_type =
      mlir::MemRefType::get(memref_shape, element_type);
  mlir::Type alloc_type = builder.getType<ValueType>(alloc_shape, memref_type);

  // Create a sair.map operation that allocates the memref.
  llvm::ArrayRef<mlir::Value> alloc_domain;
  mlir::ArrayAttr alloc_access_pattern_attr =
      builder.getArrayAttr(alloc_access_patterns);
  InsertionPoint alloc_insertion_point =
      FindInsertionPoint(0, 0, cast<ComputeOp>(producer.getOperation()));
  alloc_insertion_point.Set(builder);
  mlir::ArrayAttr memory_space_attr = builder.getArrayAttr(
      {builder.getI64IntegerAttr(ValueProducerOp::kRegister)});
  SairMapOp alloc_map_op = builder.create<SairMapOp>(
      value.getLoc(), alloc_type, alloc_domain, alloc_access_pattern_attr,
      alloc_operands, alloc_shape, alloc_insertion_point.loop_nest,
      memory_space_attr);
  // The pointer to the memory is stored in registers.
  alloc_map_op.SetMemorySpace(0, ValueProducerOp::kRegister);
  builder.setInsertionPointToStart(&alloc_map_op.block());
  mlir::AllocOp alloc_op = builder.create<mlir::AllocOp>(
      value.getLoc(), memref_type, alloc_map_op.block_inputs());
  builder.create<SairReturnOp>(value.getLoc(), alloc_op.getResult());

  // Create the sair.map operation that deallocates the memref.
  ComputeOp last_use = GetLastStorageUse(value);
  if (last_use == nullptr) {
    last_use = cast<ComputeOp>(producer.getOperation());
  }
  mlir::ArrayAttr dealloc_access_pattern =
      builder.getArrayAttr(AccessPatternAttr::get(
          builder.getContext(), /*domain_size =*/0, /*pattern =*/{}));
  InsertionPoint dealloc_insertion_point =
      FindInsertionPoint(0, 0, last_use, Direction::kAfter);
  dealloc_insertion_point.Set(builder);
  SairMapOp dealloc_map_op = builder.create<SairMapOp>(
      value.getLoc(), llvm::ArrayRef<mlir::Type>(), alloc_domain,
      dealloc_access_pattern, alloc_map_op.getResult(0), alloc_shape,
      dealloc_insertion_point.loop_nest, builder.getArrayAttr({}));
  builder.setInsertionPointToStart(&dealloc_map_op.block());
  builder.create<mlir::DeallocOp>(value.getLoc(),
                                  dealloc_map_op.block_inputs().front());
  builder.create<SairReturnOp>(value.getLoc(), llvm::ArrayRef<mlir::Value>());

  // Register the memref in output arrays.
  memref_values.push_back(alloc_map_op.getResult(0));
  memref_types.push_back(memref_type);
}

// Replaces the multidimensional results produced by a sair.map operation by
// 0-dimensional Sair values wrapping a memref.
//
// Fails if the domain is not hyper-rectangular or if one of the results cannot
// be materialized.
mlir::LogicalResult IntroduceMemRef(SairMapOp op, mlir::OpBuilder &builder) {
  if (op.results().empty()) return mlir::success();
  // TODO(ulysse): handle non-hyperectangular domains.
  // TODO(ulysse): handle the case where some memory spaces are not set
  // TODO(ulysse): handle the case where some returned values are in register
  if (!op.shape().IsHyperRectangular()) {
    return op.emitError()
           << "can only materialize hyper-rectangular Sair values";
  }

  // Create the new memref.
  llvm::SmallVector<mlir::Value, 4> operands(op.inputs());
  llvm::SmallVector<mlir::Attribute, 4> access_patterns(
      op.access_pattern_array().getAsRange<mlir::Attribute>());
  operands.reserve(operands.size() + op.getNumResults());
  access_patterns.reserve(operands.size() + op.getNumResults());
  llvm::SmallVector<mlir::Type, 4> memref_types;
  memref_types.reserve(op.getNumResults());

  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    mlir::Value result = op.getResult(i);
    if (!op.IsMemorySpaceSet(i)) {
      return op.emitError() << "no memory space specified for result " << i;
    }

    if (result.use_empty() ||
        op.GetMemorySpace(i) != ValueProducerOp::kMemory) {
      continue;
    }

    CreateMemRefForValue(result, op, operands, memref_types, builder);
    auto layout = mlir::AffineMap::getMultiDimIdentityMap(op.domain().size(),
                                                          op.getContext());
    if (failed(UpdateUsersAfterMaterialization(result, operands.back(), layout,
                                               builder))) {
      return mlir::failure();
    }
    access_patterns.push_back(
        AccessPatternAttr::get(builder.getContext(), op.domain().size(), {}));
  }

  // Replace 'op' with a new operation that writes into the memrefs instead of
  // returning values.
  mlir::ArrayAttr access_pattern_attr = builder.getArrayAttr(access_patterns);
  SairMapOp new_op = builder.create<SairMapOp>(
      op.getLoc(), llvm::ArrayRef<mlir::Type>(), op.domain(),
      access_pattern_attr, operands, op.shape(), op.loop_nestAttr(),
      /*memory_space=*/builder.getArrayAttr({}));

  // Set the body of the new operation.
  new_op.body().getBlocks().clear();
  new_op.body().getBlocks().splice(new_op.body().begin(),
                                   op.body().getBlocks());
  builder.setInsertionPoint(new_op.block().getTerminator());
  OperandRange returned_values = new_op.block().getTerminator()->getOperands();
  auto memref_arguments = new_op.block().addArguments(memref_types);
  auto indices = new_op.block().getArguments().take_front(op.domain().size());
  for (auto p : llvm::zip(returned_values, memref_arguments)) {
    builder.create<mlir::StoreOp>(op.getLoc(), std::get<0>(p), std::get<1>(p),
                                  indices);
  }
  new_op.block().getTerminator()->setOperands({});
  op.erase();
  return mlir::success();
}

// Replaces the value produced by a sair.from_memref operation by a
// 0-dimensional value wrapping a memref.
mlir::LogicalResult IntroduceMemRef(SairFromMemRefOp op,
                                    mlir::OpBuilder &builder) {
  auto from_scalar = builder.create<SairFromScalarOp>(op.getLoc(), op.memref());
  auto layout = mlir::AffineMap::getMultiDimIdentityMap(op.domain().size(),
                                                        op.getContext());
  if (failed(UpdateUsersAfterMaterialization(op.result(), from_scalar.result(),
                                             layout, builder))) {
    return mlir::failure();
  }
  op.erase();
  return mlir::success();
}

// Replaces multidimensional Sair values by 0d-dimensional Sair values
// wrapping a memref. This requires multidimensional values to be produced by
// sair.from_memref, sair.map or sair.map_reduce operations and to be used by
// sair.map or sair.map_reduce operations only.
//
// Only hyper-rectangular domains are supported for now.
class MaterializeMemRefs
    : public MaterializeMemRefsPassBase<MaterializeMemRefs> {
  void runOnFunction() override {
    mlir::OpBuilder builder(&getContext());
    getFunction().walk([this, &builder](SairFromMemRefOp op) {
      builder.setInsertionPoint(op);
      if (mlir::failed(IntroduceMemRef(op, builder))) signalPassFailure();
    });

    // Materialize results of sair.map operations. We walk a copy of the list of
    // sair.map operations as the list of operations stored in the function
    // operation will be modified when materializing memrefs.
    //
    // We do not need to walk sair.map_reduce operations as they operate in
    // place: their outputs will be materialized at the same time as their
    // initialization operands.
    llvm::SmallVector<SairMapOp, 4> map_ops;
    getFunction().walk([&](SairMapOp op) { map_ops.push_back(op); });

    for (SairMapOp op : map_ops) {
      builder.setInsertionPoint(op);
      if (mlir::failed(IntroduceMemRef(op, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateInsertCopiesPass() {
  return std::make_unique<InsertCopies>();
}

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerToMemRefPass() {
  return std::make_unique<LowerToMemRef>();
}

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateMaterializeMemRefsPass() {
  return std::make_unique<MaterializeMemRefs>();
}

}  // namespace sair
