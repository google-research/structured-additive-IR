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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "storage.h"
#include "transforms/default_lowering_attributes.h"
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

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
  assert(num_dimensions >= fusion_level);
  if (!point.loop_nest().hasValue()) return {point, direction, nullptr};
  InsertionPoint result =
      FindInsertionPoint(cast<SairOp>(point.getOperation()),
                         point.LoopNestLoops(), fusion_level, direction);
  SairProgramOp program_op = cast<SairProgramOp>(point->getParentOp());
  result.loop_nest = GetDefaultLoopNest(program_op, num_dimensions,
                                        result.loop_nest.getValue());
  return result;
}

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

// Materializes `operand` in `domain` by inserting a copy operation. Returns the
// copy operation inserted.
SairCopyOp MaterializeOperand(DomainShapeAttr shape, mlir::OperandRange domain,
                              ValueOperand &operand,
                              const InsertionPoint &insertion_point,
                              BufferAttr buffer, mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  insertion_point.Set(builder);

  auto storage_attr =
      buffer == nullptr ? nullptr : builder.getArrayAttr({buffer});

  // Build the copy operation.
  mlir::Type type = ValueType::get(shape, operand.GetType().ElementType());
  auto mappings = builder.getArrayAttr(
      operand.Mapping().ResizeUseDomain(shape.NumDimensions()));
  mlir::Location loc = operand.getOwner()->getLoc();
  SairCopyOp copy_op =
      builder.create<SairCopyOp>(loc, type, domain, mappings, operand.value(),
                                 /*loop_nest=*/insertion_point.loop_nest,
                                 /*storage=*/storage_attr);
  // Point the operand to the result of the copy operation.
  operand.set_value(copy_op.result());
  int use_domain_size =
      cast<SairOp>(operand.getOwner()).shape().NumDimensions();
  operand.SetMapping(MappingAttr::GetIdentity(builder.getContext(),
                                              domain.size(), use_domain_size));
  return copy_op;
}

// MLIR pass that inserts copies before sair.to_memref and sair.map_reduce
// operations in order to ensure that they can operate in place. No copy is
// inserted if the operation can already execute in place.
class InsertCopies : public InsertCopiesPassBase<InsertCopies> {
  // Inserts sair.copy operations in order to ensure that the mapping of
  // sair.to_memref operations are invertible and that sair.to_memref operations
  // are not using a value produced by a sair.from_memeref operation.
  void runOnFunction() override {
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);

    auto *sair_dialect = context->getLoadedDialect<SairDialect>();

    // Insert copies before sair.to_memref if the value is produced by a
    // sair.from_memref operation or if the mapping is not invertible.
    getFunction().walk([&](SairToMemRefOp op) {
      mlir::Operation *defining_op = op.value().getDefiningOp();
      auto &storage_analysis =
          getChildAnalysis<StorageAnalysis>(op->getParentOp());

      if (!isa<SairFromMemRefOp>(defining_op) &&
          op.Value().Mapping().Inverse().IsFullySpecified()) {
        return;
      }
      ValueOperand operand = op.Value();
      SairProgramOp program_op = cast<SairProgramOp>(op->getParentOp());
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
      llvm::SmallVector<mlir::StringAttr, 4> loop_names;
      loop_names.reserve(insertion_point.loop_nest.size());
      for (auto loop : insertion_point.loop_nest.getValue()) {
        loop_names.push_back(loop.cast<LoopAttr>().name());
      }
      auto buffer = BufferAttr::get(
          /*space=*/sair_dialect->memory_attr(),
          /*name=*/storage_analysis.GetFreshBufferName(),
          /*layout=*/NamedMappingAttr::GetIdentity(context, loop_names),
          context);
      MaterializeOperand(op.shape(), op.domain(), operand, insertion_point,
                         buffer, builder);
    });
  }
};

// Uses `builder` to create a series of `affine.apply` operations that apply
// individual expressions from the given `map` to `operands`, and populates the
// `result` vector with the results of application.
void EmitAffineApplyMap(mlir::Location loc, mlir::AffineMap map,
                        mlir::ValueRange operands, mlir::OpBuilder &builder,
                        llvm::SmallVectorImpl<mlir::Value> &results) {
  assert(map.getNumInputs() <= operands.size() &&
         "expected at least as many operands as map has inputs");
  operands = operands.take_front(map.getNumInputs());
  unsigned num_results = map.getNumResults();
  results.reserve(results.size() + num_results);
  for (unsigned i = 0; i < num_results; ++i) {
    results.push_back(
        builder.create<mlir::AffineApplyOp>(loc, map.getSubMap(i), operands));
  }
}

// Uses `builder` to emit an equivalent of an AffineStoreOp that does the
// application separately and uses standard StoreOp to circumvent Affine dialect
// restrictions on value provenance.
mlir::StoreOp EmitPseudoAffineStore(mlir::Location loc,
                                    mlir::Value value_to_store,
                                    mlir::Value memref, mlir::AffineMap map,
                                    mlir::ValueRange indices,
                                    mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 6> applied;
  EmitAffineApplyMap(loc, map, indices, builder, applied);
  return builder.create<mlir::StoreOp>(loc, value_to_store, memref, applied);
}

// Uses `builder` to emit an equivalent of an AffineLoadOp that performs the
// index transformation separately and uses standard Load Op to circumvent
// Affine dialect value provenance restrictions.
mlir::LoadOp EmitPseudoAffineLoad(mlir::Location loc, mlir::Value memref,
                                  mlir::AffineMap map, mlir::ValueRange indices,
                                  mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 6> applied;
  EmitAffineApplyMap(loc, map, indices, builder, applied);
  return builder.create<mlir::LoadOp>(loc, memref, applied);
}

// Information about a `to_memref` op about to be eliminated.
struct ToMemRefOpInfo {
  ToMemRefOpInfo(SairToMemRefOp op, mlir::AffineMap map)
      : op(op), inverted_access_map(map) {}

  // The op to be eliminated.
  SairToMemRefOp op;

  // Inverted access map or null if the map is not invertible.
  mlir::AffineMap inverted_access_map;
};

// Wrapper for `map_range` to extract the operation from info structure.
SairToMemRefOp ExtractOpFromInfo(const ToMemRefOpInfo &info) { return info.op; }

// Rewrites the given `op` that has results written into memrefs using
// `to_memref` so that it accepts the memref as additional arguments (to
// preserve the isolated-from-above property) and stores data those memrefs
// instead. The `to_memref` operations are then removed. `recreate` is a
// function that constructs a concrete op.
mlir::LogicalResult internMemRefs(
    SairMapOp op, llvm::ArrayRef<mlir::Value> operands,
    llvm::function_ref<SairMapOp(mlir::ValueRange, mlir::ValueRange, SairMapOp,
                                 mlir::OpBuilder &)>
        recreate,
    mlir::ConversionPatternRewriter &rewriter) {
  // Collect the `to_memref` ops that should be eliminated. Bail out early if
  // there are none.
  llvm::SmallVector<ToMemRefOpInfo, 8> to_memref_ops;
  for (mlir::OpResult result : op.getOperation()->getResults()) {
    for (mlir::Operation *user : result.getUsers()) {
      if (auto to_memref = dyn_cast<SairToMemRefOp>(user)) {
        if (!to_memref.parallel_domain().empty() ||
            to_memref.access_map().hasValue()) {
          return rewriter.notifyMatchFailure(
              user, "operation not supported by memref materialization");
        }
        mlir::AffineMap inverse_access =
            to_memref.Value().Mapping().Inverse().AsAffineMap();
        if (!inverse_access) {
          return rewriter.notifyMatchFailure(user, "non-invertible access map");
        }
        to_memref_ops.emplace_back(to_memref, inverse_access);
      }
    }
  }
  if (to_memref_ops.empty()) return mlir::failure();

  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Wrap memrefs into Sair values and pass them as additional operands to
  // sair.map so that it remains isolated from above when we write into those
  // memrefs inside the map body.
  unsigned num_memrefs = to_memref_ops.size();
  llvm::SmallVector<mlir::Value, 8> wrapped_memrefs;
  wrapped_memrefs.reserve(num_memrefs);
  for (SairToMemRefOp to_memref :
       llvm::map_range(to_memref_ops, ExtractOpFromInfo)) {
    wrapped_memrefs.push_back(to_memref.memref());
  }

  SairMapOp new_map = recreate(operands, wrapped_memrefs, op, rewriter);
  rewriter.replaceOp(op, new_map.getOperation()->getResults());

  // Recreate the body with new block arguments.
  mlir::BlockAndValueMapping mapping;
  mapping.map(op.block().getArguments(),
              new_map.block().getArguments().drop_back(num_memrefs));
  rewriter.setInsertionPointToStart(&new_map.block());
  for (mlir::Operation &nested : op.block()) {
    rewriter.clone(nested, mapping);
  }

  // Store the computed value, as given to the terminator, in the corresponding
  // memref.
  mlir::Operation *terminator = new_map.block().getTerminator();
  rewriter.setInsertionPoint(terminator);
  auto sair_op = cast<SairOp>(op.getOperation());
  unsigned domain_size = sair_op.domain().size();
  auto domain_indices = new_map.block().getArguments().take_front(domain_size);
  for (auto en : llvm::enumerate(to_memref_ops)) {
    ToMemRefOpInfo &info = en.value();
    unsigned result_pos = info.op.value().cast<OpResult>().getResultNumber();
    mlir::Value terminatorOperand = terminator->getOperand(result_pos);
    mlir::Value memref =
        new_map.block().getArguments().take_back(num_memrefs)[en.index()];
    EmitPseudoAffineStore(info.op.getLoc(), terminatorOperand, memref,
                          info.inverted_access_map, domain_indices, rewriter);
  }

  // Erase the to_memref ops that are no longer necessary.
  for (SairToMemRefOp to_memref :
       llvm::map_range(to_memref_ops, ExtractOpFromInfo)) {
    rewriter.eraseOp(to_memref);
  }

  return mlir::success();
}

// Appends `num` mappings accessing scalar (0D) values with the given
// `use_domain_size` to the list of mappings.
mlir::ArrayAttr Append0DAccesses(mlir::ArrayAttr original, size_t num,
                                 size_t use_domain_size,
                                 mlir::MLIRContext *ctx) {
  auto mapping_list = llvm::to_vector<8>(original.getValue());
  mapping_list.append(num, MappingAttr::get(ctx, use_domain_size, {}));
  return ArrayAttr::get(ctx, mapping_list);
}

// Rewrites sair.map ops producing values written to memrefs to store individual
// values instead.
class InternMemRefsIntoMap : public mlir::OpConversionPattern<SairMapOp> {
 private:
  // Creates a new sair.map operation using updated `operands` and attaching
  // `memrefs` as additional inputs.
  static SairMapOp recreate(mlir::ValueRange operands, mlir::ValueRange memrefs,
                            SairMapOp original, mlir::OpBuilder &builder) {
    mlir::MLIRContext *ctx = builder.getContext();
    auto op = cast<SairMapOp>(original.getOperation());
    SairMapOpAdaptor adaptor(operands, op.getOperation()->getAttrDictionary());
    auto inputs = llvm::to_vector<8>(adaptor.inputs());
    llvm::append_range(inputs, memrefs);
    mlir::ArrayAttr mappings = Append0DAccesses(
        adaptor.mapping_array(), memrefs.size(), op.domain().size(), ctx);
    return builder.create<SairMapOp>(
        original.getLoc(), op.getResultTypes(), adaptor.domain(), mappings,
        inputs, op.shape(), op.loop_nestAttr(), op.storageAttr());
  }

 public:
  using mlir::OpConversionPattern<SairMapOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      SairMapOp op, llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const override {
    return internMemRefs(op, operands, recreate, rewriter);
  }
};

// MLIR pass that lowers all the sair.to_memref operations in a function. It
// replaces each sair.to_memref operation by store operations in the body of the
// operation producing the stored value.
//
// Fails if the producing operation is not a SairOperationWithRegion or if the
// mapping of the sair.to_memref operation is not invertible.
class LowerToMemRef : public LowerToMemRefPassBase<LowerToMemRef> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    patterns.insert<InternMemRefsIntoMap>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<SairDialect>();

    target.addDynamicallyLegalOp<SairMapOp>([](mlir::Operation *op) {
      for (mlir::Value result : op->getResults()) {
        for (mlir::Operation *user : result.getUsers()) {
          if (isa<SairToMemRefOp>(user)) return false;
        }
      }
      return true;
    });

    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalOp<mlir::FuncOp>();
    target.addIllegalOp<SairToMemRefOp>();

    if (failed(mlir::applyFullConversion(getFunction(), target,
                                         std::move(patterns)))) {
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
    SairFromMemRefOp::Adaptor adapted(operands,
                                      op.getOperation()->getAttrDictionary());
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
  mlir::Block *body = &llvm::cast<SairMapOp>(use.getOwner()).block();

  const int operand_number = use.getOperandNumber();
  const int domain_size = sair_op.domain().size();
  const int input_position = operand_number - domain_size;
  MappingAttr old_mapping = sair_op.ValueOperands()[input_position].Mapping();
  // Set the new value and mapping.
  auto new_mapping = MappingAttr::get(builder.getContext(), domain_size, {});
  sair_op.SetMapping(input_position, new_mapping);
  use.set(memref_value);
  // Update the body of the map.
  mlir::BlockArgument scalar_argument = body->getArgument(operand_number);
  mlir::BlockArgument memref_argument =
      body->insertArgument(operand_number, memref_type);
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  builder.setInsertionPointToStart(body);
  mlir::AffineMap load_map = memref_layout.compose(old_mapping.AsAffineMap());
  auto load_indices = body->getArguments().take_front(domain_size);
  auto load_op = EmitPseudoAffineLoad(sair_op.getLoc(), memref_argument,
                                      load_map, load_indices, builder);
  scalar_argument.replaceAllUsesWith(load_op.getResult());
  body->eraseArgument(scalar_argument.getArgNumber());
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

    for (mlir::OpOperand &use :
         llvm::make_early_inc_range(old_value.getUses())) {
      if (!isa<SairMapOp>(use.getOwner())) {
        use.getOwner()
                ->emitError(
                    "can only materialize operands of sair.map operations")
                .attachNote(old_value.getLoc())
            << "while trying to materialize a value produced here";
        return mlir::failure();
      }

      UpdateUseAfterMaterialization(new_value, layout, memref_type, use,
                                    builder);
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
// value containing the size to `alloc_operands` as well as the mapping
// for accessing the size to 'alloc_mapping'.
void GetMemRefDimension(
    mlir::Value dimension, llvm::SmallVectorImpl<int64_t> &memref_shape,
    llvm::SmallVectorImpl<mlir::Value> &alloc_operands,
    llvm::SmallVectorImpl<mlir::Attribute> &alloc_mappings) {
  mlir::Operation *defining_op = dimension.getDefiningOp();
  // Sair ensures dimensions are defined in the region they are used.
  assert(defining_op);

  if (auto static_range = llvm::dyn_cast<SairStaticRangeOp>(defining_op)) {
    assert(static_range.size().getBitWidth() <= 64);
    memref_shape.push_back(static_range.size().getLimitedValue());
    return;
  }

  auto range = llvm::cast<SairDynRangeOp>(defining_op);
  assert(range.domain().empty());
  int dynamic_size = mlir::MemRefType::kDynamicSize;
  memref_shape.push_back(dynamic_size);
  alloc_operands.push_back(range.upper_bound());
  alloc_mappings.push_back(MappingAttr::get(defining_op->getContext(), 0, {}));
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
  mlir::MLIRContext *context = builder.getContext();
  auto *sair_dialect = context->getLoadedDialect<SairDialect>();

  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  llvm::SmallVector<int64_t, 4> memref_shape;
  memref_shape.reserve(producer.domain().size());
  llvm::SmallVector<mlir::Value, 4> alloc_operands;
  llvm::SmallVector<mlir::Attribute, 4> alloc_mappings;
  for (mlir::Value dimension : producer.domain()) {
    GetMemRefDimension(dimension, memref_shape, alloc_operands, alloc_mappings);
  }

  // Compute the memref type.
  mlir::Type element_type = value.getType().cast<ValueType>().ElementType();
  DomainShapeAttr alloc_shape = DomainShapeAttr::get(builder.getContext());
  mlir::MemRefType memref_type =
      mlir::MemRefType::get(memref_shape, element_type);
  mlir::Type alloc_type = ValueType::get(alloc_shape, memref_type);

  // Create a sair.map operation that allocates the memref.
  llvm::ArrayRef<mlir::Value> alloc_domain;
  mlir::ArrayAttr alloc_mapping_attr = builder.getArrayAttr(alloc_mappings);
  InsertionPoint alloc_insertion_point =
      FindInsertionPoint(0, 0, cast<ComputeOp>(producer.getOperation()));
  alloc_insertion_point.Set(builder);
  auto buffer = BufferAttr::get(
      sair_dialect->register_attr(), /*name=*/nullptr,
      NamedMappingAttr::get({}, MappingAttr::GetIdentity(context, 0)), context);
  SairMapOp alloc_map_op = builder.create<SairMapOp>(
      value.getLoc(), alloc_type, alloc_domain, alloc_mapping_attr,
      alloc_operands, alloc_shape, alloc_insertion_point.loop_nest,
      builder.getArrayAttr({buffer}));
  builder.setInsertionPointToStart(&alloc_map_op.block());
  mlir::AllocOp alloc_op = builder.create<mlir::AllocOp>(
      value.getLoc(), memref_type, alloc_map_op.block_inputs());
  builder.create<SairReturnOp>(value.getLoc(), alloc_op.getResult());

  // Create the sair.map operation that deallocates the memref.
  ComputeOp last_use = GetLastStorageUse(value);
  if (last_use == nullptr) {
    last_use = cast<ComputeOp>(producer.getOperation());
  }
  mlir::ArrayAttr dealloc_mapping = builder.getArrayAttr(MappingAttr::get(
      builder.getContext(), /*domain_size =*/0, /*mapping =*/{}));
  InsertionPoint dealloc_insertion_point =
      FindInsertionPoint(0, 0, last_use, Direction::kAfter);
  dealloc_insertion_point.Set(builder);
  SairMapOp dealloc_map_op = builder.create<SairMapOp>(
      value.getLoc(), llvm::ArrayRef<mlir::Type>(), alloc_domain,
      dealloc_mapping, alloc_map_op.getResult(0), alloc_shape,
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
  auto *sair_dialect = builder.getContext()->getLoadedDialect<SairDialect>();

  if (op.results().empty()) return mlir::success();
  // TODO(ulysse): handle non-hyperectangular domains.
  // TODO(ulysse): handle the case where some memory spaces are not set
  // TODO(ulysse): handle the case where some returned values are in register
  if (!op.shape().IsHyperRectangular()) {
    return op.emitError()
           << "can only materialize hyper-rectangular Sair values";
  }
  if (llvm::any_of(op.domain(), [](mlir::Value v) {
    ValueOrConstant bound = cast<RangeOp>(v.getDefiningOp()).LowerBound();
    return bound.is_value() ||
           bound.constant().cast<mlir::IntegerAttr>().getInt() != 0;
  })) {
    return op.emitError() << "only 0-based ranges are supported for memrefs";
  }

  // Create the new memref.
  llvm::SmallVector<mlir::Value, 4> operands(op.inputs());
  llvm::SmallVector<mlir::Attribute, 4> mappings(
      op.mapping_array().getAsRange<mlir::Attribute>());
  operands.reserve(operands.size() + op.getNumResults());
  mappings.reserve(operands.size() + op.getNumResults());
  llvm::SmallVector<mlir::Type, 4> memref_types;
  memref_types.reserve(op.getNumResults());

  for (int i = 0, e = op.getNumResults(); i < e; ++i) {
    mlir::Value result = op.getResult(i);
    BufferAttr buffer = op.Storage(i);
    if (buffer == nullptr) {
      return op.emitError() << "no memory space specified for result " << i;
    }

    if (result.use_empty() ||
        op.Storage(i).space() == sair_dialect->register_attr()) {
      continue;
    }

    CreateMemRefForValue(result, op, operands, memref_types, builder);
    auto layout = mlir::AffineMap::getMultiDimIdentityMap(op.domain().size(),
                                                          op.getContext());
    if (failed(UpdateUsersAfterMaterialization(result, operands.back(), layout,
                                               builder))) {
      return mlir::failure();
    }
    mappings.push_back(
        MappingAttr::get(builder.getContext(), op.domain().size(), {}));
  }

  // Replace 'op' with a new operation that writes into the memrefs instead of
  // returning values.
  mlir::ArrayAttr mapping_attr = builder.getArrayAttr(mappings);
  SairMapOp new_op = builder.create<SairMapOp>(
      op.getLoc(), llvm::ArrayRef<mlir::Type>(), op.domain(), mapping_attr,
      operands, op.shape(), op.loop_nestAttr(),
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
  if (!op.parallel_domain().empty() || op.access_map().hasValue()) {
    return op.emitError()
           << "operation not supported by memref materialization";
  }
  auto layout = mlir::AffineMap::getMultiDimIdentityMap(op.domain().size(),
                                                        op.getContext());
  if (failed(UpdateUsersAfterMaterialization(op.result(), op.memref(), layout,
                                             builder))) {
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
// TODO(b/174127325): take the storage attribute into account.
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
