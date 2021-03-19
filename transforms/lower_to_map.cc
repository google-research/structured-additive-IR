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
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

// Lowers `op` into a sair.map operation that takes a single value as argument
// and returns it immediately.
void RewriteCopyToMap(SairCopyOp op, mlir::OpBuilder &builder) {
  mlir::OperandRange inputs = op.getOperands().slice(op.domain().size(), 1);
  SairMapOp new_op = builder.create<SairMapOp>(
      op.getLoc(), op.getType(), op.domain(), op.mapping_array(), inputs,
      op.shape(), op.loop_nestAttr(), op.storageAttr());

  // Set the block of code in the sair.map operation to return its argument
  // unchanged.
  builder.setInsertionPointToStart(&new_op.block());
  builder.create<SairReturnOp>(op.getLoc(), new_op.block_inputs().front());
  op.result().replaceAllUsesWith(new_op.getResult(0));
  op.erase();
}

// Lowers sair.alloc operation into a sair.map that calls std.alloc internally.
mlir::LogicalResult RewriteAllocToMap(SairAllocOp op,
                                      mlir::OpBuilder &builder) {
  // Cannot emit an allocation for a map with layout.
  if (!op.MemType().getAffineMaps().empty()) return mlir::failure();

  SairMapOp map_op = builder.create<SairMapOp>(
      op.getLoc(), op.getType(), op.domain(), op.mapping_array(),
      op.dynamic_sizes(), op.shape(), op.loop_nestAttr(), op.storageAttr());

  builder.setInsertionPointToStart(&map_op.block());
  mlir::Value allocated = builder.create<mlir::memref::AllocOp>(
      op.getLoc(), op.MemType(), map_op.block_inputs(),
      /*alignment=*/nullptr);
  builder.create<SairReturnOp>(op.getLoc(), allocated);
  op.result().replaceAllUsesWith(map_op.getResult(0));
  op->erase();

  return mlir::success();
}

// Lowers sair.free operation into a sair.map that calls std.dealloc internally.
void RewriteFreeToMap(SairFreeOp op, mlir::OpBuilder &builder) {
  SairMapOp map_op = builder.create<SairMapOp>(
      op.getLoc(), llvm::None, op.domain(), op.mapping_array(), op.value(),
      op.shape(), op.loop_nestAttr(), /*memory_space=*/nullptr);

  builder.setInsertionPointToStart(&map_op.block());
  builder.create<mlir::memref::DeallocOp>(op.getLoc(),
                                          map_op.block_inputs()[0]);
  builder.create<SairReturnOp>(op.getLoc(), llvm::None);
  op->erase();
}

// Given a source op, which is either a SairLoadFromMemRefOp or a
// SairStoreToMemRefOp, populates `region` with a block that computes indices to
// accesses the memref and add them to `indices`. Returns an object that defines
// arguments to pass to the sair.map operation wrapping the region.
template <typename OpTy>
MapArguments LoadStoreIndices(OpTy source_op,
                              llvm::ArrayRef<ValueAccess> domain,
                              mlir::Region &region,
                              llvm::SmallVectorImpl<mlir::Value> &indices,
                              mlir::OpBuilder &builder) {
  static_assert(
      llvm::is_one_of<OpTy, SairLoadFromMemRefOp, SairStoreToMemRefOp>::value,
      "can extract memref indices only from memref-related Sair ops");

  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  int domain_size = domain.size();

  // Allocate a block to add index computations.
  llvm::SmallVector<mlir::Type> block_arg_types(domain_size,
                                                builder.getIndexType());
  mlir::Block *block = builder.createBlock(&region, {}, block_arg_types);
  MapArguments map_arguments(block, domain_size);

  // Forward `source_op` arguments to the block.
  for (ValueOperand operand : source_op.ValueOperands()) {
    map_arguments.AddArgument(operand.Get());
  }

  // Compute memref indices.
  indices.reserve(source_op.layout().size());
  MappingAttr inverse_layout = source_op.layout().Inverse();

  llvm::SmallVector<mlir::Value> domain_indices;
  domain_indices.reserve(domain_size + 1);
  llvm::append_range(domain_indices,
                     block->getArguments().take_front(domain_size));
  domain_indices.push_back(nullptr);
  auto s0 = mlir::getAffineSymbolExpr(0, source_op.getContext());

  for (MappingExpr layout_dim : source_op.layout()) {
    AffineExpr expr = layout_dim.AsAffineExpr();
    // Make sure expr starts at 0 and has step 1.
    RangeParameters params = layout_dim.GetRangeParameters(
        source_op.getLoc(), domain, inverse_layout, builder, map_arguments);
    domain_indices.back() =
        Materialize(source_op.getLoc(), params.begin, builder);
    auto affine_map =
        AffineMap::get(domain_size, 1, (expr - s0).floorDiv(params.step));

    mlir::Value index = builder.create<mlir::AffineApplyOp>(
        source_op.getLoc(), affine_map, domain_indices);
    indices.push_back(index);
  }

  return map_arguments;
}

// Given a source op, which is either a SairLoadFromMemRefOp or a
// SairStoreToMemRefOp, creates a map operation with the same shape, return type
// and operands. Populates the body of the map with operations to compute
// indexes to access the memrefs, and add them to `indices`. Does not add a
// return op to the body of the map operation. May add operands to the map
// operations if needed to compute `indices`.
template <typename OpTy>
SairMapOp LoadStoreMapAndIndices(OpTy source_op, mlir::OpBuilder &builder,
                                 llvm::SmallVectorImpl<mlir::Value> &indices) {
  static_assert(
      llvm::is_one_of<OpTy, SairLoadFromMemRefOp, SairStoreToMemRefOp>::value,
      "can extract memref indices only from memref-related Sair ops");
  constexpr llvm::StringRef kLayoutAttrName = "layout";
  int domain_size = source_op.domain().size();

  // Express `source_op` domain as value accesses.
  llvm::SmallVector<ValueAccess> domain;
  domain.reserve(domain_size);
  for (auto [value, shape_dim] :
       llvm::zip(source_op.domain(), source_op.shape().Dimensions())) {
    domain.push_back(
        {value, shape_dim.dependency_mapping().ResizeUseDomain(domain_size)});
  }

  mlir::Region region;
  MapArguments map_arguments =
      LoadStoreIndices(source_op, domain, region, indices, builder);

  // Allocate the map operation.
  SairMapOp map_op = builder.create<SairMapOp>(
      source_op.getLoc(), source_op->getResultTypes(), source_op.domain(),
      builder.getArrayAttr(map_arguments.mappings()), map_arguments.values(),
      source_op.shape(), source_op.loop_nestAttr(), /*storage=*/nullptr);
  map_op.body().takeBody(region);
  // Rely on ForwardAttributes to forward storage attribute as only
  // LoadFromMemRef has the attribute.
  ForwardAttributes(source_op, map_op,
                    {SairDialect::kAccessMapAttrName, kLayoutAttrName});
  return map_op;
}

// Rewrites a SairFromMemRefOp to a SairMapOp that contains a load from the
// memref.
void RewriteToMap(SairLoadFromMemRefOp op, mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  llvm::SmallVector<mlir::Value, 4> indices;

  SairMapOp map_op = LoadStoreMapAndIndices(op, builder, indices);
  builder.setInsertionPointToEnd(&map_op.block());
  mlir::Value loaded = builder.create<mlir::memref::LoadOp>(
      op.getLoc(), map_op.block_inputs()[0], indices);
  builder.create<SairReturnOp>(op.getLoc(), loaded);
  op.result().replaceAllUsesWith(map_op.getResult(0));
  op->erase();
}

// Rewrites a SairToMemRefOp to a SairMap op that contains a store into the
// memref.
void RewriteToMap(SairStoreToMemRefOp op, mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  llvm::SmallVector<mlir::Value, 4> indices;
  SairMapOp map_op = LoadStoreMapAndIndices(op, builder, indices);
  builder.setInsertionPointToEnd(&map_op.block());

  builder.create<mlir::memref::StoreOp>(op.getLoc(), map_op.block_inputs()[1],
                                        map_op.block_inputs()[0], indices);
  builder.create<SairReturnOp>(op.getLoc());
  op->erase();
}

class LowerToMap : public LowerToMapPassBase<LowerToMap> {
  // Converts sair.copy operations into sair.map operations. This is a hook for
  // the MLIR pass infrastructure.
  void runOnFunction() override {
    mlir::MLIRContext *context = &getContext();
    getFunction().walk([context](Operation *op) {
      mlir::OpBuilder builder(context);
      builder.setInsertionPoint(op);
      if (auto copy = dyn_cast<SairCopyOp>(op)) {
        RewriteCopyToMap(copy, builder);
      } else if (auto alloc = dyn_cast<SairAllocOp>(op)) {
        (void)RewriteAllocToMap(alloc, builder);
      } else if (auto free = dyn_cast<SairFreeOp>(op)) {
        RewriteFreeToMap(free, builder);
      } else if (auto from_memref = dyn_cast<SairLoadFromMemRefOp>(op)) {
        RewriteToMap(from_memref, builder);
      } else if (auto to_memref = dyn_cast<SairStoreToMemRefOp>(op)) {
        RewriteToMap(to_memref, builder);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerToMapPass() {
  return std::make_unique<LowerToMap>();
}

}  // namespace sair
