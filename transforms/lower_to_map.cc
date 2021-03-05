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
  mlir::Value allocated = builder.create<mlir::AllocOp>(
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
  builder.create<mlir::DeallocOp>(op.getLoc(), map_op.block_inputs()[0]);
  builder.create<SairReturnOp>(op.getLoc(), llvm::None);
  op->erase();
}

// Given a source op, which is either a SairLoadFromMemRefOp or a
// SairStoreToMemRefOp, populates `indices` with the values that index the
// memref given the affine access map attached to the op. May emit additional
// operations using `builder`.
template <typename OpTy>
void MemRefIndices(OpTy source_op, mlir::ValueRange block_indices,
                   mlir::OpBuilder &builder,
                   llvm::SmallVectorImpl<mlir::Value> &indices) {
  static_assert(
      llvm::is_one_of<OpTy, SairLoadFromMemRefOp, SairStoreToMemRefOp>::value,
      "can extract memref indices only from memref-related Sair ops");

  mlir::ValueRange indices_range = block_indices.slice(
      source_op.parallel_domain().size(), source_op.memref_domain().size());
  mlir::AffineMap access_map = source_op.AccessMap();
  if (access_map.isIdentity()) {
    indices.assign(indices_range.begin(), indices_range.end());
    return;
  }

  indices.reserve(access_map.getNumResults());
  for (unsigned i = 0, e = access_map.getNumResults(); i < e; ++i) {
    mlir::Value applied = builder.create<mlir::AffineApplyOp>(
        source_op.getLoc(), access_map.getSubMap(i), indices_range);
    indices.push_back(applied);
  }
}

// Rewrites a SairFromMemRefOp to a SairMapOp that contains a load from the
// memref.
void RewriteToMap(SairLoadFromMemRefOp op, mlir::OpBuilder &builder) {
  SairMapOp map_op = builder.create<SairMapOp>(
      op.getLoc(), op.result().getType(), op.domain(), op.mapping_array(),
      op.memref(), op.shape(), op.loop_nestAttr(), op.storageAttr());
  ForwardAttributes(op, map_op, {SairDialect::kAccessMapAttrName});

  llvm::SmallVector<mlir::Value, 4> indices;
  builder.setInsertionPointToStart(&map_op.block());
  MemRefIndices(op, map_op.block().getArguments(), builder, indices);
  mlir::Value loaded = builder.create<mlir::LoadOp>(
      op.getLoc(), map_op.block_inputs()[0], indices);
  builder.create<SairReturnOp>(op.getLoc(), loaded);
  op.result().replaceAllUsesWith(map_op.getResult(0));
  op->erase();
}

// Rewrites a SairToMemRefOp to a SairMap op that contains a store into the
// memref.
void RewriteToMap(SairStoreToMemRefOp op, mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 2> args({op.memref(), op.value()});
  SairMapOp map_op = builder.create<SairMapOp>(
      op.getLoc(), /*results=*/llvm::None, op.domain(), op.mapping_array(),
      args, op.shape(), op.loop_nestAttr(), /*memory_space=*/nullptr);
  ForwardAttributes(op, map_op, {SairDialect::kAccessMapAttrName});

  llvm::SmallVector<mlir::Value, 4> indices;
  builder.setInsertionPointToStart(&map_op.block());
  MemRefIndices(op, map_op.block().getArguments(), builder, indices);
  builder.create<mlir::StoreOp>(op.getLoc(), map_op.block_inputs()[1],
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
