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

#include "expansion.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace sair {

//===----------------------------------------------------------------------===//
// RegisterExpansionPatterns
//===----------------------------------------------------------------------===//

namespace {

// Expansion pattern that implements a sair.map operation by its body.
class MapExpansionPattern : public TypedExpansionPattern<SairMapOp> {
 public:
  constexpr static llvm::StringRef kName = kMapExpansionPattern;

  mlir::LogicalResult Match(SairMapOp op) const override;

  llvm::SmallVector<mlir::Value> Emit(SairMapOp op, MapBodyBuilder &map_body,
                                      mlir::OpBuilder &builder) const override;
};

mlir::LogicalResult MapExpansionPattern::Match(SairMapOp op) const {
  return mlir::success();
}

llvm::SmallVector<mlir::Value> MapExpansionPattern::Emit(
    SairMapOp op, MapBodyBuilder &map_body, mlir::OpBuilder &builder) const {
  mlir::Block *new_block = builder.getInsertionBlock();
  for (int i = 0, e = op.domain().size(); i < e; ++i) {
    op.block().getArgument(i).replaceAllUsesWith(map_body.index(i));
  }
  for (int i = 0, e = op.inputs().size(); i < e; ++i) {
    op.block_inputs()[i].replaceAllUsesWith(map_body.block_input(i));
  }
  new_block->getOperations().splice(new_block->begin(),
                                    op.block().getOperations());
  llvm::SmallVector<mlir::Value> return_values =
      new_block->getTerminator()->getOperands();
  new_block->getTerminator()->erase();
  return return_values;
}

// Expansion pattern that implements a sair.copy operation by a no-op.
class CopyExpansionPattern : public TypedExpansionPattern<SairCopyOp> {
 public:
  constexpr static llvm::StringRef kName = kCopyExpansionPattern;

  mlir::LogicalResult Match(SairCopyOp op) const override;

  llvm::SmallVector<mlir::Value> Emit(SairCopyOp op, MapBodyBuilder &map_body,
                                      mlir::OpBuilder &builder) const override;
};

mlir::LogicalResult CopyExpansionPattern::Match(SairCopyOp op) const {
  return mlir::success();
}

llvm::SmallVector<mlir::Value> CopyExpansionPattern::Emit(
    SairCopyOp op, MapBodyBuilder &map_body, mlir::OpBuilder &builder) const {
  return {map_body.block_input(0)};
}

// Expansion pattern that implements a sair.alloc operation by memref.alloc
class AllocExpansionPattern : public TypedExpansionPattern<SairAllocOp> {
 public:
  constexpr static llvm::StringRef kName = kAllocExpansionPattern;

  mlir::LogicalResult Match(SairAllocOp op) const override;

  llvm::SmallVector<mlir::Value> Emit(SairAllocOp op, MapBodyBuilder &map_body,
                                      mlir::OpBuilder &builder) const override;
};

mlir::LogicalResult AllocExpansionPattern::Match(SairAllocOp op) const {
  // Cannot emit an allocation for a map with layout.
  if (!op.MemType().getAffineMaps().empty()) return mlir::failure();
  return mlir::success();
}

llvm::SmallVector<mlir::Value> AllocExpansionPattern::Emit(
    SairAllocOp op, MapBodyBuilder &map_body, mlir::OpBuilder &builder) const {
  mlir::Value result = builder.create<mlir::memref::AllocOp>(
      op.getLoc(), op.MemType(), map_body.block_inputs(),
      /*alignment=*/nullptr);
  return {result};
}

// Expansion pattern that implements a sair.free operation by memref.free
class FreeExpansionPattern : public TypedExpansionPattern<SairFreeOp> {
 public:
  constexpr static llvm::StringRef kName = kFreeExpansionPattern;

  mlir::LogicalResult Match(SairFreeOp op) const override;

  llvm::SmallVector<mlir::Value> Emit(SairFreeOp op, MapBodyBuilder &map_body,
                                      mlir::OpBuilder &builder) const override;
};

mlir::LogicalResult FreeExpansionPattern::Match(SairFreeOp op) const {
  return mlir::success();
}

llvm::SmallVector<mlir::Value> FreeExpansionPattern::Emit(
    SairFreeOp op, MapBodyBuilder &map_body, mlir::OpBuilder &builder) const {
  builder.create<mlir::memref::DeallocOp>(op.getLoc(), map_body.block_input(0));
  return {};
}

// Insert code that computes memref access indices to `map_body`. `domain` is
// the domain of the memory access operation and `layout` a mapping from domain
// to memref dimensions.
llvm::SmallVector<mlir::Value> LoadStoreIndices(
    mlir::Location loc, llvm::ArrayRef<ValueAccess> domain, MappingAttr layout,
    MapBodyBuilder &map_body, mlir::OpBuilder &builder) {
  // The mapping from the old operation domain to the new operation domain is
  // the identity. As `GetRangeParameters` expects to find the inverse of
  // `layout` in the mapping, we express the identity as the composition of
  // layout with its inverse.
  auto inverse_layout = layout.Inverse().MakeSurjective();
  auto identity = inverse_layout.Inverse().Compose(inverse_layout);
  llvm::SmallVector<RangeParameters> range_parameters =
      GetRangeParameters(loc, layout, domain, identity, map_body, builder);

  // Allocate arguments for the affine apply operations that will compute memref
  // indices. Arguments are composed of domain indices followed by the start
  // index of the current dimension.
  llvm::SmallVector<mlir::Value> apply_args;
  apply_args.reserve(domain.size() + 1);
  llvm::append_range(apply_args, map_body.indices());
  apply_args.push_back(nullptr);

  // Compute memref indices from domain indices. Normalize domain indices so
  // that they start at 0 with step 1.
  llvm::SmallVector<mlir::Value> indices;
  indices.reserve(layout.size());
  auto s0 = mlir::getAffineSymbolExpr(0, builder.getContext());
  for (const auto &[params, layout_dim] : llvm::zip(range_parameters, layout)) {
    AffineExpr expr = (layout_dim.AsAffineExpr() - s0).floorDiv(params.step);
    auto map = mlir::AffineMap::get(domain.size(), 1, expr);
    apply_args.back() = Materialize(loc, params.begin, builder);
    auto index = builder.create<AffineApplyOp>(loc, map, apply_args);
    indices.push_back(index);
  }

  return indices;
}

// Expansion pattern that implements a sair.load_from_memref operation by
// memref.load
class LoadExpansionPattern
    : public TypedExpansionPattern<SairLoadFromMemRefOp> {
 public:
  constexpr static llvm::StringRef kName = kLoadExpansionPattern;

  mlir::LogicalResult Match(SairLoadFromMemRefOp op) const override;

  llvm::SmallVector<mlir::Value> Emit(SairLoadFromMemRefOp op,
                                      MapBodyBuilder &map_body,
                                      mlir::OpBuilder &builder) const override;
};

mlir::LogicalResult LoadExpansionPattern::Match(SairLoadFromMemRefOp op) const {
  return mlir::success();
}

llvm::SmallVector<mlir::Value> LoadExpansionPattern::Emit(
    SairLoadFromMemRefOp op, MapBodyBuilder &map_body,
    mlir::OpBuilder &builder) const {
  llvm::SmallVector<mlir::Value> indices = LoadStoreIndices(
      op.getLoc(), op.DomainWithDependencies(), op.layout(), map_body, builder);
  auto load = builder.create<mlir::memref::LoadOp>(
      op.getLoc(), map_body.block_input(0), indices);
  return {load};
}

// Expansion pattern that implements a sair.load_from_memref operation by
// memref.load
class StoreExpansionPattern
    : public TypedExpansionPattern<SairStoreToMemRefOp> {
 public:
  constexpr static llvm::StringRef kName = kStoreExpansionPattern;

  mlir::LogicalResult Match(SairStoreToMemRefOp op) const override;

  llvm::SmallVector<mlir::Value> Emit(SairStoreToMemRefOp op,
                                      MapBodyBuilder &map_body,
                                      mlir::OpBuilder &builder) const override;
};

mlir::LogicalResult StoreExpansionPattern::Match(SairStoreToMemRefOp op) const {
  return mlir::success();
}

llvm::SmallVector<mlir::Value> StoreExpansionPattern::Emit(
    SairStoreToMemRefOp op, MapBodyBuilder &map_body,
    mlir::OpBuilder &builder) const {
  llvm::SmallVector<mlir::Value> indices = LoadStoreIndices(
      op.getLoc(), op.DomainWithDependencies(), op.layout(), map_body, builder);
  builder.create<mlir::memref::StoreOp>(op.getLoc(), map_body.block_input(1),
                                        map_body.block_input(0), indices);
  return {};
}

// Registers expansion pattern of type I in `map`.
template <typename... Ts>
void RegisterExpansionPattern(
    llvm::StringMap<std::unique_ptr<ExpansionPattern>> &map) {
  (void)std::initializer_list<int>{
      0, (map.try_emplace(Ts::kName, new Ts()), 0)...};
}

}  // namespace

void RegisterExpansionPatterns(
    llvm::StringMap<std::unique_ptr<ExpansionPattern>> &map) {
  RegisterExpansionPattern<MapExpansionPattern, CopyExpansionPattern,
                           AllocExpansionPattern, FreeExpansionPattern,
                           LoadExpansionPattern, StoreExpansionPattern>(map);
}

}  // namespace sair
