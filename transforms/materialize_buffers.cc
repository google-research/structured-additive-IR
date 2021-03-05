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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "loop_nest.h"
#include "storage.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Find insertion points for alloc and free operations.
std::pair<InsertionPoint, InsertionPoint> FindInsertionPoints(
    const Buffer &buffer, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();

  // Find the first and last access to the buffer.
  ComputeOp first_access = buffer.writes().front().first;
  ComputeOp last_access = buffer.writes().front().first;
  for (auto access : buffer.writes()) {
    ComputeOp op = access.first;
    if (op->isBeforeInBlock(first_access)) {
      first_access = op;
    }
    if (last_access->isBeforeInBlock(op)) {
      last_access = op;
    }
  }
  for (auto access : buffer.reads()) {
    ComputeOp op = access.first;
    if (last_access->isBeforeInBlock(op)) {
      last_access = op;
    }
  }

  int num_loops = buffer.loop_nest().size();
  InsertionPoint alloc_point = FindInsertionPoint(
      cast<SairOp>(*first_access), first_access.LoopNestLoops(), num_loops,
      Direction::kBefore);
  InsertionPoint free_point = FindInsertionPoint(cast<SairOp>(*last_access),
                                                 last_access.LoopNestLoops(),
                                                 num_loops, Direction::kAfter);

  // Define a loop-nest in the domain of loops.
  llvm::SmallVector<mlir::Attribute> loops;
  loops.reserve(buffer.loop_nest().size());
  for (int i = 0, e = buffer.loop_nest().size(); i < e; ++i) {
    auto dim_expr = MappingDimExpr::get(i, context);
    loops.push_back(LoopAttr::get(buffer.loop_nest()[i], dim_expr, context));
  }
  auto loop_nest = builder.getArrayAttr(loops);
  alloc_point.loop_nest = loop_nest;
  free_point.loop_nest = loop_nest;

  return std::make_pair(alloc_point, free_point);
}

// Returns the shape of the memref implementing `buffer` and the list of values
// providing dynamic dimension sizes.
std::pair<mlir::SmallVector<int64_t>, ValueRange> GetMemRefShape(
    const Buffer &buffer, DomainShapeAttr shape,
    llvm::ArrayRef<mlir::Value> domain, mlir::ArrayAttr loop_nest,
    mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::OpBuilder::InsertPoint map_point = builder.saveInsertionPoint();

  // Create a block that will hold computations for memref sizes.
  mlir::Region region;
  llvm::SmallVector<mlir::Type> block_arg_types(domain.size(),
                                                builder.getIndexType());
  mlir::Block *block = builder.createBlock(&region, {}, block_arg_types);
  MapArguments map_arguments(block, domain.size());

  llvm::SmallVector<int64_t> memref_shape;
  llvm::SmallVector<mlir::Value> scalar_sizes;
  auto inverse_layout = buffer.PrefixedLayout().Inverse();
  for (MappingExpr layout_dim : buffer.layout()) {
    RangeParameters params =
        layout_dim.GetRangeParameters(buffer.getLoc(), buffer.domain(),
                                      inverse_layout, builder, map_arguments);
    int step = params.step;
    if (params.begin.is<mlir::Attribute>() &&
        params.end.is<mlir::Attribute>()) {
      // Handle constant dimension.
      int beg = params.begin.get<mlir::Attribute>()
                    .cast<mlir::IntegerAttr>()
                    .getInt();
      int end =
          params.end.get<mlir::Attribute>().cast<mlir::IntegerAttr>().getInt();
      memref_shape.push_back(llvm::divideCeil(end - beg, step));
    } else {
      memref_shape.push_back(mlir::ShapedType::kDynamicSize);
      // Handle dynamic dimensions.
      mlir::Value beg = Materialize(buffer.getLoc(), params.begin, builder);
      mlir::Value end = Materialize(buffer.getLoc(), params.end, builder);
      auto d0 = mlir::getAffineDimExpr(0, context);
      auto d1 = mlir::getAffineDimExpr(1, context);
      auto map = mlir::AffineMap::get(2, 0, (d1 - d0).ceilDiv(step));
      scalar_sizes.push_back(builder.create<mlir::AffineApplyOp>(
          buffer.getLoc(), map, llvm::makeArrayRef({beg, end})));
    }
  }

  // Create a map operation that performs memref shape computations.
  ValueRange sizes;
  if (!scalar_sizes.empty()) {
    builder.create<SairReturnOp>(buffer.getLoc(), scalar_sizes);
    builder.restoreInsertionPoint(map_point);
    llvm::SmallVector<mlir::Type> map_types(
        scalar_sizes.size(), ValueType::get(shape, builder.getIndexType()));
    llvm::SmallVector<mlir::Attribute> map_buffers(
        scalar_sizes.size(), GetRegister0DBuffer(context));
    auto map_op = builder.create<SairMapOp>(
        buffer.getLoc(), map_types, /*domain=*/domain,
        /*mapping_array=*/builder.getArrayAttr(map_arguments.mappings()),
        /*values=*/map_arguments.values(), /*shape=*/shape,
        /*loop_nest=*/loop_nest,
        /*storage=*/builder.getArrayAttr(map_buffers));
    map_op.body().takeBody(region);
    sizes = map_op.getResults();
  } else {
    builder.restoreInsertionPoint(map_point);
  }

  return std::make_pair(memref_shape, sizes);
}

mlir::Value AllocateBuffer(const Buffer &buffer,
                           const LoopFusionAnalysis &fusion_analysis,
                           mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *context = builder.getContext();
  auto [alloc_point, free_point] = FindInsertionPoints(buffer, builder);

  // Create the domain for malloc and free.
  alloc_point.Set(builder);
  LoopNest loop_nest = fusion_analysis.GetLoopNest(buffer.loop_nest());
  DomainShapeAttr shape = loop_nest.Shape();
  llvm::SmallVector<mlir::Value> domain =
      CreatePlaceholderDomain(buffer.getLoc(), shape, builder);

  // Compute memref sizes.
  auto [memref_shape, sizes] =
      GetMemRefShape(buffer, shape, domain, alloc_point.loop_nest, builder);

  // Introduce a malloc operation.
  auto memref_type = mlir::MemRefType::get(memref_shape, buffer.element_type());
  auto type = ValueType::get(shape, memref_type);
  auto identity_mapping =
      MappingAttr::GetIdentity(context, shape.NumDimensions());
  llvm::SmallVector<mlir::Attribute> size_mappings(sizes.size(),
                                                   identity_mapping);

  mlir::Value alloc = builder.create<SairAllocOp>(
      buffer.getLoc(), type, domain,
      /*mapping_array=*/builder.getArrayAttr(size_mappings), sizes,
      /*loop_nest=*/alloc_point.loop_nest,
      /*storage=*/builder.getArrayAttr(GetRegister0DBuffer(context)));

  free_point.Set(builder);
  builder.create<SairFreeOp>(
      buffer.getLoc(), domain,
      /*mapping_array=*/builder.getArrayAttr(identity_mapping), alloc,
      /*loop_nest=*/free_point.loop_nest);
  return alloc;
}

// Implements storage attributes by replacing Sair values with memrefs.
class MaterializeBuffers
    : public MaterializeBuffersPassBase<MaterializeBuffers> {
  void runOnFunction() override {
    markAnalysesPreserved<LoopFusionAnalysis, IterationSpaceAnalysis>();

    mlir::OpBuilder builder(&getContext());
    getFunction().walk([&](SairProgramOp program) {
      auto storage_analysis = getChildAnalysis<StorageAnalysis>(program);
      auto fusion_analysis = getChildAnalysis<LoopFusionAnalysis>(program);
      for (auto &[name, buffer] : storage_analysis.buffers()) {
        AllocateBuffer(buffer, fusion_analysis, builder);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateMaterializeBuffersPass() {
  return std::make_unique<MaterializeBuffers>();
}

}  // namespace sair
