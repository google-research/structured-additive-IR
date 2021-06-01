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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "loop_nest.h"
#include "sequence.h"
#include "storage.h"
#include "transforms/domain_utils.h"
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

// Creates a loop-nest that maps pointwise to the domain with given loop names.
mlir::ArrayAttr PointwiseLoopNest(llvm::ArrayRef<mlir::StringAttr> loop_names,
                                  mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();

  llvm::SmallVector<mlir::Attribute> loops;
  loops.reserve(loop_names.size());
  for (int i = 0, e = loop_names.size(); i < e; ++i) {
    auto dim_expr = MappingDimExpr::get(i, context);
    loops.push_back(LoopAttr::get(loop_names[i], dim_expr, context));
  }
  return builder.getArrayAttr(loops);
}

// Find insertion points for alloc and free operations.
std::pair<InsertionPoint, InsertionPoint> FindInsertionPoints(
    const Buffer &buffer, const SequenceAnalysis &sequence_analysis,
    mlir::OpBuilder &builder) {
  auto reads_writes =
      llvm::to_vector<8>(llvm::make_first_range(buffer.reads()));
  llvm::append_range(reads_writes, llvm::make_first_range(buffer.writes()));
  auto [first_access, last_access] = sequence_analysis.GetSpan(reads_writes);

  int num_loops = buffer.loop_nest().size();
  InsertionPoint alloc_point = sequence_analysis.FindInsertionPoint(
      cast<SairOp>(first_access.getOperation()), first_access.LoopNestLoops(),
      num_loops, Direction::kBefore);
  InsertionPoint free_point = sequence_analysis.FindInsertionPoint(
      cast<SairOp>(last_access.getOperation()), last_access.LoopNestLoops(),
      num_loops, Direction::kAfter);

  // Define a loop-nest in the domain of loops.
  mlir::ArrayAttr loop_nest = PointwiseLoopNest(buffer.loop_nest(), builder);
  alloc_point.loop_nest = loop_nest;
  free_point.loop_nest = loop_nest;
  return std::make_pair(alloc_point, free_point);
}

// Returns the shape of the memref implementing `buffer` and the list of values
// providing dynamic dimension sizes.
std::pair<mlir::SmallVector<int64_t>, ValueRange> GetMemRefShape(
    const Buffer &buffer, DomainShapeAttr shape,
    llvm::ArrayRef<mlir::Value> domain, const LoopNest &loop_nest,
    mlir::ArrayAttr loop_nest_attr, mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  mlir::OpBuilder::InsertPoint map_point = builder.saveInsertionPoint();

  // Create a block that will hold computations for memref sizes.
  mlir::Region region;
  llvm::SmallVector<mlir::Type> block_arg_types(domain.size(),
                                                builder.getIndexType());
  mlir::Block *block = builder.createBlock(&region, {}, block_arg_types);
  llvm::SmallVector<ValueAccess> map_arguments;

  llvm::SmallVector<int64_t> memref_shape;
  llvm::SmallVector<mlir::Value> scalar_sizes;

  auto loops_to_domain =
      loop_nest.DomainToLoops().Inverse().Resize(buffer.domain().size());
  llvm::SmallVector<RangeParameters> range_parameters =
      GetRangeParameters(buffer.location(), buffer.mapping(), buffer.domain(),
                         loops_to_domain, map_arguments, *block, builder);
  for (const auto &params : range_parameters) {
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
      mlir::Value beg = Materialize(buffer.location(), params.begin, builder);
      mlir::Value end = Materialize(buffer.location(), params.end, builder);
      auto d0 = mlir::getAffineDimExpr(0, context);
      auto d1 = mlir::getAffineDimExpr(1, context);
      auto map = mlir::AffineMap::get(2, 0, (d1 - d0).ceilDiv(step));
      scalar_sizes.push_back(builder.create<mlir::AffineApplyOp>(
          buffer.location(), map, llvm::makeArrayRef({beg, end})));
    }
  }

  // Create a map operation that performs memref shape computations.
  ValueRange sizes;
  if (!scalar_sizes.empty()) {
    builder.create<SairReturnOp>(buffer.location(), scalar_sizes);
    builder.restoreInsertionPoint(map_point);
    llvm::SmallVector<mlir::Type> map_types(
        scalar_sizes.size(), ValueType::get(shape, builder.getIndexType()));
    llvm::SmallVector<mlir::Attribute> map_buffers(
        scalar_sizes.size(), GetRegister0DBuffer(context));
    auto map_op = builder.create<SairMapOp>(
        buffer.location(), map_types, /*domain=*/domain,
        /*inputs=*/map_arguments, /*shape=*/shape,
        /*loop_nest=*/loop_nest_attr,
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
                           SequenceAnalysis &sequence_analysis,
                           mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *context = builder.getContext();
  auto [alloc_point, free_point] =
      FindInsertionPoints(buffer, sequence_analysis, builder);

  // Create the domain for malloc and free.
  alloc_point.Set(builder);
  LoopNest loop_nest = fusion_analysis.GetLoopNest(buffer.loop_nest());
  DomainShapeAttr shape = loop_nest.Shape();
  llvm::SmallVector<mlir::Value> domain =
      CreatePlaceholderDomain(buffer.location(), shape, builder);

  // Compute memref sizes.
  auto [memref_shape, sizes] = GetMemRefShape(buffer, shape, domain, loop_nest,
                                              alloc_point.loop_nest, builder);

  // Introduce a malloc operation.
  auto memref_type = mlir::MemRefType::get(memref_shape, buffer.element_type());
  auto type = ValueType::get(shape, memref_type);
  auto identity_mapping =
      MappingAttr::GetIdentity(context, shape.NumDimensions());
  llvm::SmallVector<mlir::Attribute> size_mappings(sizes.size(),
                                                   identity_mapping);

  mlir::Value alloc = builder.create<SairAllocOp>(
      buffer.location(), type, domain,
      /*mapping_array=*/builder.getArrayAttr(size_mappings), sizes,
      /*loop_nest=*/alloc_point.loop_nest,
      /*storage=*/builder.getArrayAttr(GetRegister0DBuffer(context)),
      /*sequence=*/IntegerAttr());
  sequence_analysis.Insert(alloc.getDefiningOp<ComputeOp>(),
                           cast<ComputeOp>(alloc_point.operation),
                           Direction::kBefore);

  free_point.Set(builder);
  auto free_op = builder.create<SairFreeOp>(
      buffer.location(), domain,
      /*mapping_array=*/builder.getArrayAttr(identity_mapping), alloc,
      /*loop_nest=*/free_point.loop_nest,
      /*sequence=*/IntegerAttr());
  sequence_analysis.Insert(free_op, cast<ComputeOp>(free_point.operation),
                           Direction::kAfter);

  return alloc;
}

// Insert a load from a buffer for the operand `operand_pos` of `op`.
void InsertLoad(ComputeOp op, int operand_pos, const Buffer &buffer,
                ValueAccess memref, const LoopFusionAnalysis &fusion_analysis,
                const IterationSpaceAnalysis &iteration_spaces,
                const StorageAnalysis &storage_analysis,
                SequenceAnalysis &sequence_analysis, mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *context = op.getContext();
  builder.setInsertionPoint(op);

  auto sair_op = cast<SairOp>(op.getOperation());
  int op_domain_size = sair_op.domain().size();
  ValueOperand operand = sair_op.ValueOperands()[operand_pos];
  const IterationSpace &op_iter_space = iteration_spaces.Get(sair_op);
  ValueStorage operand_storage = storage_analysis.GetStorage(operand.value())
                                     .Map(operand, iteration_spaces);
  mlir::Type element_type = operand.GetType().ElementType();

  // Create a placeholder domain for the load.
  DomainShapeAttr load_shape =
      fusion_analysis.GetLoopNest(op_iter_space.loop_names()).Shape();
  llvm::SmallVector<mlir::Value> load_domain =
      CreatePlaceholderDomain(buffer.location(), load_shape, builder);

  // Create a load_from_memref operation.
  auto memref_mapping =
      memref.mapping.ResizeUseDomain(op_iter_space.num_loops());
  auto loaded_type = ValueType::get(load_shape, element_type);
  mlir::ArrayAttr loop_nest =
      PointwiseLoopNest(op_iter_space.loop_names(), builder);
  BufferAttr loaded_storage = GetRegister0DBuffer(context);
  mlir::Value loaded = builder.create<SairLoadFromMemRefOp>(
      op.getLoc(), loaded_type, load_domain,
      builder.getArrayAttr({memref_mapping}), memref.value,
      operand_storage.layout(), loop_nest,
      builder.getArrayAttr({loaded_storage}), IntegerAttr());
  sequence_analysis.Insert(loaded.getDefiningOp<ComputeOp>(), op,
                           Direction::kBefore);

  // Insert a sair.proj_any operation in case the load is rematerialized.
  ValueAccess new_operand;
  if (op_iter_space.mapping().HasNoneExprs()) {
    MappingAttr proj_mapping = op_iter_space.mapping().MakeSurjective();
    DomainShapeAttr proj_shape =
        load_shape.AccessedShape(proj_mapping.Inverse());
    llvm::SmallVector<mlir::Value> proj_domain =
        CreatePlaceholderDomain(buffer.location(), proj_shape, builder);
    auto proj_type =
        ValueType::get(proj_shape.Prefix(op_domain_size), element_type);
    new_operand.value = builder.create<SairProjAnyOp>(
        op.getLoc(), proj_type,
        llvm::makeArrayRef(proj_domain).take_front(op_domain_size),
        llvm::makeArrayRef(proj_domain).drop_front(op_domain_size),
        builder.getArrayAttr(proj_mapping), loaded, proj_shape);
    new_operand.mapping = MappingAttr::GetIdentity(context, op_domain_size);
  } else {
    new_operand.value = loaded;
    new_operand.mapping = op_iter_space.mapping();
  }

  // Substitute the operand.
  operand.set_value(new_operand.value);
  operand.SetMapping(new_operand.mapping);
}

// Insert a store for the result `result_pos` of `op`.
void InsertStore(ComputeOp op, int result_pos, const Buffer &buffer,
                 ValueAccess memref, const LoopFusionAnalysis &fusion_analysis,
                 const IterationSpaceAnalysis &iteration_spaces,
                 const StorageAnalysis &storage_analysis,
                 SequenceAnalysis &sequence_analysis,
                 mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(op);

  auto sair_op = cast<SairOp>(op.getOperation());
  const IterationSpace &op_iter_space = iteration_spaces.Get(sair_op);
  mlir::Value result = op->getResult(result_pos);
  const ValueStorage &result_storage = storage_analysis.GetStorage(result);

  // Create a placeholder domain for the store operation.
  DomainShapeAttr store_shape =
      fusion_analysis.GetLoopNest(op_iter_space.loop_names()).Shape();
  llvm::SmallVector<mlir::Value> store_domain =
      CreatePlaceholderDomain(buffer.location(), store_shape, builder);

  // Create a store operation.
  auto memref_mapping =
      memref.mapping.ResizeUseDomain(op_iter_space.num_loops());
  auto result_mapping = op_iter_space.mapping().Inverse();
  mlir::ArrayAttr loop_nest =
      PointwiseLoopNest(op_iter_space.loop_names(), builder);
  auto store_to_memref_op = builder.create<SairStoreToMemRefOp>(
      op.getLoc(), store_domain,
      builder.getArrayAttr({memref_mapping, result_mapping}), memref.value,
      result, result_storage.layout(), store_shape, loop_nest, IntegerAttr());
  sequence_analysis.Insert(store_to_memref_op, op, Direction::kAfter);

  // Change result storage to register.
  op.SetStorage(result_pos, GetRegister0DBuffer(op.getContext()));
}

// Implements storage attributes by replacing Sair values with memrefs.
class MaterializeBuffers
    : public MaterializeBuffersPassBase<MaterializeBuffers> {
  void RunOnProgram(SairProgramOp program) {
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);
    auto storage_analysis = getChildAnalysis<StorageAnalysis>(program);
    auto sequence_analysis = getChildAnalysis<SequenceAnalysis>(program);
    auto fusion_analysis = getChildAnalysis<LoopFusionAnalysis>(program);
    auto iteration_spaces = getChildAnalysis<IterationSpaceAnalysis>(program);

    for (auto &[name, buffer] : storage_analysis.buffers()) {
      ValueAccess memref;
      // Allocate or retrieve the buffer.
      if (buffer.is_external()) {
        const IterationSpace &iter_space = iteration_spaces.Get(
            cast<SairOp>(buffer.import_op().getOperation()));
        ValueOperand memref_operand = buffer.import_op().MemRef();
        memref.value = memref_operand.value();
        memref.mapping =
            iter_space.mapping().Inverse().Compose(memref_operand.Mapping());
      } else {
        memref.value =
            AllocateBuffer(buffer, fusion_analysis, sequence_analysis, builder);
        memref.mapping =
            MappingAttr::GetIdentity(context, buffer.loop_nest().size());
      }

      // Insert loads and stores.
      for (auto [op, pos] : buffer.reads()) {
        InsertLoad(op, pos, buffer, memref, fusion_analysis, iteration_spaces,
                   storage_analysis, sequence_analysis, builder);
      }
      for (auto [op, pos] : buffer.writes()) {
        InsertStore(op, pos, buffer, memref, fusion_analysis, iteration_spaces,
                    storage_analysis, sequence_analysis, builder);
      }

      // Erase ToMemRefOp as it has side effects and wont be considered by
      // DCE.
      if (buffer.is_external() &&
          isa<SairToMemRefOp>(buffer.import_op().getOperation())) {
        buffer.import_op()->erase();
      }

      // Drop proj_* and fby operations as we changed the operation storage to
      // register which might make verifier complain that storage volume is
      // insufficient. These operations would be removed by DCE anyway.
      for (mlir::Value value : buffer.values()) {
        mlir::Operation *op = value.getDefiningOp();
        if (!isa<SairProjAnyOp, SairProjLastOp, SairFbyOp>(op)) {
          continue;
        }

        op->dropAllDefinedValueUses();
        op->erase();
      }
    }

    sequence_analysis.AssignInferred();
  }

  void runOnFunction() override {
    markAnalysesPreserved<LoopFusionAnalysis, IterationSpaceAnalysis>();

    auto result = getFunction().walk([&](SairOp op) -> mlir::WalkResult {
      auto storage_analysis =
          getChildAnalysis<StorageAnalysis>(op->getParentOp());
      for (mlir::Value result : op->getResults()) {
        if (!result.getType().isa<ValueType>()) continue;
        const ValueStorage &storage = storage_analysis.GetStorage(result);
        if (storage.space() == nullptr) {
          return op.emitError() << "missing memory space";
        }
        if (storage.layout() == nullptr) {
          return op.emitError() << "missing layout";
        }
        if (storage.layout().HasNoneExprs()) {
          return op.emitError() << "partial layouts are not yet supported";
        }
      }
      return mlir::success();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    getFunction().walk([&](SairProgramOp program) { RunOnProgram(program); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateMaterializeBuffersPass() {
  return std::make_unique<MaterializeBuffers>();
}

}  // namespace sair
