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
#include "sair_op_interfaces.h"
#include "sequence.h"
#include "storage.h"
#include "transforms/domain_utils.h"
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

// Creates a loop-nest that maps pointwise to the domain with given loop names.
mlir::ArrayAttr PointwiseLoopNest(llvm::ArrayRef<mlir::StringAttr> loop_names,
                                  const LoopFusionAnalysis &fusion_analysis,
                                  mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();

  llvm::SmallVector<mlir::Attribute> loops;
  loops.reserve(loop_names.size());
  for (int i = 0, e = loop_names.size(); i < e; ++i) {
    auto dim_expr = MappingDimExpr::get(i, context);
    mlir::IntegerAttr unroll = fusion_analysis.GetClass(loop_names[i])
                                   .GetUnrollAttr(*builder.getContext());
    loops.push_back(LoopAttr::get(loop_names[i], dim_expr, unroll, context));
  }
  return builder.getArrayAttr(loops);
}

// Find insertion points for alloc and free operations.
std::pair<ProgramPoint, ProgramPoint> FindInsertionPoints(
    const Buffer &buffer, const IterationSpaceAnalysis &iter_spaces,
    const SequenceAnalysis &sequence_analysis, mlir::OpBuilder &builder) {
  auto reads_writes =
      llvm::to_vector<8>(llvm::make_first_range(buffer.reads()));
  llvm::append_range(reads_writes, llvm::make_first_range(buffer.writes()));
  auto [first_access, last_access] = sequence_analysis.GetSpan(reads_writes);

  int num_loops = buffer.loop_nest().size();
  ProgramPoint alloc_point = sequence_analysis.FindInsertionPoint(
      iter_spaces, first_access, num_loops, Direction::kBefore);
  ProgramPoint free_point = sequence_analysis.FindInsertionPoint(
      iter_spaces, last_access, num_loops, Direction::kAfter);

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

  MapBodyBuilder map_body(domain.size(), context);
  llvm::SmallVector<int64_t> memref_shape;
  llvm::SmallVector<mlir::Value> scalar_sizes;

  auto loops_to_domain =
      loop_nest.DomainToLoops().Inverse().Resize(buffer.domain().size());
  builder.setInsertionPointToStart(&map_body.block());
  auto buffer_domain = llvm::to_vector<4>(llvm::map_range(
      buffer.domain(), [](ValueAccessInstance instance) -> ValueAccess {
        return {.value = instance.value.GetValue(),
                .mapping = instance.mapping};
      }));
  llvm::SmallVector<RangeParameters> range_parameters =
      GetRangeParameters(buffer.location(), buffer.mapping(), buffer_domain,
                         loops_to_domain, map_body, builder);
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
    auto decisions = DecisionsAttr::get(
        /*sequence=*/nullptr, /*loop_nest=*/loop_nest_attr,
        /*storage=*/builder.getArrayAttr(map_buffers),
        /*expansion=*/nullptr, /*copy_of=*/nullptr,
        /*operands=*/
        GetInstanceZeroOperands(context,
                                domain.size() + map_body.sair_values().size()),
        context);
    auto map_op = builder.create<SairMapOp>(
        buffer.location(), map_types, /*domain=*/domain,
        /*inputs=*/map_body.sair_values(), /*shape=*/shape,
        /*instances=*/builder.getArrayAttr({decisions}), /*copies=*/nullptr);
    map_op.body().takeBody(map_body.region());
    sizes = map_op.getResults();
  } else {
    builder.restoreInsertionPoint(map_point);
  }

  return std::make_pair(memref_shape, sizes);
}

mlir::Value AllocateBuffer(const Buffer &buffer,
                           const IterationSpaceAnalysis &iter_spaces,
                           const LoopFusionAnalysis &fusion_analysis,
                           SequenceAnalysis &sequence_analysis,
                           mlir::OpBuilder &builder) {
  mlir::MLIRContext *context = builder.getContext();
  auto [alloc_point, free_point] =
      FindInsertionPoints(buffer, iter_spaces, sequence_analysis, builder);

  // Create the domain for malloc and free.
  LoopNest loop_nest = fusion_analysis.GetLoopNest(buffer.loop_nest());
  DomainShapeAttr shape = loop_nest.Shape();
  llvm::SmallVector<mlir::Value> domain =
      CreatePlaceholderDomain(buffer.location(), shape, builder);

  // Compute memref sizes.
  mlir::ArrayAttr alloc_loop_nest =
      PointwiseLoopNest(alloc_point.loop_nest(), fusion_analysis, builder);
  auto [memref_shape, sizes] = GetMemRefShape(buffer, shape, domain, loop_nest,
                                              alloc_loop_nest, builder);

  // Introduce a malloc operation.
  auto memref_type = mlir::MemRefType::get(memref_shape, buffer.element_type());
  auto type = ValueType::get(shape, memref_type);
  auto identity_mapping =
      MappingAttr::GetIdentity(context, shape.NumDimensions());
  llvm::SmallVector<mlir::Attribute> size_mappings(sizes.size(),
                                                   identity_mapping);

  auto alloc_decisions = DecisionsAttr::get(
      /*sequence=*/nullptr,
      /*loop_nest=*/alloc_loop_nest,
      /*storage=*/builder.getArrayAttr(GetRegister0DBuffer(context)),
      /*expansion=*/builder.getStringAttr(kAllocExpansionPattern),
      /*copy_of=*/nullptr,
      /*operands=*/
      GetInstanceZeroOperands(context, domain.size() + sizes.size()), context);
  mlir::Value alloc = builder.create<SairAllocOp>(
      buffer.location(), type, domain,
      /*mapping_array=*/builder.getArrayAttr(size_mappings), sizes,
      /*decisions=*/builder.getArrayAttr({alloc_decisions}),
      /*copies=*/nullptr);
  sequence_analysis.Insert(
      ComputeOpInstance::Unique(alloc.getDefiningOp<ComputeOp>()), alloc_point);

  mlir::ArrayAttr free_loop_nest =
      PointwiseLoopNest(free_point.loop_nest(), fusion_analysis, builder);
  auto free_decisions = DecisionsAttr::get(
      /*sequence=*/nullptr,
      /*loop_nest=*/free_loop_nest,
      /*storage=*/nullptr,
      /*expansion=*/builder.getStringAttr(kFreeExpansionPattern),
      /*copy_of=*/nullptr,
      /*operands=*/GetInstanceZeroOperands(context, domain.size() + 1),
      context);
  auto free_op = builder.create<SairFreeOp>(
      buffer.location(), domain,
      /*mapping_array=*/builder.getArrayAttr(identity_mapping), alloc,
      /*instances=*/builder.getArrayAttr({free_decisions}));
  sequence_analysis.Insert(ComputeOpInstance::Unique(free_op), free_point);

  return alloc;
}

// Insert a load from a buffer for the operand `operand_pos` of `op`.
void InsertLoad(ComputeOp op, int operand_pos, const Buffer &buffer,
                ValueAccess memref, const LoopFusionAnalysis &fusion_analysis,
                const IterationSpaceAnalysis &iteration_spaces,
                const StorageAnalysis &storage_analysis,
                SequenceAnalysis &sequence_analysis, mlir::OpBuilder &builder) {
  // This isn't strictly necessary because Sair doesn't rely on textual order,
  // but leads to more readable IR with loads textually preceding the user.
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *context = op.getContext();
  builder.setInsertionPoint(op);

  auto sair_op = cast<SairOp>(op.getOperation());
  int op_domain_size = sair_op.domain().size();
  ValueOperand operand = sair_op.ValueOperands()[operand_pos];

  auto op_instance = OpInstance::Unique(sair_op);
  OperandInstance operand_instance = op_instance.Operand(operand_pos);
  const IterationSpace &op_iter_space = iteration_spaces.Get(op_instance);
  ValueStorage operand_storage =
      *storage_analysis.GetStorage(*operand_instance.GetValue())
           .Map(operand_instance, iteration_spaces);
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
      PointwiseLoopNest(op_iter_space.loop_names(), fusion_analysis, builder);
  BufferAttr loaded_storage = GetRegister0DBuffer(context);
  auto decisions = DecisionsAttr::get(
      /*sequence=*/nullptr, /*loop_nest=*/loop_nest,
      /*storage=*/builder.getArrayAttr({loaded_storage}),
      /*expansion=*/builder.getStringAttr(kLoadExpansionPattern),
      /*copy_of=*/nullptr,
      /*operands=*/GetInstanceZeroOperands(context, load_domain.size() + 1),
      context);
  mlir::Value loaded = builder.create<SairLoadFromMemRefOp>(
      op.getLoc(), loaded_type, load_domain,
      builder.getArrayAttr({memref_mapping}), memref.value,
      operand_storage.layout(), /*instances=*/builder.getArrayAttr({decisions}),
      /*copies=*/nullptr);

  auto load_instance =
      ComputeOpInstance::Unique(cast<ComputeOp>(loaded.getDefiningOp()));
  sequence_analysis.Insert(
      load_instance,
      ProgramPoint(ComputeOpInstance::Unique(op), Direction::kBefore));

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
        builder.getArrayAttr(proj_mapping), loaded, proj_shape,
        /*instances=*/
        GetInstanceZeroOperandsSingleInstance(context, proj_domain.size() + 1),
        /*copies=*/nullptr);
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
  // This isn't strictly necessary because Sair doesn't rely on textual order,
  // but leads to more readable IR with stores textually following the producer.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(op);

  auto sair_op = cast<SairOp>(op.getOperation());
  const IterationSpace &op_iter_space =
      iteration_spaces.Get(OpInstance::Unique(sair_op));
  mlir::Value result = op->getResult(result_pos);
  const ValueStorage &result_storage =
      storage_analysis.GetStorage(ResultInstance::Unique(result));

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
      PointwiseLoopNest(op_iter_space.loop_names(), fusion_analysis, builder);
  auto decisions = DecisionsAttr::get(
      /*sequence=*/nullptr, /*loop_nest=*/loop_nest, /*storage=*/nullptr,
      /*expansion=*/builder.getStringAttr(kStoreExpansionPattern),
      /*copy_of=*/nullptr,
      /*operands=*/
      GetInstanceZeroOperands(op.getContext(), store_domain.size() + 2),
      op.getContext());
  auto store_to_memref_op = builder.create<SairStoreToMemRefOp>(
      op.getLoc(), store_domain,
      builder.getArrayAttr({memref_mapping, result_mapping}), memref.value,
      result, result_storage.layout(), store_shape,
      /*instances=*/builder.getArrayAttr({decisions}),
      /*copies=*/nullptr);
  auto op_instance = ComputeOpInstance::Unique(op);
  auto to_memref_instance = ComputeOpInstance::Unique(
      cast<ComputeOp>(store_to_memref_op.getOperation()));
  sequence_analysis.Insert(to_memref_instance,
                           ProgramPoint(op_instance, Direction::kAfter));

  // Change result storage to register.
  op_instance.SetStorage(result_pos, GetRegister0DBuffer(op.getContext()));
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

    builder.setInsertionPointToStart(&program.body().front());
    for (auto &[name, buffer] : storage_analysis.buffers()) {
      ValueAccess memref;
      // Allocate or retrieve the buffer.
      if (buffer.is_external()) {
        auto import_op = cast<SairOp>(buffer.import_op().getOperation());
        const IterationSpace &iter_space =
            iteration_spaces.Get(OpInstance::Unique(import_op));
        ValueOperand memref_operand = buffer.import_op().MemRef();
        memref.value = memref_operand.value();
        memref.mapping =
            iter_space.mapping().Inverse().Compose(memref_operand.Mapping());
      } else {
        memref.value = AllocateBuffer(buffer, iteration_spaces, fusion_analysis,
                                      sequence_analysis, builder);
        memref.mapping =
            MappingAttr::GetIdentity(context, buffer.loop_nest().size());
      }

      // Insert loads and stores.
      for (auto [op, pos] : buffer.reads()) {
        auto compute_op = cast<ComputeOp>(op.GetDuplicatedOp());
        InsertLoad(compute_op, pos, buffer, memref, fusion_analysis,
                   iteration_spaces, storage_analysis, sequence_analysis,
                   builder);
      }
      for (auto [op, pos] : buffer.writes()) {
        auto compute_op = cast<ComputeOp>(op.GetDuplicatedOp());
        InsertStore(compute_op, pos, buffer, memref, fusion_analysis,
                    iteration_spaces, storage_analysis, sequence_analysis,
                    builder);
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
      for (ResultInstance value : buffer.values()) {
        mlir::Operation *op = value.defining_op().GetDuplicatedOp();
        if (!isa<SairProjAnyOp, SairProjLastOp, SairFbyOp>(op)) {
          continue;
        }

        op->dropAllDefinedValueUses();
        op->erase();
      }
    }

    sequence_analysis.AssignInferred();
  }

  void runOnOperation() override {
    markAnalysesPreserved<LoopFusionAnalysis, IterationSpaceAnalysis>();

    auto result = getOperation().walk([&](SairOp op) -> mlir::WalkResult {
      auto storage_analysis =
          getChildAnalysis<StorageAnalysis>(op->getParentOp());
      if (!op.HasExactlyOneInstance()) {
        return op.emitError() << "operations must have exactly one instance "
                                 "when materializing buffers";
      }
      for (mlir::Value result : op->getResults()) {
        if (!result.getType().isa<ValueType>()) continue;
        const ValueStorage &storage =
            storage_analysis.GetStorage(ResultInstance::Unique(result));
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

    getOperation().walk([&](SairProgramOp program) { RunOnProgram(program); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateMaterializeBuffersPass() {
  return std::make_unique<MaterializeBuffers>();
}

}  // namespace sair
