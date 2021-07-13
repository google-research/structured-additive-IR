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

#include "loop_nest.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sequence.h"
#include "storage.h"
#include "transforms/domain_utils.h"
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

// Creates a range operation for loop `loop` of `loop_nest`. Appends
// the dimension to `new_domain`.
// * `inverse_mapping` is the mapping from loops to `op` domain.
// * `new_loop_nest` contains LoopAttr attributes for outer loops and this one.
void CreateRange(SairOp op, const LoopNest &loop_nest, int loop,
                 const DomainShapeDim &loop_shape, MappingAttr inverse_mapping,
                 llvm::ArrayRef<mlir::Attribute> new_loop_nest,
                 SequenceAnalysis &sequence_analysis,
                 llvm::SmallVectorImpl<mlir::Value> &new_domain,
                 mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::MLIRContext *context = op.getContext();

  // Find the loop nest and domain of the new operation.
  int range_rank = loop_shape.dependency_mapping().MinDomainSize();
  auto range_domain = llvm::makeArrayRef(new_domain).take_front(range_rank);
  InsertionPoint insertion_point =
      sequence_analysis.FindInsertionPoint(op, new_loop_nest, range_rank);

  // Populate a new map operation body with computations for the bounds of the
  // new range.
  MapBodyBuilder map_body(range_rank, context);
  builder.setInsertionPointToStart(&map_body.block());
  RangeParameters range_parameters = GetRangeParameters(
      op.getLoc(), loop_nest.DomainToLoops().Slice(loop, 1), loop_nest.domain(),
      inverse_mapping.ResizeUseDomain(range_rank), map_body, builder)[0];

  // Create a sair.map operation with `block` as body and add a sair.return
  // operation to `block`. Create a range operation that uses the bounds
  // returned by the sair.map operations. If range parameters are constants and
  // range lower bound is zero, delete the block and create a static range
  // instead.
  mlir::Value range;
  bool is_beg_zero = range_parameters.begin.is<mlir::Attribute>() &&
                     range_parameters.begin.get<mlir::Attribute>()
                             .cast<mlir::IntegerAttr>()
                             .getInt() == 0;
  auto step =
      mlir::IntegerAttr::get(builder.getIndexType(), range_parameters.step);
  if (range_parameters.end.is<mlir::Attribute>() && is_beg_zero) {
    assert(range_rank == 0);
    insertion_point.Set(builder);
    range = builder.create<SairStaticRangeOp>(op.getLoc(), loop_shape.type());
  } else {
    llvm::SmallVector<mlir::Value> scalar_results;
    if (!is_beg_zero) {
      scalar_results.push_back(
          Materialize(op.getLoc(), range_parameters.begin, builder));
    }
    scalar_results.push_back(
        Materialize(op.getLoc(), range_parameters.end, builder));
    builder.create<SairReturnOp>(op.getLoc(), scalar_results);

    insertion_point.Set(builder);
    DomainShapeAttr range_shape = loop_shape.type().Shape();
    llvm::SmallVector<mlir::Type> map_result_types(
        scalar_results.size(),
        ValueType::get(range_shape, builder.getIndexType()));
    llvm::SmallVector<mlir::Attribute> map_buffers(
        scalar_results.size(), GetRegister0DBuffer(builder.getContext()));

    // The dimension either corresponds to a loop (in which case
    // the full loop nest is bigger than the number of inner loops) or to a
    // dimension of the original operation (in which case a sair.map should not
    // be needed).
    assert(new_loop_nest.size() >= range_rank);
    auto map_loop_nest =
        builder.getArrayAttr(new_loop_nest.take_front(range_rank));

    auto map_op = builder.create<SairMapOp>(
        op.getLoc(), map_result_types,
        /*domain=*/range_domain,
        /*inputs=*/map_body.sair_values(),
        /*shape=*/range_shape,
        /*loop_nest=*/map_loop_nest,
        /*storage=*/builder.getArrayAttr(map_buffers),
        /*sequence=*/nullptr,
        /*expansion=*/builder.getStringAttr(kMapExpansionPattern));
    sequence_analysis.Insert(cast<ComputeOp>(map_op.getOperation()),
                             cast<SairOp>(insertion_point.operation),
                             Direction::kBefore);
    map_op.body().takeBody(map_body.region());
    auto identity_mapping = MappingAttr::GetIdentity(context, range_rank);
    llvm::SmallVector<mlir::Attribute> range_mappings(scalar_results.size(),
                                                      identity_mapping);
    range = builder.create<SairDynRangeOp>(
        op.getLoc(), loop_shape.type(),
        /*domain=*/range_domain,
        /*mapping_array=*/builder.getArrayAttr(range_mappings),
        /*begin=*/is_beg_zero ? nullptr : map_op.getResult(0),
        /*end=*/map_op.getResults().back(),
        /*step=*/step);
  }

  new_domain.push_back(range);
}

// Caches loop ranges. Indexed by loop name.
using LoopRangeCache = llvm::DenseMap<mlir::Attribute, mlir::Value>;

// Create a domain for `op` where each dimension corresponds to a single loop.
llvm::SmallVector<mlir::Value> GetDomain(
    SairOp op, llvm::ArrayRef<mlir::StringAttr> loop_names,
    const LoopNest &loop_nest, DomainShapeAttr shape,
    SequenceAnalysis &sequence_analysis,
    llvm::ArrayRef<mlir::Attribute> normalized_loops, mlir::OpBuilder &builder,
    LoopRangeCache &loop_range_cache) {
  MappingAttr inverse_mapping = loop_nest.DomainToLoops().Inverse();

  llvm::SmallVector<mlir::Value> new_domain;
  new_domain.reserve(loop_names.size());

  for (int i = 0, e = loop_names.size(); i < e; ++i) {
    auto cache = loop_range_cache.find(loop_names[i]);
    if (cache != loop_range_cache.end()) {
      new_domain.push_back(cache->second);
      continue;
    }

    CreateRange(op, loop_nest, i, shape.Dimension(i), inverse_mapping,
                normalized_loops, sequence_analysis, new_domain, builder);
    loop_range_cache.try_emplace(loop_names[i], new_domain.back());
  }

  return new_domain;
}

// Partitions `domain` into subdomains so that the image of a dimension of `op`
// through `mapping` remains in the same domain. Reorders dimensions of shape
// and updates mapping to match the new domain order.
llvm::SmallVector<llvm::SmallVector<mlir::Value>> PartitionDomain(
    SairOp op, MappingAttr &mapping, DomainShapeAttr &shape,
    llvm::SmallVector<mlir::Value> domain) {
  mlir::MLIRContext *context = op.getContext();

  // Permute dimensions to preserve sub-domains.
  llvm::SmallVector<int> sub_domains = op.SubDomains();
  llvm::SmallVector<llvm::SmallVector<MappingExpr>> partitioned_mapping(
      sub_domains.size());
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> partitioned_domain(
      sub_domains.size());
  llvm::SmallVector<llvm::SmallVector<DomainShapeDim>> partitioned_shape(
      sub_domains.size());

  for (int i = 0; i < domain.size(); ++i) {
    int sub_domain = 0;
    // Find the subdomain the loop belongs to.
    int min_domain_size = mapping.Dimension(i).MinDomainSize();
    if (min_domain_size > 0) {
      while (min_domain_size > sub_domains[sub_domain]) {
        min_domain_size -= sub_domains[sub_domain++];
      }
    }

    partitioned_mapping[sub_domain].push_back(mapping.Dimension(i));
    partitioned_domain[sub_domain].push_back(domain[i]);
    partitioned_shape[sub_domain].push_back(shape.Dimension(i));
  }

  llvm::SmallVector<MappingExpr> flattened_mapping;
  flattened_mapping.reserve(mapping.size());
  for (const auto &sub_domain : partitioned_mapping) {
    llvm::append_range(flattened_mapping, sub_domain);
  }
  mapping = MappingAttr::get(context, op.domain().size(), flattened_mapping);

  llvm::SmallVector<DomainShapeDim> shape_dims;
  shape_dims.reserve(shape.NumDimensions());
  for (const auto &sub_domain : partitioned_shape) {
    for (const auto &dim : sub_domain) {
      shape_dims.emplace_back(
          dim.type(),
          dim.dependency_mapping().ResizeUseDomain(shape_dims.size()));
    }
  }
  shape = DomainShapeAttr::get(context, shape_dims);
  return partitioned_domain;
}

// Replaces `op` by a copy with a different domain such that each loop in the
// iteration space corresponds to a full dimension.
void NormalizeLoops(SairOp op, const IterationSpace &iteration_space,
                    const LoopNest &loop_nest,
                    const LoopFusionAnalysis &fusion_analysis,
                    SequenceAnalysis &sequence_analysis,
                    mlir::OpBuilder &builder,
                    LoopRangeCache &loop_range_cache) {
  if (iteration_space.num_loops() == 0) return;
  mlir::OpBuilder::InsertionGuard insertion_guard(builder);
  mlir::MLIRContext *context = op.getContext();
  DomainShapeAttr new_shape = loop_nest.NormalizedShape();

  llvm::SmallVector<mlir::Attribute> normalized_loops;
  normalized_loops.reserve(iteration_space.num_loops());
  for (int i = 0, e = iteration_space.num_loops(); i < e; ++i) {
    auto dim_expr = MappingDimExpr::get(i, context);
    mlir::StringAttr name = iteration_space.loop_names()[i];
    mlir::IntegerAttr unroll_attr =
        fusion_analysis.GetClass(name).GetUnrollAttr(*builder.getContext());
    normalized_loops.push_back(
        LoopAttr::get(name, dim_expr, unroll_attr, context));
  }

  MappingAttr mapping = iteration_space.MappingToLoops();
  llvm::SmallVector<mlir::Value> new_domain =
      GetDomain(op, iteration_space.loop_names(), loop_nest, new_shape,
                sequence_analysis, normalized_loops, builder, loop_range_cache);
  llvm::SmallVector<llvm::SmallVector<mlir::Value>> partitioned_domain =
      PartitionDomain(op, mapping, new_shape, new_domain);

  // Replace op by an operation with the domain changed.
  builder.setInsertionPoint(op);
  SairOp new_op = op.ReCreateWithNewDomain(partitioned_domain, new_shape,
                                           mapping.Inverse(), builder);
  if (auto compute_op = dyn_cast<ComputeOp>(*new_op)) {
    sequence_analysis.Insert(compute_op, op, Direction::kBefore);
    compute_op.setLoopNest(builder.getArrayAttr(normalized_loops));
  }

  MappingAttr result_mapping =
      mapping.Resize(new_op.results_rank()).ResizeUseDomain(op.results_rank());

  // Handle the case where there is no rematerialization.
  if (result_mapping.IsSurjective()) {
    for (auto [old_value, new_value] :
         llvm::zip(op->getResults(), new_op->getResults())) {
      // We do not normalize range operations, so we know that results are
      // values.
      assert(old_value.getType().isa<ValueType>());
      UpdateValueUses(old_value,
                      {new_value, mapping.Resize(new_op.results_rank())});
    }
    if (auto compute_op = dyn_cast<ComputeOp>(op.getOperation())) {
      sequence_analysis.Erase(compute_op);
    }
    op.erase();
    return;
  }

  // Create proj_any operations with placeholder dimensions for the domain, as
  // we won't use them anyway.
  result_mapping = result_mapping.MakeSurjective();
  MappingAttr inverse_result_mapping = result_mapping.Inverse();

  DomainShapeAttr proj_any_shape =
      new_shape.AccessedShape(inverse_result_mapping);
  llvm::SmallVector<mlir::Value> proj_any_domain =
      CreatePlaceholderDomain(op.getLoc(), proj_any_shape, builder);

  for (auto [old_value, new_value] :
       llvm::zip(op->getResults(), new_op->getResults())) {
    auto proj_any = builder.create<SairProjAnyOp>(
        op.getLoc(), old_value.getType(),
        llvm::makeArrayRef(proj_any_domain).take_front(op.results_rank()),
        llvm::makeArrayRef(proj_any_domain).drop_front(op.results_rank()),
        builder.getArrayAttr({result_mapping}), new_value, proj_any_shape);
    // We can directly replace uses without updating mappings as the mapping is
    // already applied for the proj_any operation.
    old_value.replaceAllUsesWith(proj_any);
  }

  if (auto compute_op = dyn_cast<ComputeOp>(op.getOperation())) {
    sequence_analysis.Erase(compute_op);
  }
  op.erase();
}

// Pass that rewrites operation domains so that each loop corresponds to a
// single dimension.
class NormalizeLoopsPass : public NormalizeLoopsPassBase<NormalizeLoopsPass> {
 public:
  void runOnFunction() override {
    mlir::OpBuilder builder(&getContext());
    getFunction().walk([&](SairProgramOp program) {
      LoopRangeCache loop_range_cache;
      auto iteration_spaces = getChildAnalysis<IterationSpaceAnalysis>(program);
      auto fusion_analysis = getChildAnalysis<LoopFusionAnalysis>(program);
      auto sequence_analysis = getChildAnalysis<SequenceAnalysis>(program);

      llvm::SmallVector<SairOp> ops;
      program.walk([&](SairOp op) {
        // Do not normalize range and placeholder operations.
        if (isa<RangeOp, SairPlaceholderOp>(op.getOperation())) return;
        ops.push_back(op);
      });

      for (SairOp op : ops) {
        if (isa<SairFromMemRefOp, SairToMemRefOp>(*op)) {
          op.emitError() << "sair.from_memref and sair.to_memref must be "
                            "eliminated before loop normalization";
          signalPassFailure();
          return;
        }

        // Do not normalize range operations.
        const IterationSpace &iteration_space = iteration_spaces.Get(op);
        if (iteration_space.mapping() != iteration_space.MappingToLoops()) {
          // This error should only occur if
          // * memref introduction or canonicalization was not run beforehand or
          // * some loop-nests are missing.
          //
          // Canonicalization will remove unused dimensions from non-compute
          // operations while memref introduction will remove non-compute
          // operations are not in the same iteration space than the producers
          // of their operands
          op.emitError() << "operation with an incomplete iteration space";
          signalPassFailure();
          return;
        }

        LoopNest loop_nest =
            fusion_analysis.GetLoopNest(iteration_space.loop_names());
        NormalizeLoops(op, iteration_space, loop_nest, fusion_analysis,
                       sequence_analysis, builder, loop_range_cache);
      }

      sequence_analysis.AssignInferred();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateNormalizeLoopsPass() {
  return std::make_unique<NormalizeLoopsPass>();
}

}  // namespace sair
