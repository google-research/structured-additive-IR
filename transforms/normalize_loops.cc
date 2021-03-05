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
#include "storage.h"
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

// Materializes `value` as an mlir value.
mlir::Value Materialize(mlir::Location loc, mlir::OpFoldResult &value,
                        mlir::OpBuilder &builder) {
  if (value.is<mlir::Value>()) return value.get<mlir::Value>();
  return builder.create<ConstantOp>(loc, value.get<mlir::Attribute>());
}

void CreateRange(SairOp op, MappingExpr expr, mlir::ValueRange old_domain,
                 DomainShapeAttr old_shape, MappingAttr inverse_mapping,
                 llvm::ArrayRef<mlir::Attribute> loop_nest,
                 const IterationSpaceAnalysis &iteration_spaces,
                 mlir::OpBuilder &builder,
                 llvm::SmallVectorImpl<mlir::Value> &new_domain,
                 llvm::SmallVectorImpl<DomainShapeDim> &new_shape) {
  mlir::OpBuilder::InsertionGuard guard(builder);

  // Find the loop nest and domain of the new operation.
  llvm::SmallBitVector dependency_mask =
      expr.AccessedShape(old_shape.Dimensions(), inverse_mapping)
          .dependency_mapping()
          .DependencyMask();
  int num_loops = dependency_mask.find_last() + 1;
  // TODO(b/175664160): this fails if op is not a ComputeOp and compute ops
  // surrounding op are not nested in loop_nest. We can fix this by normalizing
  // instructions order so that this property is satisfied.
  InsertionPoint insertion_point = FindInsertionPoint(op, loop_nest, num_loops);

  auto identity_mapping = MappingAttr::GetIdentity(op.getContext(), num_loops);
  auto range_shape = DomainShapeAttr::get(
      op.getContext(),
      llvm::ArrayRef<DomainShapeDim>(new_shape).take_front(num_loops));
  auto range_type = RangeType::get(range_shape);
  auto range_domain =
      llvm::ArrayRef<mlir::Value>(new_domain).take_front(num_loops);

  // Create a block that will hold computations for the new bounds of the
  // range.
  mlir::Region region;
  llvm::SmallVector<mlir::Type, 4> block_arg_types(num_loops,
                                                   builder.getIndexType());
  mlir::Block *block = builder.createBlock(&region, {}, block_arg_types);

  // Populate the block with the computations for the bounds of the new range.
  // Adds !sair.value arguments necessary to compute the bounds to `arguments`,
  // and `arguments_mappings` and corresponding scalars to block arguments.
  MapArguments map_arguments(block, num_loops);
  RangeParameters range_parameters =
      expr.GetRangeParameters(op.getLoc(), old_domain, old_shape,
                              inverse_mapping, builder, map_arguments);

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
    assert(num_loops == 0);
    auto size =
        range_parameters.end.get<mlir::Attribute>().cast<mlir::IntegerAttr>();
    insertion_point.Set(builder);
    range =
        builder.create<SairStaticRangeOp>(op.getLoc(), range_type, size, step);
  } else {
    llvm::SmallVector<mlir::Value, 2> scalar_results;
    if (!is_beg_zero) {
      scalar_results.push_back(
          Materialize(op.getLoc(), range_parameters.begin, builder));
    }
    scalar_results.push_back(
        Materialize(op.getLoc(), range_parameters.end, builder));
    builder.create<SairReturnOp>(op.getLoc(), scalar_results);

    insertion_point.Set(builder);
    llvm::SmallVector<mlir::Type, 2> map_result_types(
        scalar_results.size(),
        ValueType::get(range_shape, builder.getIndexType()));
    llvm::SmallVector<mlir::Attribute, 2> map_buffers(
        scalar_results.size(), GetRegister0DBuffer(builder.getContext()));

    // The dimension either corresponds to a loop (in which case
    // the full loop nest is bigger than the number of inner loops) or to a
    // dimension of the original operation (in which case a sair.map should not
    // be needed).
    assert(loop_nest.size() >= num_loops);
    auto map_loop_nest = builder.getArrayAttr(loop_nest.take_front(num_loops));

    auto map_op = builder.create<SairMapOp>(
        op.getLoc(), map_result_types,
        /*domain=*/range_domain,
        /*mapping_array=*/builder.getArrayAttr(map_arguments.mappings()),
        /*values=*/map_arguments.values(),
        /*shape=*/range_shape,
        /*loop_nest=*/map_loop_nest,
        /*storage=*/builder.getArrayAttr(map_buffers));
    map_op.body().takeBody(region);
    llvm::SmallVector<mlir::Attribute, 2> range_mappings(scalar_results.size(),
                                                         identity_mapping);
    range = builder.create<SairDynRangeOp>(
        op.getLoc(), range_type,
        /*domain=*/range_domain,
        /*mapping_array=*/builder.getArrayAttr(range_mappings),
        /*begin=*/is_beg_zero ? nullptr : map_op.getResult(0),
        /*end=*/map_op.getResults().back(),
        /*step=*/step);
  }

  new_domain.push_back(range);
  new_shape.emplace_back(range_type, identity_mapping);
}

// Replaces `op` by a copy with a different domain such that each loop in the
// iteration space corresponds to a full dimension.
mlir::LogicalResult NormalizeLoops(
    SairOp op, IterationSpaceAnalysis iteration_spaces,
    mlir::OpBuilder &builder,
    llvm::DenseMap<mlir::Attribute, std::pair<mlir::Value, DomainShapeDim>>
        &loop_range_cache) {
  llvm::ArrayRef<mlir::Attribute> iteration_space =
      iteration_spaces.IterationSpace(op);
  if (iteration_space.empty()) return mlir::success();
  mlir::MLIRContext *context = op.getContext();

  // Compute the mapping from the old domain to the new.
  llvm::SmallVector<MappingExpr, 4> iter_exprs;
  iter_exprs.reserve(iteration_space.size());
  llvm::SmallVector<mlir::Attribute, 4> normalized_loops;
  normalized_loops.reserve(iteration_space.size());
  for (auto ei : llvm::enumerate(iteration_space)) {
    LoopAttr loop = ei.value().cast<LoopAttr>();
    iter_exprs.push_back(loop.iter());
    auto dim_expr = MappingDimExpr::get(ei.index(), context);
    normalized_loops.push_back(LoopAttr::get(loop.name(), dim_expr, context));
  }
  MappingAttr mapping =
      MappingAttr::get(op.getContext(), op.domain().size(), iter_exprs);
  if (!mapping.IsFullySpecified()) {
    return op.emitError()
           << "loop normalization called on a partially specified loop nest";
  }

  // Complete the mapping so that it covers all dimensions.
  MappingAttr inverse_mapping = mapping.Inverse().MakeFullySpecified();
  mapping = inverse_mapping.Inverse();

  // Create new ranges.
  llvm::SmallVector<mlir::Value, 4> new_domain;
  llvm::SmallVector<DomainShapeDim, 4> new_shape_dims;
  for (auto ie : llvm::enumerate(mapping.Dimensions())) {
    MappingExpr iter_expr = ie.value();

    mlir::Attribute loop_name = nullptr;
    // Check if the range is already cached.
    if (ie.index() < iteration_space.size()) {
      loop_name = iteration_space[ie.index()].cast<LoopAttr>().name();
      auto it = loop_range_cache.find(loop_name);
      if (it != loop_range_cache.end()) {
        new_domain.push_back(it->second.first);
        new_shape_dims.push_back(it->second.second);
        continue;
      }
    }

    CreateRange(op, iter_expr, op.domain(), op.shape(), inverse_mapping,
                normalized_loops, iteration_spaces, builder, new_domain,
                new_shape_dims);

    // Cache the new range.
    if (loop_name == nullptr) continue;
    loop_range_cache.try_emplace(loop_name, new_domain.back(),
                                 new_shape_dims.back());
  }

  // Permute dimensions to preserve sub-domains.
  llvm::SmallVector<int, 4> sub_domains = op.SubDomains();
  llvm::SmallVector<llvm::SmallVector<MappingExpr, 4>> partitioned_mapping(
      sub_domains.size());
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> partitioned_domain(
      sub_domains.size());
  llvm::SmallVector<llvm::SmallVector<DomainShapeDim, 4>, 4> partitioned_shape(
      sub_domains.size());

  for (int i = 0; i < new_domain.size(); ++i) {
    int sub_domain = 0;
    // Find the subdomain the loop belongs to.
    int min_domain_size = mapping.Dimension(i).MinDomainSize();
    if (min_domain_size > 0) {
      while (min_domain_size > sub_domains[sub_domain]) {
        min_domain_size -= sub_domains[sub_domain++];
      }
    }

    partitioned_mapping[sub_domain].push_back(mapping.Dimension(i));
    partitioned_domain[sub_domain].push_back(new_domain[i]);
    partitioned_shape[sub_domain].push_back(new_shape_dims[i]);
  }

  llvm::SmallVector<MappingExpr> flattened_mapping;
  flattened_mapping.reserve(mapping.size());
  for (const auto &sub_domain : partitioned_mapping) {
    llvm::append_range(flattened_mapping, sub_domain);
  }
  mapping =
      MappingAttr::get(op.getContext(), op.domain().size(), flattened_mapping);

  llvm::SmallVector<DomainShapeDim, 4> reordered_new_shape_dims;
  reordered_new_shape_dims.reserve(new_shape_dims.size());
  for (const auto &sub_domain : partitioned_shape) {
    for (const auto &dim : sub_domain) {
      reordered_new_shape_dims.emplace_back(
          dim.type(), dim.dependency_mapping().ResizeUseDomain(
                          reordered_new_shape_dims.size()));
    }
  }

  // Replace op by an operation with the domain changed.
  auto new_shape =
      DomainShapeAttr::get(op.getContext(), reordered_new_shape_dims);
  builder.setInsertionPoint(op);
  SairOp new_op = op.ReCreateWithNewDomain(partitioned_domain, new_shape,
                                           mapping.Inverse(), builder);
  for (auto [old_value, new_value] :
       llvm::zip(op->getResults(), new_op->getResults())) {
    // We do not normalize range operations, so we know that results are values.
    assert(old_value.getType().isa<ValueType>());
    UpdateValueUses(old_value,
                    {new_value, mapping.Resize(new_op.results_rank())});
  }
  op.erase();

  return mlir::success();
}

// Pass that rewrites operations domains so that each loop corresponds to a
// single dimension.
class NormalizeLoopsPass : public NormalizeLoopsPassBase<NormalizeLoopsPass> {
 public:
  void runOnFunction() override {
    mlir::OpBuilder builder(&getContext());
    getFunction().walk([&](SairProgramOp program) {
      auto iteration_spaces = getChildAnalysis<IterationSpaceAnalysis>(program);
      llvm::DenseMap<mlir::Attribute, std::pair<mlir::Value, DomainShapeDim>>
          loop_range_cache;
      program.walk([&](SairOp op) {
        // Do not normalize range operations.
        if (isa<RangeOp>(op.getOperation())) return;
        if (mlir::failed(NormalizeLoops(op, iteration_spaces, builder,
                                        loop_range_cache))) {
          signalPassFailure();
        }
      });
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateNormalizeLoopsPass() {
  return std::make_unique<NormalizeLoopsPass>();
}

}  // namespace sair
