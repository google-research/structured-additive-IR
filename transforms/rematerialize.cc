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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "loop_nest.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

// Creates Sair value types with the same elemental types as those in the given
// range, and with the given shape. Appends these new types to result.
void AdaptTypesToShape(mlir::TypeRange types, DomainShapeAttr shape,
                       llvm::SmallVectorImpl<mlir::Type> &result) {
  auto range = llvm::map_range(types, [shape](mlir::Type type) -> mlir::Type {
    return ValueType::get(shape.getContext(), shape,
                          type.cast<ValueType>().ElementType());
  });
  result.append(range.begin(), range.end());
}

// Creates a new mapping array by shifting all the accessed dimensions
// starting from `insert_pos` right by `num_dims`. This reflects `num_dims`
// dimensions being inserted at `insert_pos` into the domain.
mlir::ArrayAttr MappingsInsertDims(mlir::ArrayAttr mapping_array,
                                   size_t insert_pos, size_t num_dims) {
  llvm::SmallVector<mlir::Attribute, 4> new_mappings;
  new_mappings.reserve(mapping_array.size());
  for (auto mapping : mapping_array.getAsRange<MappingAttr>()) {
    new_mappings.push_back(mapping.ShiftRight(num_dims, insert_pos));
  }
  return mlir::ArrayAttr::get(new_mappings, mapping_array.getContext());
}

// Moves the body of the source operation to the target operation and inserts
// `num` block arguments of the given type at `pos`.
template <typename OpTy>
OpTy TakeBodyAdjustArguments(OpTy target, OpTy source, int pos, int num,
                             mlir::Type type) {
  target.body().takeBody(source.body());
  for (size_t i = 0; i < num; ++i) {
    target.body().front().insertArgument(pos, type);
  }
  return target;
}

// Given an array of Sair mappings, creates a new array containing the
// same mappings with use domain extended by `num_extra_dims`.
mlir::ArrayAttr MappingsExtendUseDomain(mlir::ArrayAttr mapping_array,
                                        size_t num_extra_dims) {
  llvm::SmallVector<mlir::Attribute, 4> new_mappings;
  new_mappings.reserve(mapping_array.size());
  for (auto mapping : mapping_array.getAsRange<MappingAttr>()) {
    new_mappings.push_back(
        mapping.ResizeUseDomain(mapping.UseDomainSize() + num_extra_dims));
  }
  return mlir::ArrayAttr::get(new_mappings, mapping_array.getContext());
}

// Creates a new sair.copy operation that is intended to replace `op`. Takes the
// additional domain dimensions, the updated result type and loop nest attribute
// supplied as arguments, extracts the value being copied and the mapping
// from `op`,
SairCopyOp RecreateOp(SairCopyOp op, mlir::TypeRange result_types,
                      mlir::ValueRange extra_domain,
                      mlir::ArrayAttr loop_nest_attr,
                      DomainShapeAttr domain_shape, mlir::OpBuilder &builder) {
  assert(result_types.size() == 1);
  auto domain = llvm::to_vector<8>(op.domain());
  llvm::append_range(domain, extra_domain);
  mlir::ArrayAttr mappings =
      MappingsExtendUseDomain(op.mapping_array(), extra_domain.size());
  return builder.create<SairCopyOp>(op.getLoc(), result_types[0], domain,
                                    mappings, op.value(), loop_nest_attr,
                                    op.memory_spaceAttr());
}

// Creates a new sair.map operation that is intended to replace `op`. Takes the
// additional domain dimensions, the updated result types and the loop nest
// attribute supplied as arguments; moves the body and copies the access
// mappings from `op`.
SairMapOp RecreateOp(SairMapOp op, mlir::TypeRange result_types,
                     mlir::ValueRange extra_domain,
                     mlir::ArrayAttr loop_nest_attr,
                     DomainShapeAttr domain_shape, mlir::OpBuilder &builder) {
  auto domain = llvm::to_vector<8>(op.domain());
  llvm::append_range(domain, extra_domain);
  mlir::ArrayAttr mappings =
      MappingsExtendUseDomain(op.mapping_array(), extra_domain.size());
  auto new_op = builder.create<SairMapOp>(
      op.getLoc(), result_types, domain, mappings, op.inputs(), domain_shape,
      loop_nest_attr, op.memory_spaceAttr());

  return TakeBodyAdjustArguments(new_op, op, op.domain().size(),
                                 extra_domain.size(), builder.getIndexType());
}

// Creates a new sair.map_reduce operation that is intended to replace `op`.
// Takes the additional parallel domain dimensions, the updated result types and
// the loop nest attribute supplied as arguments; moves the body and copies the
// reduction domain from `op`; takes the mappings from `op` and changes
// them to account for the inserted parallel dimensions.
SairMapReduceOp RecreateOp(SairMapReduceOp op, mlir::TypeRange result_types,
                           mlir::ValueRange extra_domain,
                           mlir::ArrayAttr loop_nest_attr,
                           DomainShapeAttr domain_shape,
                           mlir::OpBuilder &builder) {
  auto parallel_domain = llvm::to_vector<8>(op.parallel_domain());
  mlir::ArrayAttr mapping_attr = MappingsInsertDims(
      op.mapping_array(), parallel_domain.size(), extra_domain.size());
  llvm::append_range(parallel_domain, extra_domain);

  auto new_op = builder.create<SairMapReduceOp>(
      op.getLoc(), result_types, parallel_domain, op.reduction_domain(),
      mapping_attr, op.inits(), op.inputs(), domain_shape, loop_nest_attr,
      op.memory_spaceAttr());

  return TakeBodyAdjustArguments(new_op, op, op.parallel_domain().size(),
                                 extra_domain.size(), builder.getIndexType());
}

// Returns the operand range containing parallel domain dimensions.
mlir::Operation::operand_range ParallelDomain(SairOp op) {
  if (isa<SairCopyOp, SairMapOp>(op.getOperation())) {
    return op.domain();
  } else if (auto map_reduce = dyn_cast<SairMapReduceOp>(op.getOperation())) {
    return map_reduce.parallel_domain();
  }
  llvm_unreachable("unsupported sair op");
}

// Replaces `op` by the same op with actual dimensions in the domain instead of
// rematerialization tags. Effectively introduces as many trailing domain
// operands as `loops` and extends the shape of the result accordingly. The
// `main_loops` map should contain the loop bounds for all dimensions to
// rematerialize.
mlir::LogicalResult Rematerialize(ComputeOp op,
                                  const LoopFusionAnalysis &fusion_analysis) {
  MLIRContext *ctx = op.getContext();
  auto sair_op = cast<SairOp>(op.getOperation());

  // Keep the parallel domain and store the operand position to use for new
  // domain dimensions about to be inserted.
  auto parallel_domain = ParallelDomain(sair_op);
  size_t num_parallel_dims = parallel_domain.size();
  size_t position = num_parallel_dims;

  // Find positions of loop attributes that require rematerialization. These
  // will be used in the third sweep after we changed the attribute to refer to
  // the actual dimensions.
  llvm::SmallVector<int, 4> loops;
  auto loop_nest_array = llvm::to_vector<4>(op.LoopNestLoops());
  for (size_t i = 0, e = loop_nest_array.size(); i < e; ++i) {
    auto loop = loop_nest_array[i].cast<LoopAttr>();
    if (loop.iter().isa<MappingNoneExpr>()) loops.push_back(i);
  }
  size_t num_remat = loops.size();

  // Rebuild the loop nest attribute and populate the list of extra domain
  // dimensions.
  llvm::SmallVector<mlir::Value, 4> extra_domain;
  extra_domain.reserve(num_remat);

  llvm::SmallVector<MappingExpr, 4> iter_substitutions;
  iter_substitutions.reserve(sair_op.domain().size());
  for (int i = 0; i < num_parallel_dims; ++i) {
    iter_substitutions.push_back(MappingDimExpr::get(i, ctx));
  }
  for (int i = num_parallel_dims, e = sair_op.domain().size(); i < e; ++i) {
    iter_substitutions.push_back(MappingDimExpr::get(i + num_remat, ctx));
  }

  for (size_t i = 0, e = loop_nest_array.size(); i < e; ++i) {
    // If we are inserting domain dimensions in the middle of the dimension
    // list, update the indices of trailing dimensions.
    auto loop = loop_nest_array[i].cast<LoopAttr>();
    if (!loop.iter().isa<MappingNoneExpr>()) {
      if (loop.iter().MinDomainSize() > num_parallel_dims) {
        loop_nest_array[i] = LoopAttr::get(
            loop.name(), loop.iter().SubstituteDims(iter_substitutions), ctx);
      }
      continue;
    }

    // For each loop to rematerialize, add the range as the last domain argument
    // and update the loop nest attribute accordingly.
    auto &fusion_class = fusion_analysis.GetClass(loop.name());
    if (!fusion_class.iter_expr.isa<MappingDimExpr>()) {
      // TODO(b/172908223): support all loop nests expressions.
      return op.emitError() << "rematerialization only supports plain loops";
    }
    extra_domain.push_back(fusion_class.dimensions[0]);
    loop_nest_array[i] =
        LoopAttr::get(loop.name(), MappingDimExpr::get(position++, ctx), ctx);
  }

  // Parallel shape dimensions of the original op are kept as is.
  llvm::ArrayRef<DomainShapeDim> orig_dims = sair_op.shape().Dimensions();
  size_t num_orig_dims = orig_dims.size();
  auto domain_shape_dims =
      llvm::to_vector<8>(orig_dims.take_front(num_parallel_dims));
  domain_shape_dims.reserve(num_orig_dims + num_remat);

  // Traverse the rematerialized loops in the same order as before to match the
  // indices of the newly added dimensions and construct the corresponding
  // dimensions of the operation shape.
  for (auto en : llvm::enumerate(loops)) {
    auto loop = loop_nest_array[en.value()].cast<LoopAttr>();
    const LoopFusionClass &fusion_class = fusion_analysis.GetClass(loop.name());

    // Find positions of the loops the bounds of the current rematerialized loop
    // depend on and use them to construct the dependency mapping. Make sure to
    // take positions from the current op, as the dimensions that are depended
    // upon may be already present. Use the range type of the domain dimension
    // to construct the shape.
    auto dependencies_range = llvm::map_range(
        fusion_class.dependencies, [&](mlir::StringAttr dependee) {
          auto iter = llvm::find_if(loop_nest_array, [&](mlir::Attribute attr) {
            return attr.cast<LoopAttr>().name() == dependee;
          });
          assert(iter != loop_nest_array.end() &&
                 "rematerialized dimension depends on a dimension missing from "
                 "loop_nest attribute");
          return iter->cast<LoopAttr>().iter();
        });
    // We checked that the expression was a dimension above.
    int dimension = loop.iter().cast<MappingDimExpr>().dimension();
    auto dependency_mapping = MappingAttr::get(
        ctx, dimension, llvm::to_vector<4>(dependencies_range));
    domain_shape_dims.emplace_back(
        extra_domain[en.index()].getType().cast<RangeType>(),
        dependency_mapping);
  }

  // Non-parallel shape dimensions (trailing) of the original op need to be
  // shifted right to account for the inserted dimensions.
  for (const DomainShapeDim &dim : orig_dims.drop_front(num_parallel_dims)) {
    domain_shape_dims.emplace_back(
        dim.type(),
        dim.dependency_mapping().ShiftRight(num_remat, num_parallel_dims));
  }

  // Create the new domain shape and derive the result shape from it by removing
  // non-parallel dimensions.
  auto domain_shape = DomainShapeAttr::get(ctx, domain_shape_dims);
  DomainShapeAttr result_shape =
      domain_shape.Prefix(num_parallel_dims + num_remat);

  mlir::Operation *orig_operation = op.getOperation();
  llvm::SmallVector<mlir::Type, 2> new_types;
  new_types.reserve(orig_operation->getNumResults());
  AdaptTypesToShape(orig_operation->getResultTypes(), result_shape, new_types);

  OpBuilder builder(op.getContext());
  builder.setInsertionPoint(op);
  mlir::Operation *new_operation =
      llvm::TypeSwitch<mlir::Operation *, mlir::Operation *>(orig_operation)
          .Case<SairCopyOp, SairMapOp, SairMapReduceOp>([&](auto orig_op) {
            auto new_op = RecreateOp(orig_op, new_types, extra_domain,
                                     builder.getArrayAttr(loop_nest_array),
                                     domain_shape, builder);
            return new_op.getOperation();
          })
          .Default([](mlir::Operation *op) { return nullptr; });
  if (!new_operation) return mlir::failure();

  // Project out the rematerialized dimensions from all results.
  auto value_producer = cast<ValueProducerOp>(orig_operation);
  for (unsigned i = 0, e = new_types.size(); i < e; ++i) {
    mlir::Value orig_result = orig_operation->getResult(i);
    mlir::Value remat_result = new_operation->getResult(i);

    // Use the identity mapping here since defs and uses conserved their
    // mappings. In this case, the shape of the projection operation is equal to
    // the shape of its argument.
    auto mapping = builder.getArrayAttr(MappingAttr::GetIdentity(
        op.getContext(), num_parallel_dims + num_remat));
    DomainShapeAttr shape = remat_result.getType().cast<ValueType>().Shape();

    auto proj_op = builder.create<SairProjAnyOp>(
        op.getLoc(), orig_result.getType(), parallel_domain, extra_domain,
        mapping, remat_result, shape, /*memory_space=*/nullptr);
    if (llvm::Optional<int> memory_space = value_producer.GetMemorySpace(i)) {
      proj_op.SetMemorySpace(i, memory_space);
    }
    orig_result.replaceAllUsesWith(proj_op.getResult());
  }

  op.erase();

  return mlir::success();
}

// Rematerializes loops in all compute operations in the given program.
mlir::LogicalResult RematerializeInProgram(
    SairProgramOp op, const LoopFusionAnalysis &fusion_analysis) {
  auto status = op.walk([&](ComputeOp comp) -> mlir::WalkResult {
    if (!comp.loop_nest()) return mlir::success();
    if (llvm::all_of(comp.LoopNestLoops(), [](mlir::Attribute attr) {
          return !attr.cast<LoopAttr>().iter().isa<MappingNoneExpr>();
        })) {
      return mlir::success();
    }

    return Rematerialize(comp, fusion_analysis);
  });

  return mlir::failure(status.wasInterrupted());
}

// Pass that exercises rematerialization on Sair programs.
class RematerializePass : public RematerializePassBase<RematerializePass> {
 public:
  void runOnFunction() override {
    getFunction().walk([this](SairProgramOp program) {
      const LoopFusionAnalysis &fusion_analysis =
          getChildAnalysis<LoopFusionAnalysis>(program.getOperation());
      if (mlir::failed(RematerializeInProgram(program, fusion_analysis)))
        return signalPassFailure();
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateRematerializePass() {
  return std::make_unique<RematerializePass>();
}

}  // namespace sair
