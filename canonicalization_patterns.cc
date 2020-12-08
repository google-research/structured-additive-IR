#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sair_traits.h"
#include "sair_types.h"

namespace sair {

namespace {

// Extends an mapping with the identity mapping to match the given number
// of dimensions.
MappingAttr ExtendWithIdentity(MappingAttr old_mapping, int domain_size,
                               int new_mapping_size) {
  llvm::SmallVector<MappingExpr, 4> dimensions;
  dimensions.reserve(new_mapping_size);
  llvm::append_range(dimensions, old_mapping);
  for (int i = dimensions.size(); i < new_mapping_size; ++i) {
    dimensions.push_back(MappingDimExpr::get(i, old_mapping.getContext()));
  }
  return MappingAttr::get(old_mapping.getContext(), domain_size, dimensions);
}

// Redirects `use` to the `init` operand of `op` if `op` has an empty sequential
// domain. Returns true if any change was made.
bool SimplifyFbyOp(ValueOperand &use, SairFbyOp op) {
  if (!op.sequential_domain().empty()) return false;
  use.SetMapping(use.Mapping().Compose(op.Init().Mapping()));
  use.set_value(op.init());
  return true;
}

// Simplify a sair.proj_operand of a Sair operation, by bypassing projections
// with an empty projection domain and by flattening chains of projections.
// Returns true is any simplification was made.
template <typename ProjOp>
bool SimplifyProjOp(ValueOperand &use, ProjOp op,
                    mlir::PatternRewriter &rewriter) {
  if (op.projection_domain().empty()) {
    use.SetMapping(use.Mapping().Compose(op.Value().Mapping()));
    use.set_value(op.value());
    return true;
  }

  ProjOp prev_op = op.value().template getDefiningOp<ProjOp>();
  if (prev_op == nullptr) return false;

  llvm::SmallVector<mlir::Value, 4> projection_domain;
  projection_domain.reserve(op.projection_domain().size() +
                            prev_op.projection_domain().size());
  llvm::append_range(projection_domain, op.projection_domain());
  llvm::append_range(projection_domain, prev_op.projection_domain());

  llvm::SmallVector<DomainShapeDim, 4> shape_dims;
  shape_dims.reserve(op.shape().NumDimensions() +
                     prev_op.projection_domain().size());
  llvm::append_range(shape_dims, op.shape().Dimensions());
  llvm::ArrayRef<DomainShapeDim> prev_shape_dims = prev_op.shape().Dimensions();
  llvm::append_range(shape_dims, prev_op.shape().Dimensions().drop_front(
                                     prev_op.results_rank()));
  DomainShapeAttr shape = DomainShapeAttr::get(op.getContext(), shape_dims);

  MappingAttr new_mapping =
      ExtendWithIdentity(op.Value().Mapping(), shape_dims.size(),
                         prev_op.domain().size())
          .Compose(prev_op.Value().Mapping());
  mlir::ArrayAttr mapping_array = rewriter.getArrayAttr({new_mapping});

  rewriter.setInsertionPoint(op);
  ProjOp new_op = rewriter.create<ProjOp>(
      op.getLoc(), op.getType(), op.parallel_domain(), projection_domain,
      mapping_array, prev_op.value(), shape, op.memory_spaceAttr());
  use.set_value(new_op.result());

  return true;
}

// simplify the operands of the Sair operation:
// - Folds sair.proj_any and sair.proj_last operations with an empty projection
//   domain.
// - Folds sair.fby operations with an empty sequential domain.
// - Flatten chains of sair.proj_last and sair.proj_any operations.
class SimplifySairOperands : public RewritePattern {
 public:
  SimplifySairOperands()
      : RewritePattern(1, mlir::Pattern::MatchAnyOpTypeTag()) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, mlir::PatternRewriter &rewriter) const override {
    SairOp sair_op = dyn_cast<SairOp>(op);
    if (sair_op == nullptr) return mlir::failure();

    bool simplified = false;
    rewriter.startRootUpdate(op);
    for (ValueOperand operand : sair_op.ValueOperands()) {
      mlir::Operation *defining_op = operand.value().getDefiningOp();
      if (auto proj_last = dyn_cast<SairProjLastOp>(defining_op)) {
        simplified |= SimplifyProjOp(operand, proj_last, rewriter);
      }
      if (auto proj_any = dyn_cast<SairProjAnyOp>(defining_op)) {
        simplified |= SimplifyProjOp(operand, proj_any, rewriter);
      }
      if (auto fby = dyn_cast<SairFbyOp>(defining_op)) {
        simplified |= SimplifyFbyOp(operand, fby);
      }
    }
    if (simplified) {
      rewriter.finalizeRootUpdate(op);
    } else {
      rewriter.cancelRootUpdate(op);
    }

    return mlir::success(simplified);
  }
};

// Remove duplicate inputs and duplicate outputs of sair.map operations.
class DeduplicateMapInputsOutputs : public OpRewritePattern<SairMapOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SairMapOp op, mlir::PatternRewriter &rewriter) const override {
    int domain_size = op.domain().size();
    llvm::SmallVector<mlir::Value, 4> new_operands;
    llvm::SmallVector<mlir::Attribute, 4> new_mappings;

    llvm::SmallVector<mlir::Value, 4> old_results_to_keep;
    llvm::SmallVector<mlir::Value, 4> new_scalar_results;
    llvm::SmallVector<mlir::Type, 4> new_result_types;
    llvm::SmallVector<mlir::Attribute, 4> new_memory_spaces;

    for (ValueOperand operand : op.ValueOperands()) {
      mlir::Value argument =
          op.block().getArgument(domain_size + operand.position());

      // Deduplicate inputs.
      auto previous_operands =
          op.ValueOperands().take_front(operand.position());
      for (ValueOperand previous_operand : previous_operands) {
        if (operand.value() != previous_operand.value()) continue;
        if (operand.Mapping() != previous_operand.Mapping()) continue;
        mlir::Value previous_argument =
            op.block().getArgument(domain_size + previous_operand.position());
        argument.replaceAllUsesWith(previous_argument);
        break;
      }

      // Remove dead inputs.
      if (argument.use_empty()) {
        op.block().eraseArgument(domain_size + operand.position());
        continue;
      }

      new_operands.push_back(operand.value());
      new_mappings.push_back(operand.Mapping());
    }

    SairReturnOp return_op = cast<SairReturnOp>(op.block().getTerminator());
    for (int i = 0, e = op.getNumResults(); i < e; ++i) {
      mlir::Value scalar_value = return_op.getOperand(i);
      mlir::Value result = op.getResult(i);

      // Deduplicate results.
      for (int j = 0; j < i; ++j) {
        if (scalar_value != return_op.getOperand(j)) continue;
        if (op.GetMemorySpace(i) != op.GetMemorySpace(j)) continue;
        result.replaceAllUsesWith(op.getResult(j));
        break;
      }

      // Remove dead results.
      if (result.use_empty()) continue;

      old_results_to_keep.push_back(result);
      new_scalar_results.push_back(scalar_value);
      new_result_types.push_back(result.getType());
      mlir::Attribute memory_space =
          op.GetMemorySpace(i)
              .map([&](int value) -> mlir::Attribute {
                return rewriter.getI32IntegerAttr(value);
              })
              .getValueOr(rewriter.getUnitAttr());
      new_memory_spaces.push_back(memory_space);
    }

    // Create the new operation if necessary.
    if (new_operands.size() == op.ValueOperands().size() &&
        old_results_to_keep.size() == op.getNumResults()) {
      return mlir::failure();
    }

    rewriter.setInsertionPoint(return_op);
    rewriter.create<SairReturnOp>(return_op.getLoc(), new_scalar_results);
    rewriter.eraseOp(return_op);

    rewriter.setInsertionPoint(op);
    SairMapOp new_op = rewriter.create<SairMapOp>(
        op.getLoc(), new_result_types, op.domain(),
        rewriter.getArrayAttr(new_mappings), new_operands, op.shape(),
        op.loop_nestAttr(), rewriter.getArrayAttr(new_memory_spaces));
    new_op.body().takeBody(op.body());

    for (auto p : llvm::zip(old_results_to_keep, new_op.results())) {
      std::get<0>(p).replaceAllUsesWith(std::get<1>(p));
    }

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

// Finds a ValueOperand that wraps the same Value as the given OpOperand. Value
// operands immediately follow domain operands in Sair ops.
ValueOperand ValueOperandForOpOperand(OpOperand &operand) {
  auto owner = cast<SairOp>(operand.getOwner());
  ValueOperand value_operand =
      owner.ValueOperands()[operand.getOperandNumber() - owner.domain().size()];
  return value_operand;
}

// Remove a followed-by operation that depends on its own result, i.e.
//   %1 = sair.fby[...] %0(...) then[...] %1(d0, d1, ..., dn)
// and make its users use the init value instead.
class RemoveCyclicFby : public OpRewritePattern<SairFbyOp> {
 public:
  using OpRewritePattern<SairFbyOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SairFbyOp op, PatternRewriter &rewriter) const override {
    // Only apply to cycling followed-by with identity mappings.
    if (op.result() != op.value() || !op.Value().Mapping().IsIdentity())
      return mlir::failure();

    // Update the users. The list of uses contains all uses, including multiple
    // uses of the same value by the same operation.
    for (OpOperand &operand : op.result().getUses()) {
      if (operand.getOwner() == op) continue;

      ValueOperand value_operand = ValueOperandForOpOperand(operand);
      MappingAttr mapping =
          value_operand.Mapping().Compose(op.Init().Mapping());
      value_operand.set_value(op.init());
      value_operand.SetMapping(mapping);
    }

    assert(llvm::hasSingleElement(op.result().getUses()) &&
           op.result().getUses().begin()->getOwner() == op &&
           "expected only the self-reference to remain");
    op.erase();

    return mlir::success();
  }
};

// Given a bit vector indicating which dimensions are actually in use, populate
// `parallel_dimensions` and `other_dimensions` with values from
// `parallel_domain` and `other_domain`, respectively, that correspond to the
// domain dimensions that are in use. Assume `other_domain` immediately follows
// `parallel_domain` in a sequential dimension indexing scheme. Also set
// `mapping` to be an mapping mapping original (combined) domain
// dimensions to the new dimensions.
void RemoveUnusedDomainDimensions(
    mlir::MLIRContext *context, const llvm::SmallBitVector &used_dimensions,
    mlir::ValueRange parallel_domain, mlir::ValueRange other_domain,
    llvm::SmallVectorImpl<mlir::Value> &parallel_dimensions,
    llvm::SmallVectorImpl<mlir::Value> &other_dimensions,
    MappingAttr &mapping) {
  assert(used_dimensions.size() ==
         parallel_domain.size() + other_domain.size());

  int num_parallel_dims = parallel_domain.size();
  llvm::SmallVector<MappingExpr, 4> exprs;
  for (int dimension : used_dimensions.set_bits()) {
    if (dimension >= num_parallel_dims) {
      other_dimensions.push_back(other_domain[dimension - num_parallel_dims]);
    } else {
      parallel_dimensions.push_back(parallel_domain[dimension]);
    }
    exprs.push_back(MappingDimExpr::get(dimension, context));
  }

  mapping = MappingAttr::get(context, used_dimensions.size(), exprs);
}

// Update all uses of `value` to use `newValue` instead, and compose the access
// mapping with `mappingComponent`.
void UpdateValueUses(mlir::Value value, mlir::Value newValue,
                     MappingAttr mappingComponent) {
  for (OpOperand &operand : value.getUses()) {
    ValueOperand value_operand = ValueOperandForOpOperand(operand);
    value_operand.SetMapping(value_operand.Mapping().Compose(mappingComponent));
    value_operand.set_value(newValue);
  }
}

// Canonicalization pattern that drops unused dimensions from projection ops.
template <typename OpTy>
class RemoveUnreferencedDims : public OpRewritePattern<OpTy> {
  static_assert(llvm::is_one_of<OpTy, SairProjAnyOp, SairProjLastOp>::value,
                "pattern applies to projection ops only");

 public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      OpTy op, PatternRewriter &rewriter) const override {
    // Collect dimensions that appear in the mapping.
    llvm::SmallBitVector used_dimensions(op.domain().size());
    used_dimensions |= op.Value().Mapping().DependencyMask();
    if (used_dimensions.all()) return mlir::failure();

    // Prepare op components with unused dimensions removed.
    MappingAttr mapping;
    llvm::SmallVector<mlir::Value, 4> parallel_dimensions,
        projection_dimensions;
    RemoveUnusedDomainDimensions(op.getContext(), used_dimensions,
                                 op.parallel_domain(), op.projection_domain(),
                                 parallel_dimensions, projection_dimensions,
                                 mapping);

    // The result type has the rank equal to that of the parallel domain. Trim
    // the mapping accordingly.
    MappingAttr partial_mapping = mapping.Resize(parallel_dimensions.size());

    // Recreate the op because we may be changing the result type. Mappings
    // needs to be precomposed with the inverted mapping, i.e. the mapping from
    // the old iteration space to the new iteration space is applied first.
    auto new_op = rewriter.create<OpTy>(
        op.getLoc(),
        op.getType().template cast<ValueType>().AccessedType(partial_mapping),
        parallel_dimensions, projection_dimensions,
        rewriter.getArrayAttr(mapping.Inverse().Compose(op.Value().Mapping())),
        op.value(), op.shape().AccessedShape(mapping), op.memory_spaceAttr());
    new_op.setDialectAttrs(op->getDialectAttrs());

    // Replace the original op.
    UpdateValueUses(op, new_op, partial_mapping);
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

// Canonicalization pattern that drops unused dimensions from the followed-by
// operation.
template <>
class RemoveUnreferencedDims<SairFbyOp> : public OpRewritePattern<SairFbyOp> {
 public:
  using OpRewritePattern<SairFbyOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SairFbyOp op, PatternRewriter &rewriter) const override {
    // Collect dimensions that appear in mappings.
    llvm::SmallBitVector used_dimensions(op.domain().size());
    used_dimensions |= op.Value().Mapping().DependencyMask();
    used_dimensions |= op.Init().Mapping().DependencyMask();
    if (used_dimensions.all()) return mlir::failure();

    // Prepare op components with unused dimensions removed.
    MappingAttr direct_mapping;
    llvm::SmallVector<mlir::Value, 4> parallel_dimensions,
        sequential_dimensions;
    RemoveUnusedDomainDimensions(op.getContext(), used_dimensions,
                                 op.parallel_domain(), op.sequential_domain(),
                                 parallel_dimensions, sequential_dimensions,
                                 direct_mapping);

    MappingAttr inverted_mapping = direct_mapping.Inverse();

    // Recreate the op because we may be changing the result type. Mappings
    // needs to be precomposed with the inverted mapping, i.e. the mapping from
    // the old iteration space to the new iteration space is applied first.
    auto new_op = rewriter.create<SairFbyOp>(
        op.getLoc(),
        op.getType().cast<ValueType>().AccessedType(direct_mapping),
        parallel_dimensions, sequential_dimensions,
        rewriter.getArrayAttr({inverted_mapping.Compose(op.Init().Mapping()),
                               inverted_mapping.Compose(op.Value().Mapping())}),
        op.init(), op.value(), op.memory_spaceAttr());
    new_op.setDialectAttrs(op->getDialectAttrs());

    // Replace the original op.
    UpdateValueUses(op, new_op, direct_mapping);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

}  // end namespace

void SairCopyOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairExitOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairFromMemRefOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairFbyOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
  patterns.insert<RemoveCyclicFby, RemoveUnreferencedDims<SairFbyOp>>(context);
}

void SairFromScalarOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairMapReduceOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairProjAnyOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
  patterns.insert<RemoveUnreferencedDims<SairProjAnyOp>>(context);
}

void SairProjLastOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
  patterns.insert<RemoveUnreferencedDims<SairProjLastOp>>(context);
}

void SairDynRangeOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairStaticRangeOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairToMemRefOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairMapOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
  patterns.insert<DeduplicateMapInputsOutputs>(context);
}

}  // namespace sair
