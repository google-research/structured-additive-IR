#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
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
#include "utils.h"

namespace sair {

// Extends an access pattern with the identity pattern to match the given number
// of dimensions.
static AccessPatternAttr ExtendWithIdentity(AccessPatternAttr old_pattern,
                                            int domain_size,
                                            int new_pattern_size) {
  llvm::SmallVector<int, 4> dimensions;
  dimensions.reserve(new_pattern_size);
  appendRange(dimensions, old_pattern);
  for (int i = dimensions.size(); i < new_pattern_size; ++i) {
    dimensions.push_back(i);
  }
  return AccessPatternAttr::get(old_pattern.getContext(), domain_size,
                                dimensions);
}

// Redirects `use` to the `init` operand of `op` if `op` has an empty sequential
// domain. Returns true if any change was made.
static bool SimplifyFbyOp(ValueOperand &use, SairFbyOp op) {
  if (!op.sequential_domain().empty()) return false;
  use.SetAccessPattern(use.AccessPattern().Compose(op.Init().AccessPattern()));
  use.set_value(op.init());
  return true;
}

// Simplify a sair.proj_operand of a Sair operation, by bypassing projections
// with an empty projection domain and by flattening chains of projections.
// Returns true is any simplification was made.
template <typename ProjOp>
static bool SimplifyProjOp(ValueOperand &use, ProjOp op,
                           mlir::PatternRewriter &rewriter) {
  if (op.projection_domain().empty()) {
    use.SetAccessPattern(
        use.AccessPattern().Compose(op.Value().AccessPattern()));
    use.set_value(op.value());
    return true;
  }

  ProjOp prev_op = op.value().template getDefiningOp<ProjOp>();
  if (prev_op == nullptr) return false;

  llvm::SmallVector<mlir::Value, 4> projection_domain;
  projection_domain.reserve(op.projection_domain().size() +
                            prev_op.projection_domain().size());
  appendRange(projection_domain, op.projection_domain());
  appendRange(projection_domain, prev_op.projection_domain());

  llvm::SmallVector<DomainShapeDim, 4> shape_dims;
  shape_dims.reserve(op.shape().NumDimensions() +
                     prev_op.projection_domain().size());
  appendRange(shape_dims, op.shape().Dimensions());
  llvm::ArrayRef<DomainShapeDim> prev_shape_dims = prev_op.shape().Dimensions();
  appendRange(shape_dims,
              prev_op.shape().Dimensions().drop_front(prev_op.results_rank()));
  DomainShapeAttr shape = DomainShapeAttr::get(op.getContext(), shape_dims);

  AccessPatternAttr new_access_pattern =
      ExtendWithIdentity(op.Value().AccessPattern(), shape_dims.size(),
                         prev_op.domain().size())
          .Compose(prev_op.Value().AccessPattern());
  mlir::ArrayAttr access_pattern_array =
      rewriter.getArrayAttr({new_access_pattern});

  rewriter.setInsertionPoint(op);
  ProjOp new_op = rewriter.create<ProjOp>(
      op.getLoc(), op.getType(), op.parallel_domain(), projection_domain,
      access_pattern_array, prev_op.value(), shape, op.memory_spaceAttr());
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
    llvm::SmallVector<mlir::Attribute, 4> new_access_patterns;

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
        if (operand.AccessPattern() != previous_operand.AccessPattern())
          continue;
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
      new_access_patterns.push_back(operand.AccessPattern());
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
        rewriter.getArrayAttr(new_access_patterns), new_operands, op.shape(),
        op.loop_nestAttr(), rewriter.getArrayAttr(new_memory_spaces));
    new_op.body().takeBody(op.body());

    for (auto p : llvm::zip(old_results_to_keep, new_op.results())) {
      std::get<0>(p).replaceAllUsesWith(std::get<1>(p));
    }

    rewriter.eraseOp(op);

    return mlir::success();
  }
};

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
}

void SairProjLastOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  patterns.insert<SimplifySairOperands>();
}

void SairRangeOp::getCanonicalizationPatterns(
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
