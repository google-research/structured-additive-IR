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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "sequence.h"
#include "util.h"

namespace sair {

IterationSpace::IterationSpace(llvm::SmallVector<mlir::StringAttr> loop_names,
                               MappingAttr domain_to_loops,
                               bool fully_specified)
    : loop_names_(std::move(loop_names)), fully_specified_(fully_specified) {
  assert(loop_names_.size() == domain_to_loops.size());
  mapping_ = domain_to_loops.Inverse().MakeSurjective().Inverse();
}

int IterationSpace::NumCommonLoops(const IterationSpace &other) const {
  return NumCommonLoops(other.loop_names());
}

int IterationSpace::NumCommonLoops(
    llvm::ArrayRef<mlir::StringAttr> other) const {
  auto it_pair = std::mismatch(loop_names().begin(), loop_names().end(),
                               other.begin(), other.end());
  return std::distance(loop_names().begin(), it_pair.first);
}

// Infers the iteration space for the current operation from iteration space of
// the given operand. Trims inner loops so than only loops iterating on
// dimensions mapped by the mapping remain. The resulting loop nest may
// not cover all dimensions of the current operation.
static IterationSpace InferIterationSpace(
    const IterationSpace &operand_iteration_space, ValueOperand &operand) {
  MappingAttr mapping = operand.Mapping();

  llvm::SmallVector<mlir::StringAttr> loop_names;
  for (auto [name, iter] :
       llvm::zip(operand_iteration_space.loop_names(),
                 operand_iteration_space.MappingToLoops())) {
    if (iter.MinDomainSize() > mapping.size()) break;
    loop_names.push_back(name);
  }

  MappingAttr domain_to_loops = mapping.Compose(
      operand_iteration_space.MappingToLoops().Resize(loop_names.size()));

  // If the iteration space is infered from loop-carried dimensions, trim inner
  // parallel dimensions as inner parallel dimension open at the end of the
  // previous iteration along loop-carried  dimension may not be open at the
  // beginning of the current iteration.
  if (operand.AllowUseBeforeDef()) {
    llvm::SmallBitVector carrying_dims = operand.CarryingDims();
    int domain_size = mapping.UseDomainSize();
    int new_size = loop_names.size();
    for (; new_size > 0; --new_size) {
      MappingExpr expr = domain_to_loops.Dimension(new_size - 1);
      if (expr.DependencyMask(domain_size).anyCommon(carrying_dims)) {
        break;
      }
    }
    loop_names.resize(new_size);
    domain_to_loops = domain_to_loops.Resize(new_size);
  }

  return IterationSpace(std::move(loop_names), domain_to_loops,
                        operand_iteration_space.fully_specified());
}

IterationSpaceAnalysis::IterationSpaceAnalysis(SairProgramOp program_op) {
  if (program_op == nullptr) return;
  for (mlir::Operation &op : program_op.body().front()) {
    ComputeIterationSpace(&op);
  }
}

const IterationSpace &IterationSpaceAnalysis::Get(SairOp op) const {
  return iteration_space_.find(op.getOperation())->second;
}

const IterationSpace &IterationSpaceAnalysis::ComputeIterationSpace(
    mlir::Operation *operation) {
  if (auto it = iteration_space_.find(operation);
      it != iteration_space_.end()) {
    return it->second;
  }

  mlir::MLIRContext *context = operation->getContext();
  SairOp sair_op = cast<SairOp>(operation);
  int domain_size = sair_op.domain().size();

  // Handle ComputeOp case.
  if (auto compute_op = dyn_cast<ComputeOp>(operation)) {
    int num_loops = compute_op.LoopNestLoops().size();
    llvm::SmallVector<MappingExpr> exprs;
    exprs.reserve(num_loops);
    llvm::SmallVector<mlir::StringAttr> loop_names;
    loop_names.reserve(num_loops);

    for (mlir::Attribute attr : compute_op.LoopNestLoops()) {
      LoopAttr loop = attr.cast<LoopAttr>();
      loop_names.push_back(loop.name());
      exprs.push_back(loop.iter());
    }

    bool fully_specified = compute_op.loop_nest().hasValue();
    auto mapping = MappingAttr::get(context, domain_size, exprs);
    return iteration_space_
        .try_emplace(operation, loop_names, mapping, fully_specified)
        .first->second;
  }

  // Temporarily set an empty iteration space to avoid infinite recursion.
  auto empty_mapping = MappingAttr::get(context, domain_size, {});
  llvm::SmallVector<mlir::StringAttr> empty_names;
  auto it =
      iteration_space_.try_emplace(operation, empty_names, empty_mapping, false)
          .first;

  auto infer_iteration_space = dyn_cast<InferIterationSpaceOp>(operation);
  if (infer_iteration_space == nullptr) return it->second;

  int operand_pos = infer_iteration_space.infer_iteration_space_operand();
  ValueOperand operand = sair_op.ValueOperands()[operand_pos];
  mlir::Operation *defining_op = operand.value().getDefiningOp();
  const IterationSpace &parent_iteration_space =
      ComputeIterationSpace(defining_op);
  it = iteration_space_.find(operation);
  it->second = InferIterationSpace(parent_iteration_space, operand);
  return it->second;
}

MappingAttr IterationSpaceAnalysis::TranslateMapping(
    SairOp from, SairOp to, MappingAttr mapping) const {
  MappingAttr result = TryTranslateMapping(from, to, mapping);
  assert(result != nullptr);
  return result;
}

MappingAttr IterationSpaceAnalysis::TryTranslateMapping(
    SairOp from, SairOp to, MappingAttr mapping) const {
  const IterationSpace &from_space = Get(from);
  const IterationSpace &to_space = Get(to);
  MappingAttr space_mapping = from_space.mapping()
                                  .Inverse()
                                  .Compose(mapping)
                                  .Compose(to_space.mapping())
                                  .Canonicalize();

  int num_common_loops = from_space.NumCommonLoops(to_space);
  auto common_loops_mapping = MappingAttr::GetIdentity(
      mapping.getContext(), num_common_loops, from_space.mapping().size());
  MappingAttr loops_mapping =
      common_loops_mapping.Resize(to_space.mapping().size());
  return space_mapping.Unify(loops_mapping);
}

// Analysis that keeps track of dependencies between loops.
class LoopNestConstraintsAnalysis {
 public:
  // Constraints for using a value.
  struct Constraints {
    // Loops open at producers.
    llvm::SetVector<mlir::Attribute> open_loops;
    // Loops closed at producers.
    llvm::SetVector<mlir::Attribute> closed_loops;
    // Dimensions of the value produced by closed loops.
    llvm::SmallBitVector closed_dimensions;

    explicit Constraints(int domain_size) : closed_dimensions(domain_size) {}
  };

  explicit LoopNestConstraintsAnalysis(
      SairProgramOp program, const IterationSpaceAnalysis &loop_nests) {
    for (mlir::Operation &operation : program.body().front()) {
      ComputeConstraints(&operation, loop_nests);
    }
  }

  // Returns the constraints for using the given value.
  const Constraints &GetConstraints(mlir::Value value) const {
    mlir::Operation *defining_op = value.getDefiningOp();
    // The sair.program verifier ensures that operation operands are defined
    // within the same block. This is done before calling the loop nest
    // constraints analysis.
    assert(defining_op != nullptr);
    return constraints_.find(defining_op)->second;
  }

 private:
  // Compute constraints for using values produced by the given operation.
  const Constraints &ComputeConstraints(
      mlir::Operation *operation,
      const IterationSpaceAnalysis &iteration_spaces) {
    if (auto it = constraints_.find(operation); it != constraints_.end()) {
      return it->second;
    }

    SairOp op = cast<SairOp>(operation);
    int domain_size = op.domain().size();
    Constraints constraints(domain_size);

    auto inherit_constraints = [&](mlir::Value value, MappingAttr mapping,
                                   bool loop_carried = false) {
      const Constraints &parent_constraint =
          ComputeConstraints(value.getDefiningOp(), iteration_spaces);
      for (int closed_dim : parent_constraint.closed_dimensions.set_bits()) {
        if (closed_dim >= mapping.size()) break;
        mapping.Dimension(closed_dim)
            .SetDependenciesInMask(constraints.closed_dimensions);
      }
      if (loop_carried) return;
      constraints.open_loops.set_union(parent_constraint.open_loops);
      constraints.closed_loops.set_union(parent_constraint.closed_loops);
    };

    if (!isa<ComputeOp>(operation)) {
      // Store empty constraints to avoid infinite recursion.
      constraints_.try_emplace(operation, domain_size);
      for (int i = 0, e = domain_size; i < e; ++i) {
        MappingAttr mapping = op.shape().Dimension(i).dependency_mapping();
        inherit_constraints(op.domain()[i], mapping);
      }
      for (ValueOperand operand : op.ValueOperands()) {
        inherit_constraints(operand.value(), operand.Mapping(),
                            operand.AllowUseBeforeDef());
      }
    }

    const IterationSpace &iteration_space = iteration_spaces.Get(operation);
    llvm::SmallBitVector closed_dims = op.ResultsDimDependencies();
    bool closed_dims_seen = false;
    for (int i = 0, e = iteration_space.num_loops(); i < e; ++i) {
      constraints.open_loops.insert(iteration_space.loop_names()[i]);
      MappingExpr expr = iteration_space.mapping().Dimension(i);
      llvm::SmallBitVector iter_dims = expr.DependencyMask(op.domain().size());
      if (iter_dims.anyCommon(closed_dims)) {
        constraints.closed_loops.insert(iteration_space.loop_names()[i]);
        closed_dims_seen = true;
      }
      if (closed_dims_seen) {
        constraints.closed_dimensions |= iter_dims;
      }
    }

    constraints_.erase(operation);
    return constraints_.insert({operation, std::move(constraints)})
        .first->second;
  }

  llvm::DenseMap<mlir::Operation *, Constraints> constraints_;
};

mlir::LogicalResult VerifyLoopNestWellFormed(
    ComputeOp op, llvm::ArrayRef<mlir::Attribute> loop_nest) {
  llvm::SmallVector<MappingExpr> iter_exprs;
  iter_exprs.reserve(loop_nest.size());
  auto sair_op = cast<SairOp>(op.getOperation());

  int domain_size = sair_op.domain().size();
  // Bitfield that keeps track of which dimensions are implemented by loops.
  for (int i = 0, e = loop_nest.size(); i < e; ++i) {
    LoopAttr loop = loop_nest[i].dyn_cast<LoopAttr>();
    // Delegate checking attributes type to other verifiers.
    if (loop == nullptr) return mlir::success();

    // Ensure that symbols are unique in the loop nest.
    for (int j = 0; j < i; ++j) {
      if (loop.name() == loop_nest[j].cast<LoopAttr>().name()) {
        return op.emitError()
               << "name " << loop.name() << " used twice in the same loop nest";
      }
    }

    int min_domain_size = loop.iter().MinDomainSize();
    if (loop.iter().MinDomainSize() > domain_size) {
      return op.emitError() << "dimension 'd" << min_domain_size - 1 << "' "
                            << "is out of range of the domain";
    }

    if (loop.iter().HasUnknownExprs()) {
      return op.emitError() << "loop iterators cannot contain `?` expressions";
    }

    iter_exprs.push_back(loop.iter());
  }

  mlir::MLIRContext *context = op.getContext();
  auto mapping = MappingAttr::getChecked(context, domain_size, iter_exprs);
  if (mapping == nullptr) {
    return op.emitError() << "incompatible loop iterators";
  }

  if (mapping.Inverse().HasNoneExprs()) {
    return op.emitError() << "not all dimensions are covered by the loop nest";
  }

  return VerifyMappingShape(AttrLocation(op.getLoc(), "loop_nest"), mapping,
                            sair_op.shape());
}

namespace {

// Helper class to track open loops and verify the loop structure forms a tree.
class LoopNestState {
 public:
  // Updates the list of loops currently open and closed to accomodate the
  // loop nest `loop_nest` of `op`. Returns a failure if the loop structure does
  // not form a tree or if a loop is used before its range is defined.
  mlir::LogicalResult Update(SairOp op,
                             mlir::ArrayRef<mlir::Attribute> loop_nest) {
    // Find the number of common loops.
    int common_prefix_size = 0;
    for (int e = std::min(loop_nest.size(), open_loops_.size());
         common_prefix_size < e; ++common_prefix_size) {
      LoopAttr loop = loop_nest[common_prefix_size].cast<LoopAttr>();
      if (loop.name() != open_loops_[common_prefix_size]) break;
    }

    // Reset the current fusion prefix to the number of common loops.
    if (mlir::failed(CloseLoops(common_prefix_size))) return mlir::failure();

    // Add remaining loops to the current fusion prefix.
    for (mlir::Attribute attribute : loop_nest.drop_front(common_prefix_size)) {
      LoopAttr loop = attribute.cast<LoopAttr>();

      if (closed_loops_.count(loop.name()) > 0) {
        return op.emitError()
               << "occurrences of loop " << loop.name()
               << " must be contiguous and nested in the same loops";
      }

      open_loops_.push_back(loop.name());
    }
    return mlir::success();
  }

  // Mark loops as closed, starting from the innermost`, until only
  // `num_remaining_loops` are left open.
  mlir::LogicalResult CloseLoops(int num_remaining_loops = 0) {
    while (open_loops_.size() > num_remaining_loops) {
      closed_loops_.insert(open_loops_.pop_back_val());
    }
    return mlir::success();
  };

  // Verifies that the given loops have been open before.
  mlir::LogicalResult VerifyLoopsOpen(
      const llvm::SetVector<mlir::Attribute> &loops, mlir::Location loc) const {
    for (mlir::Attribute loop : loops) {
      if (llvm::count(open_loops_, loop) == 0 &&
          !closed_loops_.contains(loop)) {
        return mlir::emitError(loc)
               << "loop " << loop
               << " must be open at or before this operation";
      }
    }
    return mlir::success();
  }

 private:
  llvm::SmallVector<mlir::StringAttr> open_loops_;
  llvm::DenseSet<mlir::Attribute> closed_loops_;
};

}  // namespace

// Verifies that dimensions that must be open before executing `op` are indeed
// open in the loop nest state.
static mlir::LogicalResult VerifyLoopsOpen(
    SairOp op, const LoopNestState &loop_nest_state,
    const LoopNestConstraintsAnalysis &loop_constaints_analysis) {
  for (mlir::Value dimension : op.domain()) {
    const auto &constraints =
        loop_constaints_analysis.GetConstraints(dimension);
    if (mlir::failed(loop_nest_state.VerifyLoopsOpen(constraints.open_loops,
                                                     op.getLoc()))) {
      return mlir::failure();
    }
  }
  for (ValueOperand operand : op.ValueOperands()) {
    const auto &constraints =
        loop_constaints_analysis.GetConstraints(operand.value());
    if (mlir::failed(loop_nest_state.VerifyLoopsOpen(constraints.open_loops,
                                                     op.getLoc()))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

// Verifies that the loop nest `op_loop_nest` of `op` is compatible with the
// constraints imposed by the operand `dependency` of `op`.
// * `dim_dependencies`: dimensions of `op` that cannot be part of the loop-nest
//    producing `dependency`.
// * `carrying_dims`: if `dependency` is a loop-carried operand, lists
//    dimensions carrying the value of `dependency` across iterations.
static mlir::LogicalResult VerifyDependency(
    SairOp op, const IterationSpace &op_loop_nest, ValueAccess dependency,
    const llvm::SmallBitVector &dim_dependencies,
    const llvm::SmallBitVector &carrying_dims,
    const IterationSpaceAnalysis &iteration_space_analysis,
    const LoopNestConstraintsAnalysis &loop_constraints_analysis) {
  auto dependency_op = cast<SairOp>(dependency.value.getDefiningOp());

  MappingAttr domain_mapping =
      dependency.mapping.Resize(dependency_op.domain().size())
          .ResizeUseDomain(op.domain().size());
  if (iteration_space_analysis.TryTranslateMapping(op, dependency_op,
                                                   domain_mapping) == nullptr) {
    return (op.emitError() << "loop nest violates a data dependency")
               .attachNote(dependency.value.getLoc())
           << "dependency from this operation";
  }

  const LoopNestConstraintsAnalysis::Constraints &constraints =
      loop_constraints_analysis.GetConstraints(dependency.value);
  for (int i = 0, e = op_loop_nest.num_loops(); i < e; ++i) {
    mlir::StringAttr name = op_loop_nest.loop_names()[i];
    if (constraints.closed_loops.contains(name)) {
      return op.emitError()
             << "loop " << name << " must be closed before this operation";
    }

    if (!constraints.open_loops.contains(name)) continue;
    MappingExpr expr = op_loop_nest.mapping().Dimension(i);
    llvm::SmallBitVector iter_dims = expr.DependencyMask(op.domain().size());
    if (!dim_dependencies.anyCommon(iter_dims)) continue;

    return (dependency.value.getDefiningOp()->emitError()
            << "operation cannot be nested in loop " << name)
               .attachNote(op.getLoc())
           << "because of this operation";
  }

  for (int dep_dimension : constraints.closed_dimensions.set_bits()) {
    int domain_size = dependency.mapping.UseDomainSize();
    if (dep_dimension >= dependency.mapping.size()) break;
    llvm::SmallBitVector mapped_dims =
        dependency.mapping.Dimension(dep_dimension).DependencyMask(domain_size);
    if (carrying_dims.anyCommon(mapped_dims)) {
      int dim = (carrying_dims & mapped_dims).find_first();
      return op.emitError()
             << "cannot take the previous value of the operand along 'd" << dim
             << "' because of the operand loop nest";
    }
  }

  return mlir::success();
}

// Verifies that the loop nest of `op` is compatible with the constraints
// imposed by its dependencies.
static mlir::LogicalResult VerifyDependencies(
    SairOp op, const IterationSpaceAnalysis &iteration_space_analysis,
    LoopNestConstraintsAnalysis &loop_constaints_analysis) {
  const IterationSpace &loop_nest = iteration_space_analysis.Get(op);

  int domain_size = op.domain().size();
  for (int i = 0; i < domain_size; ++i) {
    llvm::SmallBitVector dim_dependencies(op.domain().size());
    llvm::SmallBitVector carrying_dims(op.domain().size());
    dim_dependencies.set(i);
    MappingAttr mapping = op.shape().Dimensions()[i].dependency_mapping();
    if (mlir::failed(VerifyDependency(op, loop_nest, {op.domain()[i], mapping},
                                      dim_dependencies, carrying_dims,
                                      iteration_space_analysis,
                                      loop_constaints_analysis))) {
      return mlir::failure();
    }
  }

  for (ValueOperand operand : op.ValueOperands()) {
    if (mlir::failed(VerifyDependency(
            op, loop_nest, operand.Get(), operand.DependingDims(),
            operand.CarryingDims(), iteration_space_analysis,
            loop_constaints_analysis))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Verifies that it is possible to compute the range of loops and that the
// range is defined before it is used.
static mlir::LogicalResult VerifyLoopRanges(
    ComputeOp op, llvm::ArrayRef<mlir::Attribute> loop_nest,
    const LoopFusionAnalysis &fusion_analysis,
    const SequenceAnalysis &sequence_analysis) {
  for (mlir::Attribute attr : loop_nest) {
    LoopAttr loop = attr.cast<LoopAttr>();
    const LoopFusionClass &fusion_class = fusion_analysis.GetClass(loop.name());
    for (const auto &dimension : fusion_class.domain()) {
      if (sequence_analysis.IsBefore(op,
                                     dimension.value.getDefiningOp<SairOp>())) {
        return (op.emitError()
                << "rematerialized loop " << loop.name()
                << " indirectly uses the range before it is defined")
                   .attachNote(dimension.value.getLoc())
               << "range defined here";
      }
    }
  }

  return mlir::success();
}

// Ensure that each loop only iterate along a single sub-domain.
static mlir::LogicalResult VerifySubDomains(
    SairOp op, const IterationSpace &iteration_space) {
  llvm::SmallVector<int> sub_domains = op.SubDomains();
  assert(!sub_domains.empty() || iteration_space.num_loops() == 0);

  for (int i = 0, e = iteration_space.num_loops(); i < e; ++i) {
    MappingExpr expr = iteration_space.mapping().Dimension(i);
    llvm::SmallBitVector dimensions = expr.DependencyMask(op.domain().size());
    if (!dimensions.any()) continue;

    // Compute the sub-domain the loop belongs to. If the iterator is not fully
    // specified, then reaterializing dimensions will be added to the parallel
    // sub-domain (sub-domain 0) and so all dimensions must belong to the
    // parallel sub-domain.
    int sub_domain = 0;
    int min_dim_index = 0;
    int max_dim_index = sub_domains[0];
    if (!expr.HasNoneExprs()) {
      int first = dimensions.find_first();
      while (first >= max_dim_index) {
        min_dim_index = max_dim_index;
        max_dim_index += sub_domains[sub_domain++];
      }
    }

    // Check that all dimensions referenced by the iterator are in the
    // sub-domain.
    if (dimensions.find_first() < min_dim_index ||
        dimensions.find_last() >= max_dim_index) {
      return op.emitError() << "loop " << iteration_space.loop_names()[i]
                            << " crosses sub-domains boundaries";
    }
  }
  return mlir::success();
}

mlir::LogicalResult VerifyLoopNests(
    SairProgramOp program, const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    const SequenceAnalysis &sequence_analysis) {
  // Verify that the loop structure forms a tree, loops are open when they need
  // to and loop ranges are well defined.
  LoopNestState loop_nest_state;
  LoopNestConstraintsAnalysis loop_constraints_analysis(program,
                                                        iteration_spaces);
  for (ComputeOp compute_op : sequence_analysis.Ops()) {
    auto sair_op = cast<SairOp>(compute_op.getOperation());
    if (compute_op.loop_nest().hasValue()) {
      if (mlir::failed(
              loop_nest_state.Update(sair_op, compute_op.LoopNestLoops()))) {
        return mlir::failure();
      }
      if (mlir::failed(VerifyLoopRanges(compute_op, compute_op.LoopNestLoops(),
                                        fusion_analysis, sequence_analysis))) {
        return mlir::failure();
      }
    } else if (mlir::failed(loop_nest_state.CloseLoops())) {
      return mlir::failure();
    }
    if (mlir::failed(VerifyLoopsOpen(sair_op, loop_nest_state,
                                     loop_constraints_analysis))) {
      return mlir::failure();
    }
  }
  if (mlir::failed(loop_nest_state.CloseLoops())) return mlir::failure();

  // Verify dependencies.
  mlir::WalkResult result = program.walk([&](SairOp op) -> mlir::WalkResult {
    if (mlir::failed(VerifySubDomains(op, iteration_spaces.Get(op)))) {
      return mlir::failure();
    }
    return VerifyDependencies(op, iteration_spaces, loop_constraints_analysis);
  });
  if (result.wasInterrupted()) return mlir::failure();

  return mlir::success();
}

LoopFusionAnalysis::LoopFusionAnalysis(
    mlir::Operation *operation, const SequenceAnalysis *sequence_analysis)
    : context_(operation->getContext()) {
  SairProgramOp program_op = dyn_cast<SairProgramOp>(operation);
  if (program_op == nullptr) return;
  mlir::LogicalResult status =
      Init(program_op, sequence_analysis ? *sequence_analysis
                                         : SequenceAnalysis(program_op));
  assert(mlir::succeeded(status));
  (void)status;
}

std::optional<LoopFusionAnalysis> LoopFusionAnalysis::Create(
    SairProgramOp program_op, const SequenceAnalysis &sequence_analysis) {
  LoopFusionAnalysis analysis(program_op->getContext());
  if (mlir::failed(analysis.Init(program_op, sequence_analysis))) {
    return std::nullopt;
  }
  return analysis;
}

mlir::LogicalResult LoopFusionAnalysis::Init(
    SairProgramOp program_op, const SequenceAnalysis &sequence_analysis) {
  llvm::SmallVector<ComputeOp> work_list;
  program_op.walk([&](ComputeOp op) {
    auto sair_op = cast<SairOp>(op.getOperation());
    int domain_size = sair_op.domain().size();
    auto none_expr = MappingNoneExpr::get(context_);
    op_domain_mappings_[op.getOperation()].resize(domain_size, none_expr);
    if (!op.loop_nest().hasValue()) return;
    work_list.push_back(op);
  });

  // Handle loops by nesting levels. This ensures that we visited all occurences
  // of a loop before moving to inner loops.
  for (int level = 0; !work_list.empty(); ++level) {
    for (int i = 0; i < work_list.size(); ++i) {
      ComputeOp op = work_list[i];

      // Remove operations from the list once their loop-nest is handled.
      if (op.LoopNestLoops().size() <= level) {
        work_list[i] = work_list.back();
        work_list.pop_back();
        --i;
        continue;
      }
      if (mlir::failed(RegisterLoop(op, level, sequence_analysis))) {
        return mlir::failure();
      }
    }
  }

  // Ensure that all iterators are fully specified.
  for (auto &[name, fusion_class] : fusion_classes_) {
    if (fusion_class.mapping().HasNoneExprs()) {
      return fusion_class.EmitError() << "iterator is not fully specified";
    }
  }

  // Trim dependencies in each fusion class.
  for (auto &[name, fusion_class] : fusion_classes_) {
    DomainShapeDim loop_shape = fusion_class.NestedShape().Dimensions().back();
    int max_dependency = loop_shape.DependencyMask().find_last();
    for (const auto &dimension : fusion_class.domain()) {
      max_dependency = std::max(max_dependency,
                                dimension.mapping.DependencyMask().find_last());
    }
    fusion_class.TrimDependencies(max_dependency + 1);
  }

  return mlir::success();
}

mlir::LogicalResult LoopFusionAnalysis::RegisterLoop(
    ComputeOp op, int loop_pos, const SequenceAnalysis &sequence_analysis) {
  auto sair_op = cast<SairOp>(op.getOperation());
  int domain_size = sair_op.domain().size();

  // Retrieve outer loops information.
  llvm::SmallVector<mlir::StringAttr> loop_names;
  llvm::SmallVector<MappingExpr> iter_exprs;
  for (int i = 0; i < loop_pos; ++i) {
    LoopAttr loop = op.LoopNestLoops()[i].cast<LoopAttr>();
    loop_names.push_back(loop.name());
    iter_exprs.push_back(loop.iter());
  }
  LoopNest loop_nest = GetLoopNest(loop_names);
  auto loop_nest_mapping =
      MappingAttr::get(op.getContext(), domain_size, iter_exprs);

  LoopAttr loop = op.LoopNestLoops()[loop_pos].cast<LoopAttr>();
  auto [it, was_inserted] =
      fusion_classes_.try_emplace(loop.name(), loop.name(), op, loop_nest);
  LoopFusionClass &fusion_class = it->second;
  if (!was_inserted) {
    fusion_class.AddUse(op, sequence_analysis);
  }

  auto mapping = MappingAttr::get(context_, domain_size, {loop.iter()});
  return fusion_class.UnifyMapping(op.getLoc(), loop_nest_mapping, mapping,
                                   sair_op.DomainWithDependencies());
}

LoopNest LoopFusionAnalysis::GetLoopNest(
    llvm::ArrayRef<mlir::StringAttr> loop_names) const {
  if (loop_names.empty()) return LoopNest(context_);
  return LoopNest(&GetClass(loop_names.back()));
}

mlir::StringAttr LoopFusionAnalysis::GetFreshLoopName() {
  llvm::SmallString<10> name("loop_");
  int original_size = name.size();
  mlir::StringAttr attr;
  do {
    name.resize(original_size);
    name += std::to_string(next_loop_id_++);
    attr = mlir::StringAttr::get(context_, name);
  } while (fusion_classes_.count(attr) > 0);
  return attr;
}


LoopFusionClass::LoopFusionClass(mlir::StringAttr name, ComputeOp op,
                                 const LoopNest &loop_nest)
    : MappedDomain(op.getLoc(), "loop", name, loop_nest), last_op_(op) {
  num_dependencies_ = loop_nest.size();
  AddNonePrefixToMapping(1);
}

void LoopFusionClass::AddUse(ComputeOp op,
                             const SequenceAnalysis &sequence_analysis) {
  if (sequence_analysis.IsBefore(last_op_, op)) last_op_ = op;
}

void LoopFusionClass::TrimDependencies(int num_dependencies) {
  num_dependencies_ = num_dependencies;
  for (auto &dimension : domain_) {
    dimension.mapping = dimension.mapping.ResizeUseDomain(num_dependencies);
  }
}

ProgramPoint LoopFusionClass::EndPoint() const {
  return ProgramPoint(last_op_, Direction::kAfter, loop_nest());
}

int LoopNest::size() const {
  if (empty()) return 0;
  return fusion_class_->loop_nest().size() + 1;
}

llvm::ArrayRef<ValueAccess> LoopNest::domain() const {
  if (empty()) return {};
  return fusion_class_->domain();
}

MappingAttr LoopNest::DomainToLoops() const {
  if (empty()) return MappingAttr::get(context_, 0, {});
  return fusion_class_->NestedMapping();
}

llvm::SmallVector<mlir::StringAttr> LoopNest::LoopNames() const {
  if (empty()) return {};
  llvm::SmallVector<mlir::StringAttr> loop_names;
  loop_names.reserve(size());
  llvm::append_range(loop_names, fusion_class_->loop_nest());
  loop_names.push_back(fusion_class_->name());
  return loop_names;
}

DomainShapeAttr LoopNest::Shape() const {
  if (empty()) return DomainShapeAttr::get(context_);
  return fusion_class_->NestedShape();
}

DomainShapeAttr LoopNest::NormalizedShape() const {
  llvm::SmallVector<DomainShapeDim> normalized_shape_dims;
  DomainShapeAttr shape = Shape();
  normalized_shape_dims.reserve(shape.NumDimensions());
  for (const DomainShapeDim &dim : shape.Dimensions()) {
    int num_dependencies = dim.dependency_mapping().MinDomainSize();
    if (num_dependencies == 0) {
      normalized_shape_dims.push_back(dim);
      continue;
    }
    auto dependencies =
        llvm::makeArrayRef(normalized_shape_dims).take_front(num_dependencies);
    auto dim_type =
        DynRangeType::get(DomainShapeAttr::get(context_, dependencies));
    auto dim_mapping = MappingAttr::GetIdentity(context_, num_dependencies,
                                                normalized_shape_dims.size());
    normalized_shape_dims.emplace_back(dim_type, dim_mapping);
  }
  return DomainShapeAttr::get(context_, normalized_shape_dims);
}

}  // namespace sair
