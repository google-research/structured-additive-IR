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

namespace sair {

// Infers the iteration space for the current operation from iteration space of
// the given operand. Trims inner loops so than only loops iterating on
// dimensions mapped by the mapping remain. The resulting loop nest may
// not cover all dimensions of the current operation.
static mlir::ArrayAttr InferIterationSpace(
    mlir::ArrayAttr operand_iteration_space, ValueOperand &operand) {
  mlir::MLIRContext *context = operand_iteration_space.getContext();
  MappingAttr mapping = operand.Mapping();

  llvm::SmallVector<mlir::Attribute, 4> iteration_space;
  for (mlir::Attribute attr : operand_iteration_space.getValue()) {
    LoopAttr loop = attr.cast<LoopAttr>();
    if (loop.iter().MinDomainSize() > mapping.size()) break;

    MappingExpr new_iter = loop.iter().SubstituteDims(mapping.Dimensions());
    LoopAttr new_loop = LoopAttr::get(loop.name(), new_iter, context);
    iteration_space.push_back(new_loop);
  }
  // If the iteration space is infered from loop-carried dimensions, trim inner
  // parallel dimensions as inner parallel dimension open at the end of the
  // previous iteration along loop-carried  dimension may not be open at the
  // beginning of the current iteration.
  if (operand.AllowUseBeforeDef()) {
    llvm::SmallBitVector carrying_dims = operand.CarryingDims();
    while (!iteration_space.empty()) {
      LoopAttr loop = iteration_space.back().cast<LoopAttr>();
      int domain_size = mapping.UseDomainSize();
      if (loop.iter().DependencyMask(domain_size).anyCommon(carrying_dims)) {
        break;
      }
      iteration_space.pop_back();
    }
  }
  return mlir::ArrayAttr::get(context, iteration_space);
}

IterationSpaceAnalysis::IterationSpaceAnalysis(SairProgramOp program_op) {
  if (program_op == nullptr) return;
  for (mlir::Operation &op : program_op.body().front()) {
    ComputeIterationSpace(&op);
  }
}

llvm::ArrayRef<mlir::Attribute> IterationSpaceAnalysis::IterationSpace(
    SairOp op) const {
  return iteration_space_.find(op.getOperation())->second.getValue();
}

llvm::ArrayRef<mlir::Attribute> IterationSpaceAnalysis::IterationSpace(
    mlir::Value value) const {
  return IterationSpace(value.getDefiningOp());
}

mlir::ArrayAttr IterationSpaceAnalysis::ComputeIterationSpace(
    mlir::Operation *operation) {
  if (auto it = iteration_space_.find(operation);
      it != iteration_space_.end()) {
    return it->second;
  }

  mlir::MLIRContext *context = operation->getContext();
  mlir::ArrayAttr iteration_space = mlir::ArrayAttr::get(context, {});
  if (auto compute_op = dyn_cast<ComputeOp>(operation)) {
    iteration_space = compute_op.loop_nest().getValueOr(iteration_space);
  } else if (auto infer_iteration_space =
                 dyn_cast<InferIterationSpaceOp>(operation)) {
    // Temporarily set the loop nest to nullptr to avoid infinite recursion.
    iteration_space_[operation] = iteration_space;
    int operand_pos = infer_iteration_space.infer_iteration_space_operand();
    ValueOperand operand = cast<SairOp>(operation).ValueOperands()[operand_pos];
    mlir::Operation *defining_op = operand.value().getDefiningOp();
    mlir::ArrayAttr parent_iteration_space = ComputeIterationSpace(defining_op);
    iteration_space = InferIterationSpace(parent_iteration_space, operand);
  }
  iteration_space_[operation] = iteration_space;
  return iteration_space;
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

    mlir::ArrayRef<mlir::Attribute> iteration_space =
        iteration_spaces.IterationSpace(operation);
    llvm::SmallBitVector closed_dims = op.ResultsDimDependencies();
    bool closed_dims_seen = false;
    for (mlir::Attribute attr : iteration_space) {
      LoopAttr loop = attr.cast<LoopAttr>();
      constraints.open_loops.insert(loop.name());
      llvm::SmallBitVector iter_dims =
          loop.iter().DependencyMask(op.domain().size());
      if (iter_dims.anyCommon(closed_dims)) {
        constraints.closed_loops.insert(loop.name());
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

// Verifies that the loop_nest attribute is correct with regard to the shape of
// the operation it is attached to.
static mlir::LogicalResult VerifyLoopNestWellFormed(
    SairOp op, llvm::ArrayRef<mlir::Attribute> loop_nest) {
  llvm::SmallVector<MappingExpr, 4> iter_exprs;
  iter_exprs.reserve(loop_nest.size());

  int domain_size = op.domain().size();
  // Bitfield that keeps track of which dimensions are implemented by loops.
  for (int i = 0, e = loop_nest.size(); i < e; ++i) {
    LoopAttr loop = loop_nest[i].dyn_cast<LoopAttr>();
    if (loop == nullptr) {
      return op.emitError() << "expected a `Loop` attribute";
    }
    SairProgramOp parent = cast<SairProgramOp>(op->getParentOp());
    if (llvm::count(parent.loop_name_table(), loop.name()) == 0) {
      return op.emitError() << "loop " << loop.name()
                            << " is not declared in the parent operation";
    }

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

    iter_exprs.push_back(loop.iter());
  }

  mlir::MLIRContext *context = op.getContext();
  auto mapping = MappingAttr::getChecked(context, domain_size, iter_exprs);
  if (mapping == nullptr) {
    return op.emitError() << "incompatible loop iterators";
  }

  if (!mapping.Inverse().IsFullySpecified()) {
    return op.emitError() << "not all dimensions are covered by the loop nest";
  }

  return mlir::success();
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
  llvm::SmallVector<mlir::StringAttr, 8> open_loops_;
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
// * `mapping`: the mapping used by `op` to access `dependency`.
// * `dim_dependencies`: dimensions of `op` that cannot be part of the loop-nest
//    producing `dependency`.
// * `carrying_dims`: if `dependency` is a loop-carried operand, lists
//    dimensions carrying the value of `dependency` across iterations.
static mlir::LogicalResult VerifyDependency(
    SairOp op, llvm::ArrayRef<mlir::Attribute> op_loop_nest,
    mlir::Value dependency, MappingAttr mapping,
    const llvm::SmallBitVector &dim_dependencies,
    const llvm::SmallBitVector &carrying_dims,
    const IterationSpaceAnalysis &iteration_space_analysis,
    const LoopNestConstraintsAnalysis &loop_constraints_analysis) {
  mlir::ArrayRef<mlir::Attribute> dep_loop_nest =
      iteration_space_analysis.IterationSpace(dependency);

  // Verify dependencies with the operand loop nest.
  for (auto [op_attr, dep_attr] : llvm::zip(op_loop_nest, dep_loop_nest)) {
    LoopAttr op_loop = op_attr.cast<LoopAttr>();
    LoopAttr dep_loop = dep_attr.cast<LoopAttr>();
    if (op_loop.name() != dep_loop.name()) break;
    // Ensure that we can unify the iterator of both loops if they are fused.
    MappingExpr expected_expr =
        dep_loop.iter().SubstituteDims(mapping.Dimensions());
    if (expected_expr.Unify(op_loop.iter()) != nullptr) continue;
    return (op.emitError() << "loop " << op_loop.name()
                           << " violates a data dependency")
               .attachNote(dependency.getLoc())
           << "dependency from this operation";
  }

  const LoopNestConstraintsAnalysis::Constraints &constraints =
      loop_constraints_analysis.GetConstraints(dependency);
  for (mlir::Attribute attr : op_loop_nest) {
    LoopAttr loop = attr.cast<LoopAttr>();
    if (constraints.closed_loops.contains(loop.name())) {
      return op.emitError() << "loop " << loop.name()
                            << " must be closed before this operation";
    }

    if (!constraints.open_loops.contains(loop.name())) continue;
    llvm::SmallBitVector iter_dims =
        loop.iter().DependencyMask(op.domain().size());
    if (!dim_dependencies.anyCommon(iter_dims)) continue;

    return (dependency.getDefiningOp()->emitError()
            << "operation cannot be nested in loop " << loop.name())
               .attachNote(op.getLoc())
           << "because of this operation";
  }

  for (int dep_dimension : constraints.closed_dimensions.set_bits()) {
    int domain_size = mapping.UseDomainSize();
    if (dep_dimension >= mapping.size()) break;
    llvm::SmallBitVector mapped_dims =
        mapping.Dimension(dep_dimension).DependencyMask(domain_size);
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
    SairOp op, IterationSpaceAnalysis &iteration_space_analysis,
    LoopNestConstraintsAnalysis &loop_constaints_analysis) {
  llvm::ArrayRef<mlir::Attribute> loop_nest =
      iteration_space_analysis.IterationSpace(op);

  int domain_size = op.domain().size();
  for (int i = 0; i < domain_size; ++i) {
    llvm::SmallBitVector dim_dependencies(op.domain().size());
    llvm::SmallBitVector carrying_dims(op.domain().size());
    dim_dependencies.set(i);
    MappingAttr mapping = op.shape().Dimensions()[i].dependency_mapping();
    if (mlir::failed(VerifyDependency(op, loop_nest, op.domain()[i], mapping,
                                      dim_dependencies, carrying_dims,
                                      iteration_space_analysis,
                                      loop_constaints_analysis))) {
      return mlir::failure();
    }
  }

  for (ValueOperand operand : op.ValueOperands()) {
    if (mlir::failed(VerifyDependency(
            op, loop_nest, operand.value(), operand.Mapping(),
            operand.DependingDims(), operand.CarryingDims(),
            iteration_space_analysis, loop_constaints_analysis))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Verifies that it is possible to compute the range of loops and that the
// range is defined before it is used.
static mlir::LogicalResult VerifyLoopRanges(
    ComputeOp op, llvm::ArrayRef<mlir::Attribute> loop_nest,
    const LoopFusionAnalysis &fusion_analysis) {
  for (mlir::Attribute attr : loop_nest) {
    LoopAttr loop = attr.cast<LoopAttr>();
    const LoopFusionClass &fusion_class = fusion_analysis.GetClass(loop.name());
    if (!fusion_class.iter_expr.IsFullySpecified()) {
      return op.emitError()
             << "loop " << loop.name() << " iterator is not fully specified";
    }

    for (mlir::Value dimension : fusion_class.dimensions) {
      if (op.getOperation()->isBeforeInBlock(dimension.getDefiningOp())) {
        return (op.emitError()
                << "rematerialized loop " << loop.name()
                << " indirectly uses the range before it is defined")
                   .attachNote(dimension.getLoc())
               << "range defined here";
      }
    }
  }

  return mlir::success();
}

// Verifies that loops satisfy dependencies: if the range of loop "A" depends on
// the index of loop "B", then "A" must be nested inside "B".
// TODO(ulysse): we can compute this for all loop nests at once instead of
// duplicating the work for each loop nest (LoopNestShapeAnalysis?).
static mlir::LogicalResult VerifyLoopNestShape(
    ComputeOp loop_nest_op, llvm::ArrayRef<mlir::Attribute> loop_nest,
    const LoopFusionAnalysis &fusion_classes) {
  mlir::MLIRContext *context = loop_nest_op.getContext();
  llvm::SmallVector<DomainShapeDim, 4> joint_domain;
  // [<joint-domain>] -> (<op-domain>) mapping for each operation.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<MappingExpr, 4>>
      op_mappings;
  llvm::SmallVector<MappingExpr, 4> iter_exprs;
  auto none_expr = MappingNoneExpr::get(context);

  // Add dimensions loops iterate on to the joint domain shape while building a
  // mapping from the ops domain to the joint domain.
  for (mlir::Attribute attr : loop_nest) {
    LoopAttr loop = attr.cast<LoopAttr>();
    LoopFusionClass fusion_class = fusion_classes.GetClass(loop.name());
    // [<joint-domain>] -> (<fusion-class-domain>) mapping.
    llvm::SmallVector<MappingExpr, 4> class_mapping(
        fusion_class.dimensions.size(), none_expr);

    for (auto [operation, mapping] : fusion_class.op_dims_mappings) {
      SairOp op = cast<SairOp>(operation);
      llvm::SmallVector<MappingExpr, 4> &op_mapping = op_mappings[operation];
      op_mapping.resize(op.domain().size(), none_expr);

      for (auto en : llvm::enumerate(mapping)) {
        if (en.value().isa<MappingNoneExpr>()) continue;
        int class_dimension = en.value().cast<MappingDimExpr>().dimension();

        // If the op dimension is alreadey mapped to a joint dimension, just
        // copy the mapping for the fusion class.
        if (!op_mapping[en.index()].isa<MappingNoneExpr>()) {
          class_mapping[class_dimension] =
              class_mapping[class_dimension].Unify(op_mapping[en.index()]);
          assert(class_mapping[class_dimension] != nullptr);
          continue;
        }

        // If the class dimension is already mapped to a joint domain dimension,
        // just copy the mapping for this operation.
        if (!class_mapping[class_dimension].isa<MappingNoneExpr>()) {
          op_mapping[en.index()] =
              op_mapping[en.index()].Unify(class_mapping[class_dimension]);
          assert(op_mapping[en.index()] != nullptr);
          continue;
        }

        // Otherwise, add a new dimension to the joint domain.
        const DomainShapeDim &dim_shape = op.shape().Dimension(en.index());
        for (int dependency : dim_shape.DependencyMask().set_bits()) {
          // Check that dependencies are already added.
          if (!op_mapping[dependency].IsFullySpecified()) {
            return operation->emitError()
                   << "dimension d" << en.index() << " in loop " << loop.name()
                   << " is used before its dependencies";
          }
        }

        auto expr = MappingDimExpr::get(joint_domain.size(), context);
        auto mapping =
            MappingAttr::get(context, joint_domain.size(), op_mapping);
        op_mapping[en.index()] = expr;
        class_mapping[class_dimension] = expr;
        joint_domain.push_back(dim_shape.Apply(mapping));
      }
    }
    iter_exprs.push_back(fusion_class.iter_expr.SubstituteDims(class_mapping));
  }

  auto loops_mapping =
      MappingAttr::get(context, joint_domain.size(), iter_exprs);
  MappingAttr inverse_loops_mapping = loops_mapping.Inverse();

  for (auto en : llvm::enumerate(iter_exprs)) {
    DomainShapeDim loop_shape =
        en.value().AccessedShape(joint_domain, inverse_loops_mapping);
    int max_dep = loop_shape.DependencyMask().find_last();
    if (max_dep >= (int)en.index()) {
      LoopAttr loop = loop_nest[en.index()].cast<LoopAttr>();
      LoopAttr loop_dep = loop_nest[max_dep].cast<LoopAttr>();
      return loop_nest_op.emitError()
             << "loop " << loop.name() << " must be nested inside "
             << loop_dep.name();
    }
  }

  return mlir::success();
}

// Ensure that each loop only iterate along a single sub-domain.
static mlir::LogicalResult VerifySubDomains(
    SairOp op, llvm::ArrayRef<mlir::Attribute> iteration_space) {
  llvm::SmallVector<int, 2> sub_domains = op.SubDomains();
  assert(!sub_domains.empty() || iteration_space.empty());

  for (mlir::Attribute attr : iteration_space) {
    LoopAttr loop = attr.cast<LoopAttr>();
    llvm::SmallBitVector dimensions =
        loop.iter().DependencyMask(op.domain().size());
    if (!dimensions.any()) continue;

    // Compute the sub-domain the loop belongs to. If the iterator is not fully
    // specified, then reaterializing dimensions will be added to the parallel
    // sub-domain (sub-domain 0) and so all dimensions must belong to the
    // parallel sub-domain.
    int sub_domain = 0;
    int min_dim_index = 0;
    int max_dim_index = sub_domains[0];
    if (loop.iter().IsFullySpecified()) {
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
      return op.emitError()
             << "loop " << loop.name() << " crosses sub-domains boundaries";
    }
  }
  return mlir::success();
}

mlir::LogicalResult VerifyLoopNests(SairProgramOp program) {
  // Verify operands of Sair operands are defined in the same program. This
  // check is performed here rather that in SairOp as it is needed for other
  // verifications.
  mlir::WalkResult result = program.walk([&](SairOp op) -> mlir::WalkResult {
    for (mlir::Value dimension : op.domain()) {
      mlir::Operation *defining_op = dimension.getDefiningOp();
      if (defining_op == nullptr || defining_op->getParentOp() != program) {
        return op.emitError()
               << "sair dimensions must be defined in the region they are used";
      }
    }
    for (ValueOperand operand : op.ValueOperands()) {
      mlir::Operation *defining_op = operand.value().getDefiningOp();
      if (defining_op == nullptr || defining_op->getParentOp() != program) {
        return op.emitError()
               << "sair values must be defined in the region they are used";
      }
    }
    return mlir::success();
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Verify loop nests are correct with regard to their operation.
  result = program.walk([](ComputeOp op) -> mlir::WalkResult {
    if (!op.loop_nest().hasValue()) return mlir::WalkResult::advance();
    return VerifyLoopNestWellFormed(
        cast<SairOp>(op.getOperation()), op.LoopNestLoops());
  });
  if (result.wasInterrupted()) return mlir::failure();

  IterationSpaceAnalysis iteration_space_analysis(program);
  LoopNestConstraintsAnalysis loop_constraints_analysis(
      program, iteration_space_analysis);

  // Verify that the loop structure forms a tree, loops are open when they need
  // to and loop ranges are well defined.
  LoopNestState loop_nest_state;
  auto fusion_analysis_or_null = LoopFusionAnalysis::Create(program);
  if (fusion_analysis_or_null == std::nullopt) return mlir::failure();
  LoopFusionAnalysis fusion_analysis = fusion_analysis_or_null.value();
  result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    if (op.loop_nest().hasValue()) {
      if (mlir::failed(loop_nest_state.Update(cast<SairOp>(op.getOperation()),
                                              op.LoopNestLoops()))) {
        return mlir::failure();
      }
      if (mlir::failed(
              VerifyLoopRanges(op, op.LoopNestLoops(), fusion_analysis))) {
        return mlir::failure();
      }
      if (mlir::failed(
              VerifyLoopNestShape(op, op.LoopNestLoops(), fusion_analysis))) {
        return mlir::failure();
      }
    } else if (mlir::failed(loop_nest_state.CloseLoops())) {
      return mlir::failure();
    }
    return VerifyLoopsOpen(cast<SairOp>(op.getOperation()), loop_nest_state,
                           loop_constraints_analysis);
  });
  if (result.wasInterrupted()) return mlir::failure();
  if (mlir::failed(loop_nest_state.CloseLoops())) return mlir::failure();

  // Verify dependencies.
  result = program.walk([&](SairOp op) -> mlir::WalkResult {
    if (mlir::failed(VerifySubDomains(
            op, iteration_space_analysis.IterationSpace(op)))) {
      return mlir::failure();
    }
    return VerifyDependencies(op, iteration_space_analysis,
                              loop_constraints_analysis);
  });
  if (result.wasInterrupted()) return mlir::failure();

  return mlir::success();
}

LoopFusionAnalysis::LoopFusionAnalysis(mlir::Operation *operation) {
  SairProgramOp program_op = dyn_cast<SairProgramOp>(operation);
  if (program_op == nullptr) return;
  mlir::LogicalResult status = Init(program_op);
  assert(mlir::succeeded(status));
  (void)status;
}

std::optional<LoopFusionAnalysis> LoopFusionAnalysis::Create(
    SairProgramOp program_op) {
  LoopFusionAnalysis analysis;
  if (mlir::failed(analysis.Init(program_op))) return std::nullopt;
  return analysis;
}

mlir::LogicalResult LoopFusionAnalysis::Init(SairProgramOp program_op) {
  auto status = program_op.walk([&](ComputeOp op) -> mlir::WalkResult {
    if (!op.loop_nest().hasValue()) return mlir::success();
    llvm::ArrayRef<mlir::Attribute> loop_nest = op.LoopNestLoops();
    for (mlir::Attribute attr : loop_nest) {
      if (mlir::failed(RegisterLoop(op, attr.cast<LoopAttr>(), loop_nest))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  });
  return mlir::failure(status.wasInterrupted());
}

mlir::LogicalResult LoopFusionAnalysis::RegisterLoop(
    ComputeOp op, LoopAttr loop, llvm::ArrayRef<mlir::Attribute> loop_nest) {
  mlir::MLIRContext *context = op.getContext();
  LoopFusionClass &fusion_class = fusion_classes_[loop.name()];
  if (fusion_class.iter_expr == nullptr) {
    fusion_class.iter_expr = MappingNoneExpr::get(context);
  }

  SairOp sair_op = cast<SairOp>(op.getOperation());
  int domain_size = sair_op.domain().size();

  // Solve unification constraints.
  llvm::SmallVector<MappingExpr, 4> mapping(sair_op.domain().size(),
                                            MappingNoneExpr::get(context));
  if (mlir::failed(loop.iter().UnificationConstraints(fusion_class.iter_expr,
                                                      mapping))) {
    return op.emitError() << "failed to unify loop " << loop.name()
                          << " iterators";
  }

  llvm::SmallBitVector iter_dims = loop.iter().DependencyMask(domain_size);
  for (int dimension : iter_dims.set_bits()) {
    MappingExpr expr = mapping[dimension];
    if (expr.isa<MappingNoneExpr>()) {
      mapping[dimension] =
          MappingDimExpr::get(fusion_class.dimensions.size(), context);
      fusion_class.dimensions.push_back(sair_op.domain()[dimension]);
      // TODO(ulysse): make it work with tiling before introducing tiled loops
      // in passes.
      llvm::SmallBitVector dependencies =
          sair_op.shape().Dimension(dimension).DependencyMask();
      for (mlir::Attribute attr : loop_nest) {
        LoopAttr other_loop = attr.cast<LoopAttr>();
        llvm::SmallBitVector other_dims =
            other_loop.iter().DependencyMask(domain_size);
        if (other_dims.anyCommon(dependencies)) {
          fusion_class.dependencies.push_back(other_loop.name());
        }
      }
    } else if (auto dim_expr = expr.dyn_cast<MappingDimExpr>()) {
      mlir::Value expected_dimension =
          fusion_class.dimensions[dim_expr.dimension()];
      if (sair_op.domain()[dimension] != expected_dimension) {
        return op.emitError()
               << "dimension d" << dimension << " in loop " << loop.name()
               << " does not match previous occurrences";
      }
    } else {
      return op.emitError() << "cannot unify d" << dimension << " with '"
                            << expr << "' in loop " << loop.name();
    }
  }

  // Unify iterator expressions.
  fusion_class.iter_expr =
      fusion_class.iter_expr.Unify(loop.iter().SubstituteDims(mapping));
  assert(fusion_class.iter_expr != nullptr);

  fusion_class.op_dims_mappings[op.getOperation()] = std::move(mapping);

  return mlir::success();
}

}  // namespace sair
