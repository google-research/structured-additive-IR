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

// Infers a new loop for the current operation from the loop nest `loop_nest` of
// the given operand. Trims inner loops so than only loops iterating on
// dimensions mapped by the access pattern remain. The resulting loop nest may
// not cover all dimensions of the current operation.
static mlir::ArrayAttr InferLoopNest(mlir::ArrayAttr loop_nest,
                                     ValueOperand &operand) {
  if (loop_nest == nullptr) return nullptr;
  mlir::MLIRContext *context = loop_nest.getContext();
  AccessPatternAttr access_pattern = operand.AccessPattern();

  llvm::SmallVector<mlir::Attribute, 4> new_loop_nest;
  for (mlir::Attribute attr : loop_nest.getValue()) {
    LoopAttr loop = attr.cast<LoopAttr>();
    if (loop.iter().Rematerialize()) {
      new_loop_nest.push_back(loop);
      continue;
    }

    int dimension = loop.iter().Dimension();
    if (dimension >= access_pattern.Dimensions().size()) {
      break;
    }

    IteratorAttr new_iter =
        IteratorAttr::get(context, access_pattern.Dimension(dimension));
    LoopAttr new_loop = LoopAttr::get(loop.name(), new_iter, context);
    new_loop_nest.push_back(new_loop);
  }
  // If the loop-nest is infered from loop-carried dimensions, trim inner
  // parallel dimensions as inner parallel dimension open at the end of the
  // previous iteration along loop-carried  dimension may not be open at the
  // beginning of the current iteration.
  if (operand.AllowUseBeforeDef()) {
    llvm::SmallBitVector carrying_dims = operand.CarryingDims();
    while (!new_loop_nest.empty()) {
      LoopAttr loop = new_loop_nest.back().cast<LoopAttr>();
      if (!loop.iter().Rematerialize() &&
          carrying_dims.test(loop.iter().Dimension())) {
        break;
      }
      new_loop_nest.pop_back();
    }
  }
  return mlir::ArrayAttr::get(new_loop_nest, context);
}

IterationSpaceAnalysis::IterationSpaceAnalysis(SairProgramOp program) {
  for (mlir::Operation &op : program.body().front()) {
    ComputeIterationSpace(&op);
  }
}

mlir::ArrayAttr IterationSpaceAnalysis::IterationSpace(SairOp op) const {
  return iteration_space_.find(op.getOperation())->second;
}

mlir::ArrayAttr IterationSpaceAnalysis::IterationSpace(
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
  mlir::ArrayAttr iteration_space = mlir::ArrayAttr::get({}, context);
  if (auto compute_op = dyn_cast<ComputeOp>(operation)) {
    iteration_space = compute_op.loop_nest().getValueOr(iteration_space);
  } else if (auto infer_iteration_space =
                 dyn_cast<InferIterationSpaceOp>(operation)) {
    // Temporarily set the loop nest to nullptr to avoid infinite recursion.
    iteration_space_[operation] = iteration_space;
    int operand_pos = infer_iteration_space.infer_iteration_space_operand();
    ValueOperand operand = cast<SairOp>(operation).ValueOperands()[operand_pos];
    mlir::ArrayAttr parent_iteration_space =
        ComputeIterationSpace(operand.value().getDefiningOp());
    iteration_space = InferLoopNest(parent_iteration_space, operand);
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

    auto inherit_constraints = [&](mlir::Value value,
                                   AccessPatternAttr access_pattern,
                                   bool loop_carried = false) {
      const Constraints &parent_constraint =
          ComputeConstraints(value.getDefiningOp(), iteration_spaces);
      for (int closed_dim : parent_constraint.closed_dimensions.set_bits()) {
        if (closed_dim > access_pattern.Dimensions().size()) break;
        int op_dimension = access_pattern.Dimension(closed_dim);
        constraints.closed_dimensions.set(op_dimension);
      }
      if (loop_carried) return;
      constraints.open_loops.set_union(parent_constraint.open_loops);
      constraints.closed_loops.set_union(parent_constraint.closed_loops);
    };

    if (!isa<ComputeOp>(operation)) {
      // Store empty constraints to avoid infinite recursion.
      constraints_.try_emplace(operation, domain_size);
      for (int i = 0, e = domain_size; i < e; ++i) {
        AccessPatternAttr access_pattern =
            op.shape().Dimension(i).dependency_pattern();
        inherit_constraints(op.domain()[i], access_pattern);
      }
      for (ValueOperand operand : op.ValueOperands()) {
        inherit_constraints(operand.value(), operand.AccessPattern(),
                            operand.AllowUseBeforeDef());
      }
    }

    mlir::ArrayAttr iteration_space =
        iteration_spaces.IterationSpace(operation);
    llvm::SmallBitVector closed_dims = op.ResultsDimDependencies();
    bool closed_dims_seen = false;
    for (mlir::Attribute attr : iteration_space.getValue()) {
      LoopAttr loop = attr.cast<LoopAttr>();
      constraints.open_loops.insert(loop.name());
      if (loop.iter().Rematerialize()) continue;
      int dimension = loop.iter().Dimension();
      if (closed_dims.test(dimension)) {
        constraints.closed_loops.insert(loop.name());
        closed_dims_seen = true;
      }
      if (closed_dims_seen && dimension < op.results_rank()) {
        constraints.closed_dimensions.set(dimension);
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
  int domain_size = op.domain().size();
  // Bitfield that keeps track of which dimensions are implemented by loops.
  llvm::SmallBitVector covered_dimensions(domain_size);
  llvm::SmallVector<int, 8> steps(domain_size, -1);
  for (int i = 0, e = loop_nest.size(); i < e; ++i) {
    LoopAttr loop = loop_nest[i].dyn_cast<LoopAttr>();
    if (loop == nullptr) {
      return op.emitError() << "expected a `Loop` attribute";
    }
    SairProgramOp parent = cast<SairProgramOp>(op.getParentOp());
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

    if (loop.iter().Rematerialize()) continue;
    int dimension = loop.iter().Dimension();

    if (dimension >= domain_size) {
      return op.emitError() << "dimension 'd" << loop.iter().Dimension() << "' "
                            << "is out of range of the domain";
    }

    llvm::SmallBitVector missing_outer_dims =
        op.shape().Dimensions()[dimension].DependencyMask() &
        ~covered_dimensions;
    if (missing_outer_dims.any()) {
      return op.emitError()
             << "dependencies of dimension 'd" << dimension << "' "
             << "must be nested in dimension 'd"
             << *missing_outer_dims.set_bits_begin() << "'";
    }

    int step = loop.iter().Step();
    // Mark the dimension as covered if the loop has step one. This ensure that
    // we iterate over the points of each dimensions, and not just the tiles.
    if (step == 1) covered_dimensions.set(dimension);

    if (steps[dimension] != -1 && steps[dimension] <= step) {
      return op.emitError() << "loop " << loop.name()
                            << " step must be less than in outer loops";
    }
    steps[dimension] = step;
  }

  if (!covered_dimensions.all()) {
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
    for (int e = std::min(loop_nest.size(), open_groups_.size());
         common_prefix_size < e; ++common_prefix_size) {
      LoopAttr loop = loop_nest[common_prefix_size].cast<LoopAttr>();
      FusionGroup &group = open_groups_[common_prefix_size];
      if (loop.name() != group.name) break;

      group.ops.push_back(op);
      if (loop.iter().Rematerialize()) continue;

      mlir::Value dimension = op.domain()[loop.iter().Dimension()];

      // Set the group dimension if not already set.
      if (group.dimension == nullptr) {
        group.dimension = dimension;
        group.step = loop.iter().Step();
        group.defining_op = op;
        continue;
      }

      // Ensure the loop definition matches the rest of the group.
      if (group.dimension != dimension || group.step != loop.iter().Step()) {
        return (op.emitError()
                << "loop " << group.name
                << " dimension does not match previous occurrences")
                   .attachNote(group.defining_op.getLoc())
               << "previous occurrence here";
      }
    }

    // Reset the current fusion prefix to the number of common loops.
    if (mlir::failed(CloseLoops(common_prefix_size))) return mlir::failure();

    // Add remaining loops to the current fusion prefix.
    for (mlir::Attribute attribute : loop_nest.drop_front(common_prefix_size)) {
      LoopAttr loop = attribute.cast<LoopAttr>();

      if (closed_groups_.count(loop.name()) > 0) {
        return op.emitError()
               << "occurrences of loop " << loop.name()
               << " must be contiguous and nested in the same loops";
      }

      FusionGroup &group = open_groups_.emplace_back();
      group.name = loop.name();
      group.defining_op = op;
      group.ops.push_back(op);

      if (loop.iter().Rematerialize()) continue;
      group.dimension = op.domain()[loop.iter().Dimension()];
      group.step = loop.iter().Step();
    }
    return mlir::success();
  }

  // Mark loops as closed, starting from the innermost`, until only
  // `num_remaining_loops` are left open.
  mlir::LogicalResult CloseLoops(int num_remaining_loops = 0) {
    while (open_groups_.size() > num_remaining_loops) {
      FusionGroup group = open_groups_.pop_back_val();
      if (group.dimension == nullptr) {
        group.defining_op.emitError()
            << "loop " << group.name
            << " must have the 'iter' field set in at least one operation";
        return mlir::failure();
      }
      for (SairOp op : group.ops) {
        if (!group.dimension.getDefiningOp()->isBeforeInBlock(op)) {
          (op.emitError() << "rematerialized loop " << group.name
                          << " indirectly uses the range before it is defined")
                  .attachNote(group.dimension.getLoc())
              << "range defined here";
          return mlir::failure();
        }
      }
      closed_groups_.insert(group.name);
    }
    return mlir::success();
  };

  // Verifies that the given loops have been open before.
  mlir::LogicalResult VerifyLoopsOpen(
      const llvm::SetVector<mlir::Attribute> &loops, mlir::Location loc) const {
    for (mlir::Attribute loop : loops) {
      if (closed_groups_.contains(loop)) continue;
      if (llvm::any_of(open_groups_, [&](const FusionGroup &group) {
            return group.name == loop;
          })) {
        continue;
      }
      return mlir::emitError(loc)
             << "loop " << loop << " must be open at or before this operation";
    }
    return mlir::success();
  }

 private:
  struct FusionGroup {
    mlir::StringAttr name;
    mlir::Value dimension;
    int step;
    // If dimension == nullptr, contains the first operation referencing the
    // loop. Otherwise, contains the first operation referencing the loop with
    // the `iter` field set.
    SairOp defining_op;
    llvm::SmallVector<SairOp, 4> ops;
  };

  llvm::SmallVector<FusionGroup, 8> open_groups_;
  llvm::DenseSet<mlir::Attribute> closed_groups_;
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
// * `access_pattern`: the access pattern used by `op` to access `dependency`.
// * `dim_dependencies`: dimensions of `op` that cannot be part of the loop-nest
//    producing `dependency`.
// * `carrying_dims`: if `dependency` is a loop-carried operand, lists
//    dimensions carrying the value of `dependency` across iterations.
static mlir::LogicalResult VerifyDependency(
    SairOp op, mlir::ArrayAttr op_loop_nest, mlir::Value dependency,
    AccessPatternAttr access_pattern,
    const llvm::SmallBitVector &dim_dependencies,
    const llvm::SmallBitVector &carrying_dims,
    const IterationSpaceAnalysis &iteration_space_analysis,
    const LoopNestConstraintsAnalysis &loop_constraints_analysis) {
  mlir::ArrayAttr dep_loop_nest =
      iteration_space_analysis.IterationSpace(dependency);
  if (dep_loop_nest == nullptr) return mlir::success();

  // Verify dependencies with the operand loop nest.
  for (auto [op_attr, dep_attr] :
       llvm::zip(op_loop_nest.getValue(), dep_loop_nest.getValue())) {
    LoopAttr op_loop = op_attr.cast<LoopAttr>();
    LoopAttr dep_loop = dep_attr.cast<LoopAttr>();
    if (op_loop.name() != dep_loop.name()) break;

    // Check mapped dimensions. If two dimensions are mapped by the access
    // pattern, the op dimension must be fused with the operand dimension
    // or be in a separate loop nest.
    if (dep_loop.iter().Rematerialize()) continue;
    int dep_dimension = dep_loop.iter().Dimension();
    if (dep_dimension >= access_pattern.Dimensions().size()) continue;
    int mapped_dimension = access_pattern.Dimension(dep_dimension);
    if (mapped_dimension == AccessPatternAttr::kNoDimension) continue;
    if (op_loop.iter().Rematerialize() ||
        op_loop.iter().Dimension() != mapped_dimension) {
      return (op.emitError()
              << "loop " << op_loop.name() << " violates a data dependency")
                 .attachNote(dependency.getLoc())
             << "dependency from this operation";
    }
  }

  const LoopNestConstraintsAnalysis::Constraints &constraints =
      loop_constraints_analysis.GetConstraints(dependency);
  for (mlir::Attribute attr : op_loop_nest) {
    LoopAttr loop = attr.cast<LoopAttr>();
    if (constraints.closed_loops.contains(loop.name())) {
      return op.emitError() << "loop " << loop.name()
                            << " must be closed before this operation";
    }

    if (loop.iter().Rematerialize()) continue;
    if (!dim_dependencies.test(loop.iter().Dimension())) continue;
    if (!constraints.open_loops.contains(loop.name())) continue;

    return (dependency.getDefiningOp()->emitError()
            << "operation cannot be nested in loop " << loop.name())
               .attachNote(op.getLoc())
           << "because of this operation";
  }

  for (int dep_dimension : constraints.closed_dimensions.set_bits()) {
    int op_dimension = access_pattern.Dimension(dep_dimension);
    if (carrying_dims.test(op_dimension)) {
      return op.emitError()
             << "cannot take the previous value of the operand along 'd"
             << op_dimension << "' because of the operand loop nest";
    }
  }

  return mlir::success();
}

// Verifies that the loop nest of `op` is compatible with the constraints
// imposed by its dependencies.
static mlir::LogicalResult VerifyDependencies(
    SairOp op, IterationSpaceAnalysis &iteration_space_analysis,
    LoopNestConstraintsAnalysis &loop_constaints_analysis) {
  mlir::ArrayAttr loop_nest = iteration_space_analysis.IterationSpace(op);
  if (loop_nest == nullptr) return mlir::success();

  int domain_size = op.domain().size();
  for (int i = 0; i < domain_size; ++i) {
    llvm::SmallBitVector dim_dependencies(op.domain().size());
    llvm::SmallBitVector carrying_dims(op.domain().size());
    dim_dependencies.set(i);
    AccessPatternAttr access_pattern =
        op.shape().Dimensions()[i].dependency_pattern();
    if (mlir::failed(VerifyDependency(op, loop_nest, op.domain()[i],
                                      access_pattern, dim_dependencies,
                                      carrying_dims, iteration_space_analysis,
                                      loop_constaints_analysis))) {
      return mlir::failure();
    }
  }

  for (ValueOperand operand : op.ValueOperands()) {
    if (mlir::failed(VerifyDependency(
            op, loop_nest, operand.value(), operand.AccessPattern(),
            operand.DependingDims(), operand.CarryingDims(),
            iteration_space_analysis, loop_constaints_analysis))) {
      return mlir::failure();
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

  // Verify that the loop structure forms a tree.
  LoopNestState loop_nest_state;
  result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    if (op.loop_nest().hasValue()) {
      if (mlir::failed(loop_nest_state.Update(cast<SairOp>(op.getOperation()),
                                              op.LoopNestLoops()))) {
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
    return VerifyDependencies(op, iteration_space_analysis,
                              loop_constraints_analysis);
  });
  if (result.wasInterrupted()) return mlir::failure();

  return mlir::success();
}

}  // namespace sair
