#include "loop_nest.h"

namespace sair {

// A dependency of a Sair operation.
struct Dependency {
  // Operation the use operation depends on.
  ComputeOp def;
  // Dimensions of the def operation that must complete before the current
  // instance of the use operation execute.
  llvm::SmallBitVector def_only_dimensions;
  // Dimensions of the use operation that cannot execute before the accessed
  // instance of def is computed.
  llvm::SmallBitVector use_only_dimensions;
  // Dimension of the use operation that carry the dependency accross
  // iterations. They must be fused with the dimensions of the def operation
  // they map to.
  llvm::SmallBitVector carrying_dimensions;
  // Dimensions of the def operation that must complete at the previous
  // iteration of `carrying_dimensions`. In practice, this means that they are
  // nested in carrying dimensions.
  llvm::SmallBitVector prev_def_only_dimensions;
  // Point-to-point communication pattern from def to use.
  AccessPatternAttr mapped_dimensions;
};

static void AddDependencyFrom(
    SairOp def, AccessPatternAttr access_pattern,
    llvm::SmallBitVector def_only_dimensions,
    llvm::SmallBitVector use_only_dimensions,
    llvm::SmallBitVector carrying_dimensions,
    llvm::SmallBitVector prev_def_only_dimensions, bool is_loop_carried,
    llvm::SmallVectorImpl<Dependency> &dependencies,
    llvm::SmallVectorImpl<llvm::SmallBitVector> &use_dim_dependencies);

// Appends dependencies of `op` to the list of dependencies. Depdencies are
// expressed in the domain of a user operation.
// * `access_pattern`: the access pattern from `op` domain to the user domain.
// * `def_only_dimension`, `use_only_dimensions`, `carrying_dimensions` and
//   `prev_def_only_dimensions`: dimensions to add to the fields of the same
//   name when creating the `Dependency` object.
// * `is_loop_carried`: indicates if the dependency being built is a
//   loop-carried dependency.
// * `dependencies`: the list of dependencies to fill.
// * `use_dim_dependencies`: dependencies on the dimensions of the use domain,
//    to be filled by this function.
static void AddDependencies(
    SairOp op, AccessPatternAttr access_pattern,
    const llvm::SmallBitVector &def_only_dimensions,
    const llvm::SmallBitVector &use_only_dimensions,
    const llvm::SmallBitVector &carrying_dimensions,
    const llvm::SmallBitVector &prev_def_only_dimensions, bool is_loop_carried,
    llvm::SmallVectorImpl<Dependency> &dependencies,
    llvm::SmallVectorImpl<llvm::SmallBitVector> &use_dim_dependencies) {
  assert(access_pattern.Dimensions().size() == op.domain().size());

  auto add_dependency = [&](SairOp def, AccessPatternAttr def_access_pattern,
                            const llvm::SmallBitVector &new_use_only_dimensions,
                            const llvm::SmallBitVector &new_carrying_dimensions,
                            bool new_is_loop_carried) {
    AccessPatternAttr full_def_access_pattern =
        def_access_pattern.Resize(def.domain().size());

    llvm::SmallBitVector new_def_only_dimensions =
        full_def_access_pattern.ApplyInverse(def_only_dimensions);
    llvm::SmallBitVector new_prev_def_only_dimensions =
        full_def_access_pattern.ApplyInverse(prev_def_only_dimensions);

    // Otherwise, we need to compute transitive dependencies.
    AccessPatternAttr new_access_pattern =
        access_pattern.Compose(full_def_access_pattern);

    AddDependencyFrom(def, new_access_pattern, new_def_only_dimensions,
                      new_use_only_dimensions, new_carrying_dimensions,
                      new_prev_def_only_dimensions, new_is_loop_carried,
                      dependencies, use_dim_dependencies);
  };

  // Dimension dependencies are computed before entering the dimension.
  for (int i = 0, e = op.domain().size(); i < e; ++i) {
    SairOp def = op.domain()[i].getDefiningOp();
    llvm::SmallBitVector new_use_only_dimensions = use_only_dimensions;
    if (access_pattern.Dimension(i) != AccessPatternAttr::kNoDimension) {
      new_use_only_dimensions.set(access_pattern.Dimension(i));
    }
    AccessPatternAttr dependency_pattern =
        op.shape().Dimensions()[i].dependency_pattern();

    add_dependency(def, dependency_pattern, new_use_only_dimensions,
                   carrying_dimensions, is_loop_carried);
  }

  for (int i = 0, e = op.ValueOperands().size(); i < e; ++i) {
    ValueOperand operand = op.ValueOperands()[i];
    SairOp def = operand.value().getDefiningOp();
    llvm::SmallBitVector new_carrying_dimensions =
        access_pattern.Apply(op.CarryingDimensions(i)) | carrying_dimensions;
    llvm::SmallBitVector new_use_only_dimensions =
        access_pattern.Apply(op.DimsDependingOnOperand(i)) |
        use_only_dimensions;

    if (!operand.AllowUseBeforeDef()) {
      add_dependency(def, operand.AccessPattern(), new_use_only_dimensions,
                     new_carrying_dimensions, is_loop_carried);
      continue;
    }

    // If the operand is a loop-carried dependency, we force
    // use-only dimensions to be nested in forced fused dimensions instead of
    // putting them in use_only.
    llvm::SmallBitVector empty_use_only_dimensions(use_only_dimensions.size());
    for (int outer_dim : new_carrying_dimensions.set_bits()) {
      for (int inner_dim : new_use_only_dimensions.set_bits()) {
        use_dim_dependencies[inner_dim].set(outer_dim);
      }
    }

    add_dependency(def, operand.AccessPattern(), empty_use_only_dimensions,
                   new_carrying_dimensions, true);
  }
}

// Adds a dependency from `def` to the list of dependencies.
static void AddDependencyFrom(
    SairOp def, AccessPatternAttr access_pattern,
    llvm::SmallBitVector def_only_dimensions,
    llvm::SmallBitVector use_only_dimensions,
    llvm::SmallBitVector carrying_dimensions,
    llvm::SmallBitVector prev_def_only_dimensions, bool is_loop_carried,
    llvm::SmallVectorImpl<Dependency> &dependencies,
    llvm::SmallVectorImpl<llvm::SmallBitVector> &use_dim_dependencies) {
  if (is_loop_carried) {
    prev_def_only_dimensions |= def.ResultsDimDependencies();
  } else {
    def_only_dimensions |= def.ResultsDimDependencies();
  }

  // If the producer is a compute op, we can express the dependency
  // directly on the producer.
  ComputeOp compute_op = dyn_cast<ComputeOp>(def.getOperation());
  if (compute_op != nullptr) {
    dependencies.push_back({compute_op, def_only_dimensions,
                            use_only_dimensions, carrying_dimensions,
                            prev_def_only_dimensions, access_pattern});
    return;
  }

  AddDependencies(def, access_pattern, def_only_dimensions, use_only_dimensions,
                  carrying_dimensions, prev_def_only_dimensions,
                  is_loop_carried, dependencies, use_dim_dependencies);
}

// Adds dependencies of `op` to `dependencies` and for each dimension `di` of
// `op`, adds the dimensions `di` must be nested in to
// `dimension_dependencies[i]`.
static void GetDependencies(
    SairOp op, llvm::SmallVectorImpl<Dependency> &dependencies,
    llvm::SmallVectorImpl<llvm::SmallBitVector> &dimension_dependencies) {
  AccessPatternAttr access_pattern =
      AccessPatternAttr::GetIdentity(op.getContext(), op.domain().size());
  int domain_size = op.domain().size();
  llvm::SmallBitVector def_only_dimensions(domain_size);
  llvm::SmallBitVector use_only_dimensions(domain_size);
  llvm::SmallBitVector fuse_dimensions(domain_size);
  llvm::SmallBitVector prev_def_only_dimensions(domain_size);
  AddDependencies(op, access_pattern, def_only_dimensions, use_only_dimensions,
                  fuse_dimensions, prev_def_only_dimensions, false,
                  dependencies, dimension_dependencies);
}

// Checks that loop does not iterate along a dimension flagged in
// `dependencies`. `op` is the operation of `loop` and `other_op` is the other
// operation involved in the dependency.
static mlir::LogicalResult CheckLoopDependency(
    LoopAttr loop, const llvm::SmallBitVector &dependencies,
    mlir::Operation *op, mlir::Operation *other_op) {
  if (loop.iter().Rematerialize()) return mlir::success();
  int dimension = loop.iter().Dimension();
  if (!dependencies.test(dimension)) return mlir::success();
  return (op->emitError() << "operation cannot be nested in loop "
                          << loop.name())
             .attachNote(other_op->getLoc())
         << "because of this operation";
}

// Verifies that the dependency of `use` is satisfied.
static mlir::LogicalResult VerifyDependency(
    SairOp use, llvm::ArrayRef<mlir::Attribute> use_loop_nest,
    Dependency &dependency) {
  if (!dependency.def.loop_nest().hasValue()) return mlir::success();
  llvm::ArrayRef<mlir::Attribute> def_loop_nest =
      dependency.def.LoopNestLoops();

  int min_size = std::min(use_loop_nest.size(), def_loop_nest.size());
  for (int i = 0; i < min_size; ++i) {
    LoopAttr use_loop = use_loop_nest[i].cast<LoopAttr>();
    LoopAttr def_loop = def_loop_nest[i].cast<LoopAttr>();
    if (use_loop.name() != def_loop.name()) break;

    if (failed(CheckLoopDependency(use_loop, dependency.use_only_dimensions,
                                   dependency.def, use)) ||
        failed(CheckLoopDependency(def_loop, dependency.def_only_dimensions,
                                   use, dependency.def))) {
      return mlir::failure();
    }

    // Check mapped dimensions. If two dimensions are mapped by the access
    // pattern, the def dimension must be fused with the use dimension
    // or be in a separate loop nest.
    if (def_loop.iter().Rematerialize()) continue;
    int mapped_dimension =
        dependency.mapped_dimensions.Dimension(def_loop.iter().Dimension());
    if (mapped_dimension == AccessPatternAttr::kNoDimension) continue;
    if (use_loop.iter().Rematerialize() ||
        use_loop.iter().Dimension() != mapped_dimension) {
      return (use.emitError()
              << "loop " << use_loop.name() << " violates a data dependency")
                 .attachNote(dependency.def.getLoc())
             << "dependency from this operation";
    }
  }

  // Check that dimensions that must be fused are fused.
  int last_prev_def_only_dim = AccessPatternAttr::kNoDimension;
  for (int i = 0, e = def_loop_nest.size(); i < e; ++i) {
    LoopAttr def_loop = def_loop_nest[i].cast<LoopAttr>();
    if (def_loop.iter().Rematerialize()) continue;
    int def_dimension = def_loop.iter().Dimension();

    if (dependency.prev_def_only_dimensions.test(def_dimension)) {
      last_prev_def_only_dim = def_dimension;
    }

    int use_dimension = dependency.mapped_dimensions.Dimension(def_dimension);
    if (use_dimension == AccessPatternAttr::kNoDimension ||
        !dependency.carrying_dimensions.test(use_dimension)) {
      continue;
    }

    if (last_prev_def_only_dim != AccessPatternAttr::kNoDimension) {
      return dependency.def.emitError()
             << "dimension 'd" << last_prev_def_only_dim
             << "' must be nested in dimension 'd" << def_dimension << "'";
    }

    if (i < use_loop_nest.size()) {
      LoopAttr use_loop = use_loop_nest[i].cast<LoopAttr>();
      if (use_loop.name() == def_loop.name()) continue;
    }

    return (use.emitError()
            << "operation must be fused with loop " << def_loop.name())
               .attachNote(dependency.def.getLoc())
           << "because of a dependency from this operation";
  }

  return mlir::success();
}

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

mlir::LogicalResult VerifyLoopNests(SairProgramOp program) {
  // Verify loop nests are correct with regard to their operation.
  mlir::WalkResult result = program.walk([](ComputeOp op) -> mlir::WalkResult {
    if (!op.loop_nest().hasValue()) return mlir::success();
    return VerifyLoopNestWellFormed(
        cast<SairOp>(op.getOperation()), op.LoopNestLoops());
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Verify that the loop structure forms a tree.
  LoopNestState loop_nest_state;
  result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    if (!op.loop_nest().hasValue()) { return loop_nest_state.CloseLoops(); }
    return loop_nest_state.Update(cast<SairOp>(op.getOperation()),
                                  op.LoopNestLoops());
  });
  if (result.wasInterrupted()) return mlir::failure();
  if (mlir::failed(loop_nest_state.CloseLoops())) return mlir::failure();

  // Verify dependencies.
  result = program.walk([](ComputeOp op) -> mlir::WalkResult {
    if (!op.loop_nest().hasValue()) return mlir::success();
    SairOp sair_op = cast<SairOp>(op.getOperation());
    int domain_size = sair_op.domain().size();
    // Retrieve dependencies.
    llvm::SmallVector<Dependency, 4> dependencies;
    llvm::SmallVector<llvm::SmallBitVector, 4> dimension_dependencies(
        domain_size, llvm::SmallBitVector(domain_size));
    GetDependencies(sair_op, dependencies, dimension_dependencies);

    for (Dependency &dependency : dependencies) {
      if (mlir::failed(
              VerifyDependency(sair_op, op.LoopNestLoops(), dependency))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  });
  if (result.wasInterrupted()) return mlir::failure();

  return mlir::success();
}

}  // namespace sair
