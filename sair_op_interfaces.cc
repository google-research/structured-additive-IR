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

#include "sair_op_interfaces.h"

#include <iterator>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_ops.h"
#include "sair_traits.h"
#include "sair_types.h"

namespace sair {

void EraseOperand(int position, llvm::StringRef segment_sizes_attribute_name,
                  mlir::Operation *op) {
  op->eraseOperand(position);
  // Find the corresponding segment.
  auto sizes_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
      segment_sizes_attribute_name);
  llvm::SmallVector<llvm::APInt, 4> sizes(sizes_attr.getIntValues());
  int total_operands = 0;
  for (int i = 0, e = sizes.size(); i < e; ++i) {
    int current = sizes[i].getLimitedValue();
    if (total_operands + current > position) {
      --sizes[i];
      break;
    }
  }
  // Update the segment.
  auto new_size_attr =
      mlir::DenseIntElementsAttr::get(sizes_attr.getType(), sizes);
  op->setAttr(segment_sizes_attribute_name, new_size_attr);
}

void AppendOperand(mlir::Value operand,
                   llvm::StringRef segment_sizes_attribute_name,
                   mlir::Operation *op) {
  llvm::SmallVector<mlir::Value, 4> operands = op->getOperands();
  operands.push_back(operand);
  op->setOperands(operands);
  // Update the last segment.
  auto sizes_attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(
      segment_sizes_attribute_name);
  llvm::SmallVector<llvm::APInt, 4> sizes(sizes_attr.getIntValues());
  ++sizes.back();
  auto new_size_attr =
      mlir::DenseIntElementsAttr::get(sizes_attr.getType(), sizes);
  op->setAttr(segment_sizes_attribute_name, new_size_attr);
}

void AppendAccessPattern(AccessPatternAttr access_pattern,
                         mlir::Operation *op) {
  auto old_attribute =
      op->getAttrOfType<mlir::ArrayAttr>(SairDialect::kAccessPatternAttrName);
  llvm::SmallVector<mlir::Attribute, 4> access_patterns(
      old_attribute.getAsRange<mlir::Attribute>());
  access_patterns.push_back(access_pattern);
  auto new_attribute = mlir::ArrayAttr::get(access_patterns, op->getContext());
  op->setAttr(SairDialect::kAccessPatternAttrName, new_attribute);
}

llvm::Optional<int> GetMemorySpace(int result, mlir::Operation *op) {
  llvm::Optional<mlir::ArrayAttr> array =
      cast<ValueProducerOp>(op).memory_space();
  if (!array.hasValue()) return llvm::None;
  mlir::IntegerAttr space = array.getValue()[result].dyn_cast<IntegerAttr>();
  if (space == nullptr) return llvm::None;
  return space.getInt();
}

llvm::Optional<int> GetMemorySpace(mlir::Value value) {
  assert(value.getType().isa<ValueType>());
  // Sair requires !sair.value operands to be defined by an operation in the
  // same block, ensuring that value.getDefiningOp() is well defined.
  ValueProducerOp producer = cast<ValueProducerOp>(value.getDefiningOp());
  for (int i = 0, e = producer.getOperation()->getNumResults(); i < e; ++i) {
    if (producer.getOperation()->getResult(i) == value) {
      return producer.GetMemorySpace(i);
    }
  }
  llvm_unreachable("value not found in the defining operation");
}

void SetMemorySpace(int result, llvm::Optional<int> memory_space,
                    mlir::Operation *op) {
  auto old_attribute =
      op->getAttrOfType<mlir::ArrayAttr>(ValueProducerOp::kMemorySpaceAttrName);
  llvm::SmallVector<mlir::Attribute, 4> memory_spaces;
  if (old_attribute == nullptr) {
    auto unit_attr = mlir::UnitAttr::get(op->getContext());
    memory_spaces.resize(op->getNumResults(), unit_attr);
  } else {
    memory_spaces.append(old_attribute.begin(), old_attribute.end());
  }
  if (memory_space.hasValue()) {
    memory_spaces[result] = mlir::IntegerAttr::get(
        mlir::IntegerType::get(64, op->getContext()), memory_space.getValue());
  } else {
    memory_spaces[result] = mlir::UnitAttr::get(op->getContext());
  }
  auto new_attribute = mlir::ArrayAttr::get(memory_spaces, op->getContext());
  op->setAttr(ValueProducerOp::kMemorySpaceAttrName, new_attribute);
}

// Returns the first loop of the loop_nest attribute of the operation, if any.
static LoopAttr FirstLoopOrNull(mlir::Operation *op) {
  ComputeOp compute_op = dyn_cast_or_null<ComputeOp>(op);
  if (compute_op == nullptr) return nullptr;
  if (!compute_op.loop_nest().hasValue()) return nullptr;
  llvm::ArrayRef<mlir::Attribute> loop_nest =
      compute_op.loop_nest().getValue().getValue();
  if (loop_nest.empty()) return nullptr;
  return loop_nest.front().dyn_cast<LoopAttr>();
}

mlir::LogicalResult VerifyValueProducerOp(mlir::Operation *operation) {
  ValueProducerOp op = cast<ValueProducerOp>(operation);
  // All results must be Sair values. This is not a user-facing error. It should
  // be verified by operations implementing `SairValueProducerOp`.
  assert(llvm::all_of(operation->getResultTypes(),
                      [](mlir::Type type) { return type.isa<ValueType>(); }));
  llvm::Optional<mlir::ArrayAttr> memory_space_attr = op.memory_space();
  if (!memory_space_attr.hasValue()) return mlir::success();
  llvm::ArrayRef<mlir::Attribute> memory_spaces =
      memory_space_attr.getValue().getValue();

  if (memory_spaces.size() != operation->getNumResults()) {
    return op.emitError()
           << "wrong number of entries for the memory_space attribute";
  }

  bool needs_allocation = false;
  for (int i = 0, e = memory_spaces.size(); i < e; ++i) {
    mlir::Attribute attr = memory_spaces[i];
    if (attr.isa<mlir::UnitAttr>()) continue;

    int memory_space = attr.cast<mlir::IntegerAttr>().getInt();
    ValueType type = operation->getResult(i).getType().cast<ValueType>();
    switch (memory_space) {
      case ValueProducerOp::kMemory:
        // TODO(ulysse): support lowering index values to memory.
        if (type.ElementType().isa<mlir::IndexType>()) {
          return op.emitError()
                 << "index variables cannot be allocated in memory";
        }
        needs_allocation = true;
        break;
      case ValueProducerOp::kRegister:
        if (!type.Shape().Is0d()) {
          // TODO(ulysse): consider the dimensionality of the layout instead,
          // once layout attributes are implemented.
          return op.emitError() << "only 0D values may be stored in registers";
        }
        break;
      default:
        return op.emitError() << "unexpected memory space";
    }
  }

  // Ensure that we can introduce the malloc between the producer of dimension
  // sizes and the current op.
  // TODO(ulysse): can we fold this in the generic interface for exposing
  // dependencies?
  LoopAttr first_loop = FirstLoopOrNull(op);
  if (!needs_allocation || first_loop == nullptr) return mlir::success();

  for (mlir::Value dimension : cast<SairOp>(operation).domain()) {
    SairRangeOp defining_op = dyn_cast<SairRangeOp>(dimension.getDefiningOp());
    if (defining_op == nullptr) continue;
    LoopAttr size_first_loop =
        FirstLoopOrNull(defining_op.size().getDefiningOp());
    if (first_loop.name() == size_first_loop.name()) {
      return op.emitError()
             << "operation cannot be nested in loop " << first_loop.name()
             << ": dimension sizes must be defined before entering the loop "
                "nest";
    }
  }

  return mlir::success();
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
      dependency.def.loop_nest().getValue().getValue();

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
  int last_pref_def_only_dim = AccessPatternAttr::kNoDimension;
  for (int i = 0, e = def_loop_nest.size(); i < e; ++i) {
    LoopAttr def_loop = def_loop_nest[i].cast<LoopAttr>();
    if (def_loop.iter().Rematerialize()) continue;
    int def_dimension = def_loop.iter().Dimension();

    if (dependency.prev_def_only_dimensions.test(def_dimension)) {
      last_pref_def_only_dim = def_dimension;
    }

    int use_dimension = dependency.mapped_dimensions.Dimension(def_dimension);
    if (use_dimension == AccessPatternAttr::kNoDimension ||
        !dependency.carrying_dimensions.test(use_dimension)) {
      continue;
    }

    if (last_pref_def_only_dim != AccessPatternAttr::kNoDimension) {
      return dependency.def.emitError() << "dimension 'd" << def_dimension
                                        << "' must be nested in dimension 'd"
                                        << last_pref_def_only_dim << "'";
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

mlir::LogicalResult VerifyComputeOp(mlir::Operation *op) {
  ComputeOp compute_op = cast<ComputeOp>(op);
  if (!compute_op.loop_nest().hasValue()) return mlir::success();
  llvm::ArrayRef<mlir::Attribute> loop_nest =
      compute_op.loop_nest().getValue().getValue();

  SairProgramOp parent = dyn_cast<SairProgramOp>(op->getParentOp());
  // Delegate checking that `parent` is a SairProgramOp to SairOp verifier.
  if (parent == nullptr) return mlir::success();
  SairOp sair_op = cast<SairOp>(op);
  int domain_size = sair_op.domain().size();

  // Retrieve dependencies.
  llvm::SmallVector<Dependency, 4> dependencies;
  llvm::SmallVector<llvm::SmallBitVector, 4> dimension_dependencies;
  dimension_dependencies.reserve(domain_size);
  for (const DomainShapeDim &dim : sair_op.shape().Dimensions()) {
    dimension_dependencies.push_back(dim.DependencyMask());
    dimension_dependencies.back().resize(domain_size);
  }
  GetDependencies(sair_op, dependencies, dimension_dependencies);

  // Bitfield that keeps track of which dimensions are implemented by loops.
  llvm::SmallBitVector covered_dimensions(domain_size);
  llvm::SmallVector<int, 8> steps(sair_op.domain().size(), -1);
  for (int i = 0, e = loop_nest.size(); i < e; ++i) {
    LoopAttr loop = loop_nest[i].cast<LoopAttr>();
    if (llvm::count(parent.loop_name_table(), loop.name()) == 0) {
      return op->emitError() << "loop " << loop.name()
                             << " is not declared in the parent operation";
    }

    // Ensure that symbols are unique in the loop nest.
    for (int j = 0; j < i; ++j) {
      if (loop.name() == loop_nest[j].cast<LoopAttr>().name()) {
        return op->emitError()
               << "name " << loop.name() << " used twice in the same loop nest";
      }
    }

    if (loop.iter().Rematerialize()) continue;
    int dimension = loop.iter().Dimension();

    llvm::SmallBitVector missing_outer_dims =
        dimension_dependencies[dimension] & ~covered_dimensions;
    if (missing_outer_dims.any()) {
      return op->emitError()
             << "dependencies of dimension 'd" << dimension << "' "
             << "must be nested in dimension 'd"
             << *missing_outer_dims.set_bits_begin() << "'";
    }

    int step = loop.iter().Step();
    // Mark the dimension as covered if the loop has step one. This ensure that
    // we iterate over the points of each dimensions, and not just the tiles.
    if (step == 1) covered_dimensions.set(dimension);

    if (steps[dimension] != -1 && steps[dimension] <= step) {
      return op->emitError() << "loop " << loop.name()
                             << " step must be less than in outer loops";
    }
    steps[dimension] = step;
  }

  if (!covered_dimensions.all()) {
    return op->emitError() << "not all dimensions are covered by the loop nest";
  }

  for (Dependency &dependency : dependencies) {
    if (mlir::failed(VerifyDependency(sair_op, loop_nest, dependency))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

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

void GetDependencies(
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

#include "sair_op_interfaces.cc.inc"

}  // namespace sair
