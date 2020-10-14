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
#include "utils.h"

namespace sair {

AccessPatternAttr ValueOperand::AccessPattern() {
  return cast<SairOp>(operand_->getOwner()).access_pattern_array()
      .getValue()[index_]
      .template cast<::sair::AccessPatternAttr>();
}

void ValueOperand::SetAccessPattern(AccessPatternAttr access_pattern) {
  SairOp op = cast<SairOp>(operand_->getOwner());
  op.SetAccessPattern(index_, access_pattern);
}

ValueOperandRange::ValueOperandRange()
    : RangeBaseT(std::make_pair(nullptr, 0), 0) {}

ValueOperandRange::ValueOperandRange(
    llvm::MutableArrayRef<mlir::OpOperand> operands)
    : RangeBaseT(std::make_pair(operands.data(), 0), operands.size()) {}

ValueOperandRange::PtrPair ValueOperandRange::offset_base(PtrPair base_ptr,
                                                          ptrdiff_t offset) {
  base_ptr.first += offset;
  base_ptr.second += offset;
  return base_ptr;
}

ValueOperand ValueOperandRange::dereference_iterator(PtrPair base_ptr,
                                                     ptrdiff_t offset) {
  return ValueOperand(base_ptr.first + offset, base_ptr.second + offset);
}

llvm::SmallBitVector ValueOperand::DimsDependingOnOperand() const {
  return cast<SairOp>(operand_->getOwner()).DimsDependingOnOperand(index_);
}

bool ValueOperand::AllowUseBeforeDef() const {
  return cast<SairOp>(operand_->getOwner()).AllowUseBeforeDef(index_);
}

// Sair operations are only allowed inside a SairProgramOp.
mlir::LogicalResult VerifySairOpParent(mlir::Operation *operation) {
  if (isa<SairProgramOp>(operation->getParentOp())) {
    return mlir::success();
  }

  return operation->emitOpError()
         << "expected to be immediately contained in a '"
         << SairProgramOp::getOperationName() << "'";
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
    appendRange(memory_spaces, old_attribute.getValue());
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

mlir::LogicalResult VerifySairOp(Operation *op) {
  SairOp sair_op = cast<SairOp>(op);
  // Check that the domain has the right shape.
  if (llvm::size(sair_op.domain()) != sair_op.shape().NumDimensions()) {
    return sair_op.emitError("unexpected number of dimensions");
  }
  for (auto pair :
       llvm::zip(sair_op.domain(), sair_op.shape().Dimensions())) {
    if (std::get<0>(pair).getType() != std::get<1>(pair).type()) {
      return sair_op.emitError("unexpected dimension type");
    }
  }
  // Check that the domain is defined locally.
  for (mlir::Value dimension : sair_op.domain()) {
    mlir::Operation *defining_op = dimension.getDefiningOp();
    if (!defining_op ||
        defining_op->getParentRegion() != op->getParentRegion()) {
      return op->emitError()
             << "sair dimensions must be defined in the region they are used";
    }
    if (!defining_op->isBeforeInBlock(op)) {
      return (op->emitError() << "dimension used before its definition")
                 .attachNote(defining_op->getLoc())
             << "definition here";
    }
  }
  // Check that operands start with the domain.
  if (!sair_op.domain().empty() &&
      sair_op.domain().begin() != op->operand_begin()) {
    return sair_op.emitError()
           << "expected operands to start with the domain";
  }
  // Check that there is enough operands.
  int min_num_operands =
      sair_op.shape().NumDimensions() + sair_op.access_pattern_array().size();
  if (op->getNumOperands() < min_num_operands) {
    return sair_op.emitError() << "unexpected number of operands";
  }

  if (!sair_op.ValueOperands().empty()) {
    // Verify that the "access_pattern_array" attribute exists.
    if (!op->getAttr(::sair::SairDialect::kAccessPatternAttrName)) {
      return mlir::emitError(op->getLoc())
             << "missing " << ::sair::SairDialect::kAccessPatternAttrName
             << " attribute";
    }
    for (mlir::Attribute pattern : sair_op.access_pattern_array()) {
      if (pattern.cast<::sair::AccessPatternAttr>().DependsOnDimension(
              ::sair::AccessPatternAttr::kNoDimension)) {
        return mlir::emitError(op->getLoc())
               << "all dimensions of the accessed domain must be mapped";
      }
    }
  }

  // Check !sair.value operands.
  for (::sair::ValueOperand v : sair_op.ValueOperands()) {
    auto value_type = v.GetType().template dyn_cast<::sair::ValueType>();
    if (!value_type) {
      return mlir::emitError(v.value().getLoc())
             << "expected a !sair.value operand";
    }
    if (v.AccessPattern().UseDomainSize() != sair_op.domain().size()) {
      return mlir::emitError(op->getLoc()) << "invalid use domain size";
    }
    ::sair::DomainShapeAttr expected_shape =
        sair_op.shape().Inverse(v.AccessPattern());
    if (expected_shape != value_type.Shape()) {
      return mlir::emitError(v.value().getLoc())
             << "access pattern incompatible with the operand shape";
    }
    mlir::Operation *defining_op = v.value().getDefiningOp();
    if (!defining_op ||
        defining_op->getParentRegion() != op->getParentRegion()) {
      return op->emitError()
             << "sair values must be defined in the region they are used";
    }
    if (!defining_op->isBeforeInBlock(op) && !v.AllowUseBeforeDef()) {
      return (op->emitError() << "operand used before its definition")
                 .attachNote(defining_op->getLoc())
             << "definition here";
    }

    llvm::SmallBitVector dependency_mask = v.AccessPattern().DependencyMask();
    if (dependency_mask.anyCommon(v.DimsDependingOnOperand())) {
      return op->emitError() << "an operand access pattern references a "
                                "dimension that depends on the operand";
    }
  }

  // Check that returned Sair values have the right shape.
  ::sair::DomainShapeAttr results_shape =
      sair_op.shape().Prefix(sair_op.results_rank());
  for (mlir::Value result : op->getResults()) {
    ::sair::SairShapedType type =
        result.getType().dyn_cast<::sair::SairShapedType>();
    if (type == nullptr) continue;
    if (type.Shape() != results_shape) {
      return op->emitError() << "unexpected shape: expected " << results_shape
                             << ", got " << type.Shape();
    }
  }

  return ::sair::VerifySairOpParent(sair_op);
}

// Returns the first loop of the loop_nest attribute of the operation, if any.
static LoopAttr FirstLoopOrNull(mlir::Operation *op) {
  ComputeOp compute_op = dyn_cast_or_null<ComputeOp>(op);
  if (compute_op == nullptr) return nullptr;
  if (!compute_op.loop_nest().hasValue()) return nullptr;
  if (compute_op.LoopNestLoops().empty()) return nullptr;
  return compute_op.LoopNestLoops().front().dyn_cast<LoopAttr>();
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
    SairDynRangeOp defining_op =
        dyn_cast<SairDynRangeOp>(dimension.getDefiningOp());
    if (defining_op == nullptr) continue;
    auto is_producer_fused = [&](mlir::Value value) {
      if (value == nullptr) return false;
      LoopAttr loop = FirstLoopOrNull(value.getDefiningOp());
      if (loop == nullptr) return false;
      return first_loop.name() == loop.name();
    };
    if (is_producer_fused(defining_op.lower_bound()) ||
        is_producer_fused(defining_op.upper_bound())) {
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
  llvm::ArrayRef<mlir::Attribute> loop_nest = compute_op.LoopNestLoops();

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

mlir::LogicalResult VerifyRangeOp(mlir::Operation *op) {
  RangeOp range_op = cast<RangeOp>(op);
  if (!range_op.step().isStrictlyPositive()) {
    return range_op.emitError() << "step must be strictly positive";
  }
  return mlir::success();
}

#include "sair_op_interfaces.cc.inc"

}  // namespace sair
