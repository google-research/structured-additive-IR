// Copyright 2021 Google LLC
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

#include "mapped_domain.h"

#include "llvm/Support/Casting.h"
#include "loop_nest.h"

namespace sair {

MappedDomain::MappedDomain(mlir::Location loc, llvm::StringRef kind,
                           mlir::StringAttr name, const LoopNest &loop_nest)
    : AttrLocation(loc, kind, name) {
  llvm::append_range(domain_, loop_nest.getDomain());
  loop_nest_ = loop_nest.LoopNames();
  mapping_ = MappingAttr::get(context(), domain_.size(), {});
  loops_mapping_ = loop_nest.DomainToLoops();
}

DomainShapeAttr MappedDomain::DomainShape() const {
  llvm::SmallVector<DomainShapeDim> shape_dims;
  shape_dims.reserve(domain_.size());
  for (const ValueAccessInstance &access : domain_) {
    auto type = llvm::cast<DimensionType>(access.value.GetType());
    shape_dims.emplace_back(type, access.mapping);
  }
  return DomainShapeAttr::get(context(), shape_dims);
}

MappingAttr MappedDomain::NestedMapping() const {
  return mapping_.AddPrefix(loops_mapping_.Dimensions());
}

DomainShapeAttr MappedDomain::NestedShape() const {
  return DomainShape().AccessedShape(NestedMapping());
}

void MappedDomain::AddNonePrefixToMapping(int new_dimensions) {
  assert(new_dimensions >= 0);
  llvm::SmallVector<MappingExpr> exprs(new_dimensions,
                                       MappingNoneExpr::get(context()));
  mapping_ = mapping_.AddPrefix(exprs);
}

mlir::LogicalResult MappedDomain::ResolveUnification(
    const OpInstance &op, int dimension_id,
    const ValueAccessInstance &dimension, MappingExpr &constraint) {
  // Ignore placeholders.
  mlir::Operation *dim_op = dimension.value.defining_op().GetDuplicatedOp();
  if (isa<SairPlaceholderOp>(dim_op)) return mlir::success();

  if (constraint.isa<MappingNoneExpr, MappingUnknownExpr>()) {
    // If the dimension is new, extend the domain.
    constraint = MappingDimExpr::get(domain_.size(), context());
    assert(dimension.mapping.IsSurjective());
    domain_.push_back(dimension);
  } else if (auto dim_expr = llvm::dyn_cast<MappingDimExpr>(constraint)) {
    // If the dimension must be unified with an existing dimension, ensure that
    // they match.
    const ValueAccessInstance &old_dimension = domain_[dim_expr.dimension()];
    if (dimension.value != old_dimension.value ||
        dimension.mapping.Dimensions() != old_dimension.mapping.Dimensions()) {
      mlir::InFlightDiagnostic diag =
          op.EmitError() << "use of dimension d" << dimension_id << " in "
                         << *this << " does not match previous occurrences";
      diag.attachNote(location()) << "previous occurence here";
      return mlir::failure();
    }
  } else {
    // Only allow unification between plain dimensions.
    mlir::InFlightDiagnostic diag =
        op.EmitError() << "use of dimension d" << dimension_id << " in "
                       << *this
                       << " cannot be unified with previous occurences";
    diag.attachNote(location()) << "previous occurence here";
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult MappedDomain::UnifyMapping(
    const OpInstance &op, MappingAttr loop_nest_mapping,
    MappingAttr new_mapping,
    llvm::ArrayRef<ValueAccessInstance> new_mapping_domain) {
  // Compute unification constraints.
  auto none = MappingNoneExpr::get(context());
  llvm::SmallVector<MappingExpr> constraints(new_mapping_domain.size(), none);
  AssertSuccess(
      UnificationConstraints(loop_nest_mapping, loops_mapping_, constraints));
  if (mlir::failed(
          UnificationConstraints(new_mapping, mapping_, constraints))) {
    mlir::InFlightDiagnostic diag =
        op.EmitError() << *this << " cannot be unified with previous occurence";
    diag.attachNote(location()) << "previous occurence here";
    return mlir::failure();
  }

  // Resolve unification constraints and extend the domain.
  llvm::SmallBitVector new_mapping_dims = new_mapping.DependencyMask();
  for (int dimension_id : new_mapping_dims.set_bits()) {
    ValueAccessInstance dimension = new_mapping_domain[dimension_id];
    auto constraints_mapping = MappingAttr::get(
        context(), domain_.size(),
        llvm::ArrayRef(constraints).take_front(dimension_id));
    dimension.mapping = constraints_mapping.Compose(dimension.mapping);
    if (!dimension.mapping.IsSurjective()) {
      return op.EmitError()
             << *this << " mapping depends on loops it cannot be nested in";
    }

    if (mlir::failed(ResolveUnification(op, dimension_id, dimension,
                                        constraints[dimension_id]))) {
      return mlir::failure();
    }
  }

  // Apply unification.
  mapping_ = mapping_.ResizeUseDomain(domain_.size());
  auto constraint_mapping =
      MappingAttr::get(context(), domain_.size(), constraints);
  mapping_ = constraint_mapping.Compose(new_mapping).Unify(mapping_);
  assert(mapping_ != nullptr);
  return mlir::success();
}

void MappedDomain::SetLoopNest(const LoopNest &new_loop_nest) {
  assert(new_loop_nest.size() <= loop_nest_.size());
  assert(llvm::ArrayRef(new_loop_nest.LoopNames()) ==
         llvm::ArrayRef(loop_nest_).take_front(new_loop_nest.size()));
  loop_nest_.resize(new_loop_nest.size());
  loops_mapping_ = new_loop_nest.DomainToLoops();

  // Compute dimensions from the domain to keep.
  llvm::SmallBitVector to_keep = mapping_.DependencyMask();
  to_keep.set(0, new_loop_nest.getDomain().size());

  // Create a mapping to rename dimensions.
  auto none = MappingNoneExpr::get(context());
  llvm::SmallVector<MappingExpr> renaming(domain_.size(), none);

  // Remove dimensions that were only used by the old loop nest from the domain.
  llvm::SmallVector<ValueAccessInstance> new_domain;
  for (int dimension_id : to_keep.set_bits()) {
    renaming[dimension_id] = MappingDimExpr::get(new_domain.size(), context());
    ValueAccessInstance dimension = domain_[dimension_id];

    auto renaming_mapping =
        MappingAttr::get(context(), new_domain.size(),
                         llvm::ArrayRef(renaming).take_front(dimension_id));
    dimension.mapping = renaming_mapping.Compose(dimension.mapping);
    new_domain.push_back(dimension);
  }

  auto renaming_mapping =
      MappingAttr::get(context(), new_domain.size(), renaming);
  domain_ = std::move(new_domain);
  mapping_ = renaming_mapping.Compose(mapping_);
}

}  // namespace sair
