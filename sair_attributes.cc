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

#include "sair_attributes.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "sair_op_interfaces.h"

namespace sair {

#include "sair_attr_interfaces.cc.inc"

mlir::Value MapArguments::AddArgument(ValueAccess value) {
  values_.push_back(value.value);
  mappings_.push_back(value.mapping);
  return body_->addArgument(value.ElementType());
}

mlir::OpFoldResult MapArguments::AddArgument(ValueOrConstant value) {
  if (value.is_constant()) return value.constant();
  return AddArgument(value.value());
}

mlir::ValueRange MapArguments::Indices() const {
  return body_->getArguments().take_front(domain_size_);
}

//===----------------------------------------------------------------------===//
// MappingDimExpr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::MappingDimExpr.
// Instances of this class are allocate by MLIR type system in a dedicated
// arena. Not intended for direct use.
class impl::MappingDimExprStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies MappingDimExpr for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = int;

  // Creates a MappingDimExprStorage using the provided allocator. Hook
  // for MLIR attribute system.
  static MappingDimExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<MappingDimExprStorage>())
        MappingDimExprStorage(key);
  }

  // Compares the MappingDimExprStorage identification key with this
  // object.
  bool operator==(const KeyTy &key) const { return key == dimension_; }

  // Returns the dimension represented by the operation.
  int dimension() const { return dimension_; }

 private:
  // Constructs a storage object for the provided key. such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit MappingDimExprStorage(KeyTy key) : dimension_(key) {}

  int dimension_;
};

MappingDimExpr MappingDimExpr::get(int dimension, mlir::MLIRContext *context) {
  return Base::get(context, dimension);
}

int MappingDimExpr::dimension() const { return getImpl()->dimension(); }

MappingExpr MappingDimExpr::SubstituteDims(
    mlir::ArrayRef<MappingExpr> exprs) const {
  if (dimension() >= exprs.size()) {
    return MappingNoneExpr::get(getContext());
  }
  return exprs[dimension()];
}

DomainShapeDim MappingDimExpr::AccessedShape(
    llvm::ArrayRef<DomainShapeDim> accessing_shape,
    MappingAttr inverted_mapping) const {
  return accessing_shape[dimension()].Apply(inverted_mapping);
}

mlir::LogicalResult MappingDimExpr::SetInverse(
    MappingExpr context_inverse,
    llvm::MutableArrayRef<MappingExpr> inverses) const {
  MappingExpr inverse = inverses[dimension()].Unify(context_inverse);
  if (inverse == nullptr) return mlir::failure();
  inverses[dimension()] = inverse;
  return mlir::success();
}

MappingExpr MappingDimExpr::Unify(MappingExpr other_expr) const {
  if (other_expr == *this) return *this;
  if (other_expr.isa<MappingNoneExpr>()) return *this;
  return MappingExpr();
}

mlir::LogicalResult MappingDimExpr::UnificationConstraints(
    MappingExpr other_expr,
    llvm::MutableArrayRef<MappingExpr> constraints) const {
  if (other_expr.isa<MappingNoneExpr>()) return mlir::success();

  MappingExpr &constraint = constraints[dimension()];
  if (constraint == other_expr) return mlir::success();
  if (!constraint.isa<MappingNoneExpr>()) return mlir::failure();
  constraint = other_expr;
  return mlir::success();
}

mlir::AffineExpr MappingDimExpr::AsAffineExpr() const {
  return mlir::getAffineDimExpr(dimension(), getContext());
}

RangeParameters MappingDimExpr::GetRangeParameters(
    mlir::Location loc, mlir::ValueRange domain, DomainShapeAttr shape,
    MappingAttr inverse_mapping, mlir::OpBuilder &builder,
    MapArguments &map_arguments) const {
  auto range_op = mlir::cast<RangeOp>(domain[dimension()].getDefiningOp());
  auto mapping = shape.Dimension(dimension())
                     .dependency_mapping()
                     .ResizeUseDomain(map_arguments.Indices().size());
  assert(mapping.IsFullySpecified());
  return {
      .begin = map_arguments.AddArgument(range_op.LowerBound().Map(mapping)),
      .end = map_arguments.AddArgument(range_op.UpperBound().Map(mapping)),
      .step = static_cast<int>(range_op.step().getSExtValue())};
}

//===----------------------------------------------------------------------===//
// MappingNoneExpr
//===----------------------------------------------------------------------===//

MappingNoneExpr MappingNoneExpr::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

MappingExpr MappingNoneExpr::MakeFullySpecified(int &num_dimensions) const {
  return MappingDimExpr::get(num_dimensions++, getContext());
}

//===----------------------------------------------------------------------===//
// MappingStripeExpr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::MappingStripeExpr.
// Instances of this class are allocate by MLIR type system in a dedicated
// arena. Not intended for direct use.
class impl::MappingStripeExprStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies MappingStripeExpr for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = std::tuple<mlir::Attribute, int, int>;

  // Creates a MappingStripeExprStorage using the provided allocator. Hook
  // for MLIR attribute system.
  static MappingStripeExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<MappingStripeExprStorage>())
        MappingStripeExprStorage(key);
  }

  // Compares the MappingStripeExpr identification key with this object.
  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == operand_ && std::get<1>(key) == step_ &&
           std::get<2>(key) == size_;
  }

  // The striped expression.
  MappingExpr operand() const { return operand_; }

  // The stripe step. This is one for point expressions.
  int step() const { return step_; }

  // The expression range. This is `None` for outermost stripe expressions.
  llvm::Optional<int> size() const {
    return size_ == kNoSize ? llvm::Optional<int>() : size_;
  }

  // Internal encoding for a `None` size.
  static constexpr int kNoSize = 0;

 private:
  // Constructs a storage object for the provided key. such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit MappingStripeExprStorage(KeyTy key)
      : operand_(std::get<0>(key)),
        step_(std::get<1>(key)),
        size_(std::get<2>(key)) {}

  MappingExpr operand_;
  int step_;
  int size_;
};

MappingStripeExpr MappingStripeExpr::get(MappingExpr operand, int step,
                                         llvm::Optional<int> size) {
  assert(operand != nullptr);
  assert(!size.hasValue() || size.getValue() >= step);
  int int_size = size.hasValue() ? size.getValue()
                                 : impl::MappingStripeExprStorage::kNoSize;
  return Base::get(operand.getContext(),
                   std::make_tuple(operand, step, int_size));
}

MappingExpr MappingStripeExpr::operand() const { return getImpl()->operand(); }

int MappingStripeExpr::step() const { return getImpl()->step(); }

llvm::Optional<int> MappingStripeExpr::size() const {
  return getImpl()->size();
}

MappingExpr MappingStripeExpr::MakeFullySpecified(int &num_dimensions) const {
  return MappingStripeExpr::get(operand().MakeFullySpecified(num_dimensions),
                                step(), size());
}

MappingExpr MappingStripeExpr::SubstituteDims(
    llvm::ArrayRef<MappingExpr> exprs) const {
  return MappingStripeExpr::get(operand().SubstituteDims(exprs), step(),
                                size());
}

MappingExpr MappingStripeExpr::Unify(MappingExpr other_expr) const {
  if (other_expr.isa<MappingNoneExpr>()) return *this;
  MappingStripeExpr other_stripe = other_expr.dyn_cast<MappingStripeExpr>();
  if (other_stripe == nullptr || size() != other_stripe.size() ||
      step() != other_stripe.step()) {
    return MappingExpr();
  }
  MappingExpr unified_operand = operand().Unify(other_stripe.operand());
  if (unified_operand == nullptr) return MappingExpr();
  return MappingStripeExpr::get(unified_operand, step(), size());
}

mlir::LogicalResult MappingStripeExpr::UnificationConstraints(
    MappingExpr other_expr,
    llvm::MutableArrayRef<MappingExpr> constraints) const {
  if (other_expr.isa<MappingNoneExpr>()) return mlir::success();
  auto other_stripe = other_expr.dyn_cast<MappingStripeExpr>();
  if (other_stripe == nullptr || size() != other_stripe.size() ||
      step() != other_stripe.step()) {
    return mlir::failure();
  }
  return operand().UnificationConstraints(other_stripe.operand(), constraints);
}

DomainShapeDim MappingStripeExpr::AccessedShape(
    llvm::ArrayRef<DomainShapeDim> accessing_shape,
    MappingAttr inverted_mapping) const {
  mlir::MLIRContext *context = getContext();

  DomainShapeDim inner_shape =
      operand().AccessedShape(accessing_shape, inverted_mapping);
  auto inverse_subexpr = operand()
                             .FindInInverse(inverted_mapping.Dimensions())
                             .cast<MappingUnStripeExpr>();

  // Append dependencies to larger stripes to the dependency mapping.
  llvm::SmallVector<MappingExpr, 4> dependency_mapping_exprs;
  llvm::append_range(dependency_mapping_exprs,
                     inner_shape.dependency_mapping());

  llvm::SmallVector<DomainShapeDim, 4> type_shape;
  llvm::append_range(type_shape, inner_shape.type().Shape().Dimensions());
  RangeType type = inner_shape.type();

  for (auto [expr, step] :
       llvm::zip(inverse_subexpr.operands(), inverse_subexpr.factors())) {
    if (step == this->step()) break;
    type_shape.emplace_back(
        type, MappingAttr::GetIdentity(context, type_shape.size()));
    type = RangeType::get(DomainShapeAttr::get(context, type_shape));
    dependency_mapping_exprs.push_back(expr);
  }

  auto dependency_mapping = MappingAttr::get(
      context, inverted_mapping.UseDomainSize(), dependency_mapping_exprs);
  return DomainShapeDim(type, dependency_mapping);
}

mlir::LogicalResult MappingStripeExpr::SetInverse(
    MappingExpr context_inverse,
    llvm::MutableArrayRef<MappingExpr> inverses) const {
  MappingExpr none = MappingNoneExpr::get(getContext());
  llvm::SmallVector<MappingExpr, 3> operands;
  llvm::SmallVector<int, 2> factors;

  if (size().hasValue()) {
    operands.push_back(none);
    factors.push_back(size().getValue());
  }
  operands.push_back(context_inverse);
  if (step() != 1) {
    operands.push_back(none);
    factors.push_back(step());
  }

  return operand().SetInverse(MappingUnStripeExpr::get(operands, factors),
                              inverses);
}

MappingExpr MappingStripeExpr::FindInInverse(
    llvm::ArrayRef<MappingExpr> inverse) const {
  auto unstripe_expr =
      operand().FindInInverse(inverse).cast<MappingUnStripeExpr>();
  auto factor_it = llvm::find(unstripe_expr.factors(), step());
  int pos = std::distance(unstripe_expr.factors().begin(), factor_it);
  return unstripe_expr.operands()[pos];
}

mlir::AffineExpr MappingStripeExpr::AsAffineExpr() const {
  return step() * operand().AsAffineExpr().floorDiv(step());
}

MappingExpr MappingStripeExpr::Canonicalize() const {
  MappingExpr new_operand = operand().Canonicalize();
  // Use a lambda so that we can break from the control flow.
  auto simplify = [&]() {
    auto unstripe = new_operand.dyn_cast<MappingUnStripeExpr>();
    if (unstripe == nullptr) return MappingExpr();
    llvm::ArrayRef<int> factors = unstripe.factors();
    auto it = llvm::find(factors, step());
    if (it == factors.end() && step() != 1) return MappingExpr();
    if (it == factors.begin() && size().hasValue()) return MappingExpr();
    if (it != factors.begin() && size() != *std::prev(it)) return MappingExpr();
    return unstripe.operands()[it - factors.begin()];
  };

  MappingExpr simplified = simplify();
  if (simplified != nullptr) return simplified;
  return MappingStripeExpr::get(new_operand, step(), size());
}

RangeParameters MappingStripeExpr::GetRangeParameters(
    mlir::Location loc, mlir::ValueRange domain, DomainShapeAttr shape,
    MappingAttr inverse_mapping, mlir::OpBuilder &builder,
    MapArguments &map_arguments) const {
  // Compute range parameters for the operand.
  RangeParameters operand_parameters = operand().GetRangeParameters(
      loc, domain, shape, inverse_mapping, builder, map_arguments);
  int step = this->step() * operand_parameters.step;

  // If the stripe covers the entire operand range, no additional computation is
  // needed.
  if (!size().hasValue()) {
    return {operand_parameters.begin, operand_parameters.end, step};
  }

  // Compute the begin index. For this, look for the unstripe operation
  // corresponding to `this` in the inverse mapping, and find the expression of
  // the outer stripe dimension.
  auto inverse_expr = operand()
                          .FindInInverse(inverse_mapping.Dimensions())
                          .cast<MappingUnStripeExpr>();
  int inverse_pos = llvm::find(inverse_expr.factors(), size().getValue()) -
                    inverse_expr.factors().begin();
  auto begin_map =
      mlir::AffineMap::get(map_arguments.Indices().size(), 0,
                           inverse_expr.operands()[inverse_pos].AsAffineExpr());
  mlir::Value begin = builder.create<mlir::AffineApplyOp>(
      loc, begin_map, map_arguments.Indices());

  // Compute the end index as `min(begin + size, operand_size)`.
  mlir::Type index_type = builder.getIndexType();
  auto size = builder.create<mlir::ConstantOp>(
      loc, index_type,
      builder.getIndexAttr(this->size().getValue() * operand_parameters.step));
  auto uncapped_end =
      builder.create<mlir::AddIOp>(loc, index_type, begin, size);
  mlir::Value operand_end;
  if (operand_parameters.end.is<mlir::Attribute>()) {
    operand_end = builder.create<mlir::ConstantOp>(
        loc, index_type, operand_parameters.end.get<mlir::Attribute>());
  } else {
    operand_end = operand_parameters.end.get<mlir::Value>();
  }
  auto is_capped = builder.create<mlir::CmpIOp>(loc, CmpIPredicate::ult,
                                                operand_end, uncapped_end);
  mlir::Value end = builder.create<mlir::SelectOp>(
      loc, builder.getIndexType(), is_capped, operand_end, uncapped_end);

  return {begin, end, step};
}

//===----------------------------------------------------------------------===//
// MappingUnStripeExpr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::MappingUnStripeExpr.
// Instances of this class are allocate by MLIR type system in a dedicated
// arena. Not intended for direct use.
class impl::MappingUnStripeExprStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies MappingUnStripeExpr for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = std::pair<llvm::ArrayRef<MappingExpr>, llvm::ArrayRef<int>>;

  // Creates a MappingUnStripeExprStorage using the provided allocator.
  // Hook for MLIR attribute system.
  static MappingUnStripeExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<MappingUnStripeExprStorage>())
        MappingUnStripeExprStorage(allocator.copyInto(key.first),
                                   allocator.copyInto(key.second));
  }

  // Compares the MappingUnStripeExprStorage identification key with this
  // object.
  bool operator==(const KeyTy &key) const {
    return key.first == operands_ && key.second == factors_;
  }

  // Stripe expressions that are combined to obtain the unstriped expression.
  llvm::ArrayRef<MappingExpr> operands() const { return operands_; }

  // Stripe expression sizes.
  llvm::ArrayRef<int> factors() const { return factors_; }

 private:
  // Constructs a storage object for the provided key. such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit MappingUnStripeExprStorage(llvm::ArrayRef<MappingExpr> operands,
                                      llvm::ArrayRef<int> factors)
      : operands_(operands), factors_(factors) {}

  llvm::ArrayRef<MappingExpr> operands_;
  llvm::ArrayRef<int> factors_;
};

MappingUnStripeExpr MappingUnStripeExpr::get(
    llvm::ArrayRef<MappingExpr> stripes, llvm::ArrayRef<int> factors) {
  assert(factors.empty() || factors[0] > 1);
  assert(stripes.size() == factors.size() + 1);
#ifndef NDEBUG
  for (int i = 0; i + 1 < factors.size(); ++i) {
    assert(factors[i] > factors[i + 1]);
  }
#endif
  return Base::get(stripes[0].getContext(), std::make_pair(stripes, factors));
}

llvm::ArrayRef<MappingExpr> MappingUnStripeExpr::operands() const {
  return getImpl()->operands();
}

llvm::ArrayRef<int> MappingUnStripeExpr::factors() const {
  return getImpl()->factors();
}

bool MappingUnStripeExpr::IsFullySpecified() const {
  return llvm::all_of(operands(),
                      [](MappingExpr expr) { return expr.IsFullySpecified(); });
}

void MappingUnStripeExpr::SetDependenciesInMask(
    llvm::SmallBitVector &mask) const {
  for (MappingExpr expr : operands()) {
    expr.SetDependenciesInMask(mask);
  }
}

int MappingUnStripeExpr::MinDomainSize() const {
  int max = 0;
  for (MappingExpr expr : operands()) {
    max = std::max(max, expr.MinDomainSize());
  }
  return max;
}

MappingExpr MappingUnStripeExpr::MakeFullySpecified(int &num_dimensions) const {
  llvm::SmallVector<MappingExpr, 4> new_exprs;
  new_exprs.reserve(operands().size());
  for (MappingExpr expr : operands()) {
    new_exprs.push_back(expr.MakeFullySpecified(num_dimensions));
  }
  return MappingUnStripeExpr::get(new_exprs, factors());
}

MappingExpr MappingUnStripeExpr::SubstituteDims(
    llvm::ArrayRef<MappingExpr> exprs) const {
  llvm::SmallVector<MappingExpr, 4> new_exprs;
  new_exprs.reserve(operands().size());
  for (MappingExpr expr : operands()) {
    new_exprs.push_back(expr.SubstituteDims(exprs));
  }
  return MappingUnStripeExpr::get(new_exprs, factors());
}

DomainShapeDim MappingUnStripeExpr::AccessedShape(
    llvm::ArrayRef<DomainShapeDim> accessing_shape,
    MappingAttr inverted_mapping) const {
  // The shape of the unstriped dimension is the shape of the outer-most striped
  // dimensions. Other striped dimensions have similar shape, but with
  // additional dependencies to outer striped dimensions.
  return operands().front().AccessedShape(accessing_shape, inverted_mapping);
}

mlir::LogicalResult MappingUnStripeExpr::SetInverse(
    MappingExpr context_inverse,
    llvm::MutableArrayRef<MappingExpr> inverses) const {
  for (int i = 0, e = factors().size(); i <= e; ++i) {
    int step = i == e ? 1 : factors()[i];
    llvm::Optional<int> size =
        i == 0 ? llvm::Optional<int>() : factors()[i - 1];
    auto stripe_expr = MappingStripeExpr::get(context_inverse, step, size);
    if (mlir::failed(operands()[i].SetInverse(stripe_expr.cast<MappingExpr>(),
                                              inverses))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

MappingExpr MappingUnStripeExpr::Unify(MappingExpr other_expr) const {
  if (other_expr.isa<MappingNoneExpr>()) return *this;
  MappingUnStripeExpr other_unstripe =
      other_expr.dyn_cast<MappingUnStripeExpr>();
  if (other_unstripe == nullptr) return MappingExpr();

  llvm::SmallVector<MappingExpr, 4> new_exprs;
  llvm::SmallVector<int, 3> new_factors;

  int this_cursor = 0;
  int other_cursor = 0;
  while (this_cursor < operands().size()) {
    int this_step = this_cursor < factors().size() ? factors()[this_cursor] : 1;
    int other_step = other_cursor < other_unstripe.factors().size()
                         ? other_unstripe.factors()[other_cursor]
                         : 1;
    if (this_step < other_step) {
      // We can only subdivide none exprs.
      if (!operands()[this_cursor].isa<MappingNoneExpr>()) {
        return MappingExpr();
      }
      new_factors.push_back(other_step);
      new_exprs.push_back(other_unstripe.operands()[other_cursor]);
      ++other_cursor;
    } else if (this_step > other_step) {
      // We can only subdivide none exprs.
      if (!other_unstripe.operands()[other_cursor].isa<MappingNoneExpr>()) {
        return MappingExpr();
      }
      new_factors.push_back(this_step);
      new_exprs.push_back(operands()[this_cursor]);
      ++this_cursor;
    } else {
      if (this_step > 1) {
        new_factors.push_back(this_step);
      }
      new_exprs.push_back(operands()[this_cursor].Unify(
          other_unstripe.operands()[other_cursor]));
      ++this_cursor;
      ++other_cursor;
    }
  }
  return MappingUnStripeExpr::get(new_exprs, new_factors);
}

mlir::LogicalResult MappingUnStripeExpr::UnificationConstraints(
    MappingExpr other_expr,
    llvm::MutableArrayRef<MappingExpr> constraints) const {
  if (other_expr.isa<MappingNoneExpr>()) return mlir::success();
  MappingUnStripeExpr other_unstripe =
      other_expr.dyn_cast<MappingUnStripeExpr>();
  if (other_unstripe == nullptr) return mlir::failure();

  int this_cursor = 0;
  int other_cursor = 0;
  while (this_cursor < operands().size()) {
    int this_step = this_cursor < factors().size() ? factors()[this_cursor] : 1;
    int other_step = other_cursor < other_unstripe.factors().size()
                         ? other_unstripe.factors()[other_cursor]
                         : 1;
    if (this_step < other_step) {
      // We can only subdivide none exprs.
      if (!operands()[this_cursor].isa<MappingNoneExpr>()) {
        return mlir::failure();
      }
      ++other_cursor;
    } else if (this_step > other_step) {
      // We can only subdivide none exprs.
      if (!other_unstripe.operands()[other_cursor].isa<MappingNoneExpr>()) {
        return mlir::failure();
      }
      ++this_cursor;
    } else {
      if (mlir::failed(operands()[this_cursor].UnificationConstraints(
              other_unstripe.operands()[other_cursor], constraints))) {
        return mlir::failure();
      }
      ++this_cursor;
      ++other_cursor;
    }
  }
  return mlir::success();
}

MappingExpr MappingUnStripeExpr::FindInInverse(
    llvm::ArrayRef<MappingExpr> inverse) const {
  return operands()[0]
      .FindInInverse(inverse)
      .cast<MappingStripeExpr>()
      .operand();
}

mlir::AffineExpr MappingUnStripeExpr::AsAffineExpr() const {
  return operands().back().AsAffineExpr();
}

MappingExpr MappingUnStripeExpr::Canonicalize() const {
  llvm::SmallVector<MappingExpr, 4> new_operands;
  new_operands.reserve(operands().size());
  for (MappingExpr operand : operands()) {
    new_operands.push_back(operand.Canonicalize());
  }

  // Use a lambda so that we can easily break the control flow.
  auto simplify = [&]() -> MappingExpr {
    auto stripe = new_operands.front().dyn_cast<MappingStripeExpr>();
    if (stripe == nullptr) return MappingExpr();
    if (stripe.size().hasValue()) return MappingExpr();
    auto inner_stripes = llvm::makeArrayRef(new_operands).drop_front();
    for (auto [expr, factor] : llvm::zip(inner_stripes, factors())) {
      if (factor != stripe.step()) return MappingExpr();
      auto new_stripe = expr.dyn_cast<MappingStripeExpr>();
      if (new_stripe == nullptr) return MappingExpr();
      if (new_stripe.size() != factor) return MappingExpr();
      if (new_stripe.operand() != stripe.operand()) return MappingExpr();
      stripe = new_stripe;
    }
    return stripe.operand();
  };

  MappingExpr simplified = simplify();
  if (simplified != nullptr) return simplified;
  return MappingUnStripeExpr::get(new_operands, factors());
}

RangeParameters MappingUnStripeExpr::GetRangeParameters(
    mlir::Location loc, mlir::ValueRange domain, DomainShapeAttr shape,
    MappingAttr inverse_mapping, mlir::OpBuilder &builder,
    MapArguments &map_arguments) const {
  RangeParameters inner_parameters = operands()[0].GetRangeParameters(
      loc, domain, shape, inverse_mapping, builder, map_arguments);
  inner_parameters.step = 1;
  return inner_parameters;
}

//===----------------------------------------------------------------------===//
// MappingAttr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::MappingAttr. Instances
// of this class are allocated by MLIR type system in a dedicated arena. Not
// intended for direct use.
class impl::MappingAttrStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifying MappingAttrStorage for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = std::pair<int, llvm::ArrayRef<MappingExpr>>;

  // Creates an MappingAttrStorage using the provided allocator. Hook for
  // MLIR attribute system.
  static MappingAttrStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<MappingAttrStorage>()) MappingAttrStorage(
        std::make_pair(key.first, allocator.copyInto(key.second)));
  }

  // Compares the MappingAttrStorage identification key with this object.
  bool operator==(const KeyTy &key) const {
    return key.first == use_domain_size_ && key.second == mapping_;
  }

  // Returns the number of dimensions in the use domain.
  int use_domain_size() const { return use_domain_size_; }

  // Returns the list of dimensions along which a variable is accessed.
  // Dimensions are identified by their position in the domain definition.
  llvm::ArrayRef<MappingExpr> mapping() const { return mapping_; }

 private:
  // Constructs a storage object from the provided key. Such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit MappingAttrStorage(KeyTy key)
      : use_domain_size_(key.first), mapping_(key.second) {}

  int use_domain_size_;

  // The list of dimensions along which a Sair variable is accessed.
  llvm::ArrayRef<MappingExpr> mapping_;
};

// Verifies that the expressions form a valid access mapping.
static mlir::LogicalResult VerifyMappingExprs(
    mlir::MLIRContext *context, int use_domain_size,
    llvm::ArrayRef<MappingExpr> mapping_exprs) {
  llvm::SmallVector<MappingExpr, 4> inverted_exprs(
      use_domain_size, MappingNoneExpr::get(context));
  for (int i = 0, e = mapping_exprs.size(); i < e; ++i) {
    MappingExpr dim_expr = MappingDimExpr::get(i, context);
    if (mlir::failed(mapping_exprs[i].SetInverse(dim_expr, inverted_exprs))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

MappingAttr MappingAttr::get(mlir::MLIRContext *context, int use_domain_size,
                             llvm::ArrayRef<MappingExpr> mapping) {
  assert(
      mlir::succeeded(VerifyMappingExprs(context, use_domain_size, mapping)));
  return Base::get(context, std::make_pair(use_domain_size, mapping));
}

MappingAttr MappingAttr::getChecked(mlir::MLIRContext *context,
                                    int use_domain_size,
                                    llvm::ArrayRef<MappingExpr> mapping) {
  if (mlir::failed(VerifyMappingExprs(context, use_domain_size, mapping))) {
    return nullptr;
  }
  return Base::get(context, std::make_pair(use_domain_size, mapping));
}

MappingAttr MappingAttr::GetIdentity(mlir::MLIRContext *context,
                                     int num_dimensions, int use_domain_size) {
  if (use_domain_size == -1) use_domain_size = num_dimensions;
  assert(use_domain_size >= num_dimensions);
  llvm::SmallVector<MappingExpr, 4> mapping;
  mapping.reserve(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    mapping.push_back(MappingDimExpr::get(i, context));
  }
  return MappingAttr::get(context, use_domain_size, mapping);
}

MappingAttr MappingAttr::FromAffineMap(mlir::AffineMap map) {
  assert(map.isProjectedPermutation());
  llvm::SmallVector<MappingExpr, 8> dimensions;

  dimensions.reserve(map.getNumResults());
  for (mlir::AffineExpr expr : map.getResults()) {
    dimensions.push_back(MappingDimExpr::get(
        expr.cast<mlir::AffineDimExpr>().getPosition(), map.getContext()));
  }
  return get(map.getContext(), map.getNumDims(), dimensions);
}

llvm::ArrayRef<MappingExpr> MappingAttr::Dimensions() const {
  return getImpl()->mapping();
}

int MappingAttr::UseDomainSize() const { return getImpl()->use_domain_size(); }

MappingAttr MappingAttr::Compose(MappingAttr other) const {
  llvm::SmallVector<MappingExpr, 4> new_mapping_dims;
  new_mapping_dims.reserve(other.size());
  for (MappingExpr other_expr : other) {
    new_mapping_dims.push_back(other_expr.SubstituteDims(Dimensions()));
  }
  return MappingAttr::get(getContext(), UseDomainSize(), new_mapping_dims);
}

mlir::AffineMap MappingAttr::AsAffineMap() const {
  llvm::SmallVector<mlir::AffineExpr, 4> affine_exprs;
  affine_exprs.reserve(Dimensions().size());
  for (MappingExpr expr : *this) {
    affine_exprs.push_back(expr.AsAffineExpr());
  }
  return mlir::AffineMap::get(UseDomainSize(), 0, affine_exprs, getContext());
}

bool MappingAttr::IsFullySpecified() const {
  return llvm::all_of(getImpl()->mapping(),
                      [](MappingExpr expr) { return expr.IsFullySpecified(); });
}

MappingAttr MappingAttr::MakeFullySpecified() const {
  int num_dimensions = UseDomainSize();
  llvm::SmallVector<MappingExpr, 4> new_exprs;
  new_exprs.reserve(size());
  for (MappingExpr expr : Dimensions()) {
    new_exprs.push_back(expr.MakeFullySpecified(num_dimensions));
  }
  return MappingAttr::get(getContext(), num_dimensions, new_exprs);
}

bool MappingAttr::IsIdentity() const {
  for (auto en : llvm::enumerate(getImpl()->mapping())) {
    auto dim_expr = en.value().dyn_cast<MappingDimExpr>();
    if (dim_expr == nullptr || dim_expr.dimension() != en.index()) return false;
  }
  return true;
}

llvm::SmallBitVector MappingAttr::DependencyMask() const {
  llvm::SmallBitVector mask(UseDomainSize());
  for (MappingExpr expr : *this) {
    expr.SetDependenciesInMask(mask);
  }
  return mask;
}

bool MappingAttr::IsInjective(int num_dimensions) const {
  llvm::SmallBitVector mask = DependencyMask();
  mask.resize(num_dimensions);
  return mask.all();
}

MappingAttr MappingAttr::ResizeUseDomain(int new_size) const {
  int old_size = UseDomainSize();
  if (new_size == old_size) return *this;
  if (new_size >= old_size) {
    return MappingAttr::get(getContext(), new_size, Dimensions());
  }

  mlir::MLIRContext *context = getContext();
  MappingExpr none = MappingNoneExpr::get(context);
  llvm::SmallVector<MappingExpr, 4> substitutions(old_size, none);
  substitutions.reserve(old_size);
  for (int i = 0; i < new_size; ++i) {
    substitutions[i] = MappingDimExpr::get(i, context);
  }

  llvm::SmallVector<MappingExpr, 4> new_exprs;
  new_exprs.reserve(size());
  for (MappingExpr expr : *this) {
    new_exprs.push_back(expr.SubstituteDims(substitutions));
  }

  return MappingAttr::get(getContext(), new_size, new_exprs);
}

MappingAttr MappingAttr::Resize(int new_size) const {
  int current_size = Dimensions().size();
  if (new_size == current_size) return *this;
  if (new_size < current_size) {
    return MappingAttr::get(getContext(), UseDomainSize(),
                            Dimensions().take_front(new_size));
  }
  llvm::SmallVector<MappingExpr, 4> dimensions(Dimensions().begin(),
                                               Dimensions().end());
  dimensions.resize(new_size, MappingNoneExpr::get(getContext()));
  return MappingAttr::get(getContext(), UseDomainSize(), dimensions);
}

MappingAttr MappingAttr::ShiftRight(int offset, int start_from) const {
  mlir::MLIRContext *context = getContext();

  llvm::SmallVector<MappingExpr, 4> substitutions;
  substitutions.reserve(UseDomainSize());
  for (int i = 0; i < start_from; ++i) {
    substitutions.push_back(MappingDimExpr::get(i, context));
  }
  for (int i = start_from, e = UseDomainSize(); i < e; ++i) {
    substitutions.push_back(MappingDimExpr::get(i + offset, context));
  }

  llvm::SmallVector<MappingExpr, 8> new_dimensions;
  new_dimensions.reserve(Dimensions().size());
  for (MappingExpr dim : Dimensions()) {
    new_dimensions.push_back(dim.SubstituteDims(substitutions));
  }

  return MappingAttr::get(context, UseDomainSize() + offset, new_dimensions);
}

MappingAttr MappingAttr::Inverse() const {
  mlir::MLIRContext *context = getContext();
  llvm::SmallVector<MappingExpr, 4> inverted_exprs(
      UseDomainSize(), MappingNoneExpr::get(context));
  for (int i = 0, e = size(); i < e; ++i) {
    MappingExpr dim_expr = MappingDimExpr::get(i, context);
    auto status = Dimension(i).SetInverse(dim_expr, inverted_exprs);
    assert(mlir::succeeded(status));
  }
  return MappingAttr::get(context, size(), inverted_exprs);
}

MappingAttr MappingAttr::Canonicalize() const {
  llvm::SmallVector<MappingExpr, 4> exprs;
  exprs.reserve(size());
  for (MappingExpr expr : Dimensions()) {
    exprs.push_back(expr.Canonicalize());
  }
  return MappingAttr::get(getContext(), UseDomainSize(), exprs);
}

//===----------------------------------------------------------------------===//
// NamedMappingAttr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::NamedMappingAttr. Instances of
// this class are allocated by MLIR type system in a dedicated arena. Not
// intended for direct use.
class impl::NamedMappingAttrStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifying MappingAttrStorage for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = std::pair<llvm::ArrayRef<mlir::StringAttr>, mlir::Attribute>;

  // Creates a NamedMappingAttrStorage using the provided allocator. Hook for
  // MLIR attribute system.
  static NamedMappingAttrStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<NamedMappingAttrStorage>())
        NamedMappingAttrStorage(
            std::make_pair(allocator.copyInto(key.first), key.second));
  }

  // Compares the NamedMappingAttrStorage identification key with this object.
  bool operator==(const KeyTy &key) const {
    return key.first == names_ && key.second == mapping_;
  }

  llvm::ArrayRef<mlir::StringAttr> names() const { return names_; }

  MappingAttr mapping() const { return mapping_; }

 private:
  // Constructs a storage object from the provided key. Such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit NamedMappingAttrStorage(KeyTy key)
      : names_(key.first), mapping_(key.second.cast<MappingAttr>()) {}

  llvm::ArrayRef<mlir::StringAttr> names_;
  MappingAttr mapping_;
};

NamedMappingAttr NamedMappingAttr::get(llvm::ArrayRef<mlir::StringAttr> names,
                                       MappingAttr mapping) {
  return Base::get(mapping.getContext(), names, mapping);
}

llvm::ArrayRef<mlir::StringAttr> NamedMappingAttr::names() const {
  return getImpl()->names();
}

MappingAttr NamedMappingAttr::mapping() const { return getImpl()->mapping(); }

//===----------------------------------------------------------------------===//
// DomainShapeDim
//===----------------------------------------------------------------------===//

DomainShapeDim::DomainShapeDim(RangeType type, MappingAttr dependency_mapping)
    : type_(type), dependency_mapping_(dependency_mapping) {
  assert(type != nullptr);
  assert(dependency_mapping != nullptr);
}

DomainShapeDim DomainShapeDim::Apply(MappingAttr mapping) const {
  return DomainShapeDim(type_, mapping.Compose(dependency_mapping_));
}

bool operator==(const DomainShapeDim &lhs, const DomainShapeDim &rhs) {
  return lhs.type() == rhs.type() &&
         lhs.dependency_mapping() == rhs.dependency_mapping();
}

unsigned hash_value(const DomainShapeDim &shape_dim) {
  return llvm::hash_combine(shape_dim.type(), shape_dim.dependency_mapping());
}

//===----------------------------------------------------------------------===//
// DomainShapeAttr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::DomainShapeAttr. Instances of
// this class are allocate by MLIR type system in a dedicated arena. Not
// intended for direct use.
class impl::DomainShapeAttrStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies DomainShapeAttrStorage for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = llvm::ArrayRef<DomainShapeDim>;

  // Creates a DomainShapeAttrStorage using the provided allocator. Hook for
  // MLIR attribute system.
  static DomainShapeAttrStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<DomainShapeAttrStorage>())
        DomainShapeAttrStorage(allocator.copyInto(key));
  }

  // Compares the DomainShapeAttrStorage identification key with this object.
  bool operator==(const KeyTy &key) const { return key == dimensions_; }

  // Returns the shape of the iteration dimensions that compose the domain.
  llvm::ArrayRef<DomainShapeDim> dimensions() const { return dimensions_; }

 private:
  // Constructs a storage object for the provided key. such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit DomainShapeAttrStorage(KeyTy key) : dimensions_(key) {}

  // The shape of the dimensions that compose the iteration domain.
  llvm::ArrayRef<DomainShapeDim> dimensions_;
};

DomainShapeAttr DomainShapeAttr::get(mlir::MLIRContext *context,
                                     llvm::ArrayRef<DomainShapeDim> dims) {
  // We omit the use domain size when printing dimensions, so we ensure it
  // has a fixed value.
  for (int i = 0, e = dims.size(); i < e; ++i) {
    assert(dims[i].dependency_mapping().UseDomainSize() == i);
    assert(dims[i].dependency_mapping().IsFullySpecified());
    (void) i;
  }

  return Base::get(context, dims);
}

DomainShapeAttr DomainShapeAttr::HyperRectangular(mlir::MLIRContext *context,
                                                  int rank) {
  DomainShapeAttr empty_shape = DomainShapeAttr::get(context);
  RangeType range_type = RangeType::get(empty_shape);
  llvm::SmallVector<DomainShapeDim, 4> dims;
  dims.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    MappingAttr mapping = MappingAttr::get(context, i, {});
    dims.emplace_back(range_type, mapping);
  }
  return DomainShapeAttr::get(context, dims);
}

DomainShapeAttr DomainShapeAttr::Prefix(int size) {
  if (size == NumDimensions()) return *this;
  return DomainShapeAttr::get(getContext(), Dimensions().take_front(size));
}

int DomainShapeAttr::NumDimensions() const {
  return getImpl()->dimensions().size();
}

bool DomainShapeAttr::Is0d() const { return getImpl()->dimensions().empty(); }

llvm::ArrayRef<DomainShapeDim> DomainShapeAttr::Dimensions() const {
  return getImpl()->dimensions();
}

bool DomainShapeAttr::IsHyperRectangular() const {
  for (auto &dim : getImpl()->dimensions()) {
    if (dim.DependencyMask().any()) return false;
  }
  return true;
}

bool DomainShapeAttr::IsPrefixOf(DomainShapeAttr other) {
  return Dimensions() == other.Dimensions().take_front(NumDimensions());
}

DomainShapeAttr DomainShapeAttr::AccessedShape(MappingAttr mapping) const {
  llvm::SmallVector<DomainShapeDim, 4> shape;
  shape.reserve(mapping.size());
  MappingAttr inverted_mapping = mapping.Inverse();
  for (int i = 0, e = mapping.size(); i < e; ++i) {
    DomainShapeDim shape_dim = mapping.Dimension(i).AccessedShape(
        Dimensions(), inverted_mapping.ResizeUseDomain(i));
    assert(shape_dim.dependency_mapping().UseDomainSize() == i);
    shape.push_back(shape_dim);
  }
  return DomainShapeAttr::get(getContext(), shape);
}

DomainShapeAttr DomainShapeAttr::Product(DomainShapeAttr other) const {
  return ProductAt(NumDimensions(), other);
}

DomainShapeAttr DomainShapeAttr::ProductAt(int pos,
                                           DomainShapeAttr other) const {
  // The leftmost `pos` domain dimensions are kept as is, with the same
  // dependency mapping.
  auto shape = llvm::to_vector<8>(Dimensions().take_front(pos));
  shape.reserve(NumDimensions() + other.NumDimensions());

  // All dimensions coming from the other domain must have their dependencies
  // shifted by `pos` to make sure their refer their new positions.
  for (const DomainShapeDim &dim : other.Dimensions()) {
    shape.emplace_back(dim.type(), dim.dependency_mapping().ShiftRight(pos));
  }

  // The remaining original dimensions must have their dependencies update to
  // account for the inserted dimensions.
  for (const DomainShapeDim &dim : Dimensions().drop_front(pos)) {
    shape.emplace_back(dim.type(), dim.dependency_mapping().ShiftRight(
                                       other.NumDimensions(), pos));
  }

  return DomainShapeAttr::get(getContext(), shape);
}

}  // namespace sair

#include "sair_structs.cc.inc"
