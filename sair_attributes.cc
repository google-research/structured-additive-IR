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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/MLIRContext.h"

namespace sair {

//===----------------------------------------------------------------------===//
// AccessPatternExpr
//===----------------------------------------------------------------------===//

llvm::SmallBitVector AccessPatternExpr::DependencyMask(int domain_size) const {
  llvm::SmallBitVector mask(domain_size);
  SetDependenciesInMask(mask);
  return mask;
}

//===----------------------------------------------------------------------===//
// AccessPatternDimExpr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::AccessPatternDimExpr.
// Instances of this class are allocate by MLIR type system in a dedicated
// arena. Not intended for direct use.
class impl::AccessPatternDimExprStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies AccessPatternDimExpr for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = int;

  // Creates a AccessPatternDimExprStorage using the provided allocator. Hook
  // for MLIR attribute system.
  static AccessPatternDimExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<AccessPatternDimExprStorage>())
        AccessPatternDimExprStorage(key);
  }

  // Compares the AccessPatternDimExprStorage identification key with this
  // object.
  bool operator==(const KeyTy &key) const { return key == dimension_; }

  // Returns the dimension represented by the operation.
  int dimension() const { return dimension_; }

 private:
  // Constructs a storage object for the provided key. such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit AccessPatternDimExprStorage(KeyTy key) : dimension_(key) {}

  int dimension_;
};

AccessPatternDimExpr AccessPatternDimExpr::get(int dimension,
                                               mlir::MLIRContext *context) {
  return Base::get(context, dimension);
}

int AccessPatternDimExpr::dimension() const { return getImpl()->dimension(); }

AccessPatternExpr AccessPatternDimExpr::SubstituteDims(
    mlir::ArrayRef<AccessPatternExpr> exprs) const {
  if (dimension() >= exprs.size()) {
    return AccessPatternNoneExpr::get(getContext());
  }
  return exprs[dimension()];
}

DomainShapeDim AccessPatternDimExpr::AccessedShape(
    llvm::ArrayRef<DomainShapeDim> accessing_shape,
    AccessPatternAttr inverted_pattern) const {
  return accessing_shape[dimension()].Apply(inverted_pattern);
}

mlir::LogicalResult AccessPatternDimExpr::SetInverse(
    AccessPatternExpr context_inverse,
    llvm::MutableArrayRef<AccessPatternExpr> inverses) const {
  AccessPatternExpr inverse = inverses[dimension()].Unify(context_inverse);
  if (inverse == nullptr) return mlir::failure();
  inverses[dimension()] = inverse;
  return mlir::success();
}

AccessPatternExpr AccessPatternDimExpr::Unify(
    AccessPatternExpr other_expr) const {
  if (other_expr == *this) return *this;
  if (other_expr.isa<AccessPatternNoneExpr>()) return *this;
  return AccessPatternExpr(nullptr);
}

mlir::LogicalResult AccessPatternDimExpr::UnificationConstraints(
    AccessPatternExpr other_expr,
    llvm::MutableArrayRef<AccessPatternExpr> constraints) const {
  if (other_expr.isa<AccessPatternNoneExpr>()) return mlir::success();

  AccessPatternExpr &constraint = constraints[dimension()];
  if (constraint == other_expr) return mlir::success();
  if (!constraint.isa<AccessPatternNoneExpr>()) return mlir::failure();
  constraint = other_expr;
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AccessPatternNoneExpr
//===----------------------------------------------------------------------===//

AccessPatternNoneExpr AccessPatternNoneExpr::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

AccessPatternExpr AccessPatternNoneExpr::MakeFullySpecified(
    int &num_dimensions) const {
  return AccessPatternDimExpr::get(num_dimensions++, getContext());
}

//===----------------------------------------------------------------------===//
// AccessPatternStripeExpr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::AccessPatternStripeExpr.
// Instances of this class are allocate by MLIR type system in a dedicated
// arena. Not intended for direct use.
class impl::AccessPatternStripeExprStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies AccessPatternStripeExpr for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = std::tuple<mlir::Attribute, int, int>;

  // Creates a AccessPatternStripeExprStorage using the provided allocator. Hook
  // for MLIR attribute system.
  static AccessPatternStripeExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<AccessPatternStripeExprStorage>())
        AccessPatternStripeExprStorage(key);
  }

  // Compares the AccessPatternStripeExpr identification key with this object.
  bool operator==(const KeyTy &key) const {
    return std::get<0>(key) == operand_ && std::get<1>(key) == step_ &&
           std::get<2>(key) == size_;
  }

  // The striped expression.
  AccessPatternExpr operand() const { return operand_; }

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
  explicit AccessPatternStripeExprStorage(KeyTy key)
      : operand_(std::get<0>(key)),
        step_(std::get<1>(key)),
        size_(std::get<2>(key)) {}

  AccessPatternExpr operand_;
  int step_;
  int size_;
};

AccessPatternStripeExpr AccessPatternStripeExpr::get(AccessPatternExpr operand,
                                                     int step,
                                                     llvm::Optional<int> size) {
  assert(operand != nullptr);
  assert(!size.hasValue() || size.getValue() >= step);
  int int_size = size.hasValue()
                     ? size.getValue()
                     : impl::AccessPatternStripeExprStorage::kNoSize;
  return Base::get(operand.getContext(),
                   std::make_tuple(operand, step, int_size));
}

AccessPatternExpr AccessPatternStripeExpr::operand() const {
  return getImpl()->operand();
}

int AccessPatternStripeExpr::step() const { return getImpl()->step(); }

llvm::Optional<int> AccessPatternStripeExpr::size() const {
  return getImpl()->size();
}

AccessPatternExpr AccessPatternStripeExpr::MakeFullySpecified(
    int &num_dimensions) const {
  return AccessPatternStripeExpr::get(
      operand().MakeFullySpecified(num_dimensions), step(), size());
}

AccessPatternExpr AccessPatternStripeExpr::SubstituteDims(
    llvm::ArrayRef<AccessPatternExpr> exprs) const {
  return AccessPatternStripeExpr::get(operand().SubstituteDims(exprs), step(),
                                      size());
}

AccessPatternExpr AccessPatternStripeExpr::Unify(
    AccessPatternExpr other_expr) const {
  if (other_expr.isa<AccessPatternNoneExpr>()) return *this;
  AccessPatternStripeExpr other_stripe =
      other_expr.dyn_cast<AccessPatternStripeExpr>();
  if (other_stripe == nullptr || size() != other_stripe.size() ||
      step() != other_stripe.step()) {
    return AccessPatternExpr(nullptr);
  }
  AccessPatternExpr unified_operand = operand().Unify(other_stripe.operand());
  if (unified_operand == nullptr) return AccessPatternExpr(nullptr);
  return AccessPatternStripeExpr::get(unified_operand, step(), size());
}

mlir::LogicalResult AccessPatternStripeExpr::UnificationConstraints(
    AccessPatternExpr other_expr,
    llvm::MutableArrayRef<AccessPatternExpr> constraints) const {
  if (other_expr.isa<AccessPatternNoneExpr>()) return mlir::success();
  auto other_stripe = other_expr.dyn_cast<AccessPatternStripeExpr>();
  if (other_stripe == nullptr || size() != other_stripe.size() ||
      step() != other_stripe.step()) {
    return mlir::failure();
  }
  return operand().UnificationConstraints(other_stripe.operand(), constraints);
}

DomainShapeDim AccessPatternStripeExpr::AccessedShape(
    llvm::ArrayRef<DomainShapeDim> accessing_shape,
    AccessPatternAttr inverted_pattern) const {
  mlir::MLIRContext *context = getContext();

  DomainShapeDim inner_shape =
      operand().AccessedShape(accessing_shape, inverted_pattern);
  auto inverse_subexpr = operand()
                             .FindInInverse(inverted_pattern.Dimensions())
                             .cast<AccessPatternUnStripeExpr>();

  // Append dependencies to larger stripes to the dependency pattern.
  llvm::SmallVector<AccessPatternExpr, 4> dependency_pattern_exprs;
  llvm::append_range(dependency_pattern_exprs,
                     inner_shape.dependency_pattern());

  llvm::SmallVector<DomainShapeDim, 4> type_shape;
  llvm::append_range(type_shape, inner_shape.type().Shape().Dimensions());
  RangeType type = inner_shape.type();

  for (auto [expr, step] :
       llvm::zip(inverse_subexpr.operands(), inverse_subexpr.factors())) {
    if (step == this->step()) break;
    type_shape.emplace_back(
        type, AccessPatternAttr::GetIdentity(context, type_shape.size()));
    type = RangeType::get(context, DomainShapeAttr::get(context, type_shape));
    dependency_pattern_exprs.push_back(expr);
  }

  auto dependency_pattern = AccessPatternAttr::get(
      context, inverted_pattern.UseDomainSize(), dependency_pattern_exprs);
  return DomainShapeDim(type, dependency_pattern);
}

mlir::LogicalResult AccessPatternStripeExpr::SetInverse(
    AccessPatternExpr context_inverse,
    llvm::MutableArrayRef<AccessPatternExpr> inverses) const {
  AccessPatternExpr none = AccessPatternNoneExpr::get(getContext());
  llvm::SmallVector<AccessPatternExpr, 3> operands;
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

  return operand().SetInverse(AccessPatternUnStripeExpr::get(operands, factors),
                              inverses);
}

AccessPatternExpr AccessPatternStripeExpr::FindInInverse(
    llvm::ArrayRef<AccessPatternExpr> inverse) const {
  auto unstripe_expr =
      operand().FindInInverse(inverse).cast<AccessPatternUnStripeExpr>();
  auto factor_it = llvm::find(unstripe_expr.factors(), step());
  int pos = std::distance(unstripe_expr.factors().begin(), factor_it);
  return unstripe_expr.operands()[pos];
}

//===----------------------------------------------------------------------===//
// AccessPatternUnStripeExpr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::AccessPatternUnStripeExpr.
// Instances of this class are allocate by MLIR type system in a dedicated
// arena. Not intended for direct use.
class impl::AccessPatternUnStripeExprStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies AccessPatternUnStripeExpr for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy =
      std::pair<llvm::ArrayRef<AccessPatternExpr>, llvm::ArrayRef<int>>;

  // Creates a AccessPatternUnStripeExprStorage using the provided allocator.
  // Hook for MLIR attribute system.
  static AccessPatternUnStripeExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<AccessPatternUnStripeExprStorage>())
        AccessPatternUnStripeExprStorage(allocator.copyInto(key.first),
                                         allocator.copyInto(key.second));
  }

  // Compares the AccessPatternUnStripeExprStorage identification key with this
  // object.
  bool operator==(const KeyTy &key) const {
    return key.first == operands_ && key.second == factors_;
  }

  // Stripe expressions that are combined to obtain the unstriped expression.
  llvm::ArrayRef<AccessPatternExpr> operands() const { return operands_; }

  // Stripe expression sizes.
  llvm::ArrayRef<int> factors() const { return factors_; }

 private:
  // Constructs a storage object for the provided key. such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit AccessPatternUnStripeExprStorage(
      llvm::ArrayRef<AccessPatternExpr> operands, llvm::ArrayRef<int> factors)
      : operands_(operands), factors_(factors) {}

  llvm::ArrayRef<AccessPatternExpr> operands_;
  llvm::ArrayRef<int> factors_;
};

AccessPatternUnStripeExpr AccessPatternUnStripeExpr::get(
    llvm::ArrayRef<AccessPatternExpr> stripes, llvm::ArrayRef<int> factors) {
  assert(factors.empty() || factors[0] > 1);
  assert(stripes.size() == factors.size() + 1);
#ifndef NDEBUG
  for (int i = 0; i + 1 < factors.size(); ++i) {
    assert(factors[i] > factors[i + 1]);
  }
#endif
  return Base::get(stripes[0].getContext(), std::make_pair(stripes, factors));
}

llvm::ArrayRef<AccessPatternExpr> AccessPatternUnStripeExpr::operands() const {
  return getImpl()->operands();
}

llvm::ArrayRef<int> AccessPatternUnStripeExpr::factors() const {
  return getImpl()->factors();
}

bool AccessPatternUnStripeExpr::IsFullySpecified() const {
  return llvm::all_of(operands(), [](AccessPatternExpr expr) {
    return expr.IsFullySpecified();
  });
}

void AccessPatternUnStripeExpr::SetDependenciesInMask(
    llvm::SmallBitVector &mask) const {
  for (AccessPatternExpr expr : operands()) {
    expr.SetDependenciesInMask(mask);
  }
}

int AccessPatternUnStripeExpr::MinDomainSize() const {
  int max = 0;
  for (AccessPatternExpr expr : operands()) {
    max = std::max(max, expr.MinDomainSize());
  }
  return max;
}

AccessPatternExpr AccessPatternUnStripeExpr::MakeFullySpecified(
    int &num_dimensions) const {
  llvm::SmallVector<AccessPatternExpr, 4> new_exprs;
  new_exprs.reserve(operands().size());
  for (AccessPatternExpr expr : operands()) {
    new_exprs.push_back(expr.MakeFullySpecified(num_dimensions));
  }
  return AccessPatternUnStripeExpr::get(new_exprs, factors());
}

AccessPatternExpr AccessPatternUnStripeExpr::SubstituteDims(
    llvm::ArrayRef<AccessPatternExpr> exprs) const {
  llvm::SmallVector<AccessPatternExpr, 4> new_exprs;
  new_exprs.reserve(operands().size());
  for (AccessPatternExpr expr : operands()) {
    new_exprs.push_back(expr.SubstituteDims(exprs));
  }
  return AccessPatternUnStripeExpr::get(new_exprs, factors());
}

DomainShapeDim AccessPatternUnStripeExpr::AccessedShape(
    llvm::ArrayRef<DomainShapeDim> accessing_shape,
    AccessPatternAttr inverted_pattern) const {
  // The shape of the unstriped dimension is the shape of the outer-most striped
  // dimensions. Other striped dimensions have similar shape, but with
  // additional dependencies to outer striped dimensions.
  return operands().front().AccessedShape(accessing_shape, inverted_pattern);
}

mlir::LogicalResult AccessPatternUnStripeExpr::SetInverse(
    AccessPatternExpr context_inverse,
    llvm::MutableArrayRef<AccessPatternExpr> inverses) const {
  for (int i = 0, e = factors().size(); i <= e; ++i) {
    int step = i == e ? 1 : factors()[i];
    llvm::Optional<int> size =
        i == 0 ? llvm::Optional<int>() : factors()[i - 1];
    auto stripe_expr =
        AccessPatternStripeExpr::get(context_inverse, step, size);
    if (mlir::failed(operands()[i].SetInverse(
            stripe_expr.cast<AccessPatternExpr>(), inverses))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

AccessPatternExpr AccessPatternUnStripeExpr::Unify(
    AccessPatternExpr other_expr) const {
  if (other_expr.isa<AccessPatternNoneExpr>()) return *this;
  AccessPatternUnStripeExpr other_unstripe =
      other_expr.dyn_cast<AccessPatternUnStripeExpr>();
  if (other_unstripe == nullptr) return AccessPatternExpr(nullptr);

  llvm::SmallVector<AccessPatternExpr, 4> new_exprs;
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
      if (!operands()[this_cursor].isa<AccessPatternNoneExpr>()) {
        return AccessPatternExpr(nullptr);
      }
      new_factors.push_back(other_step);
      new_exprs.push_back(other_unstripe.operands()[other_cursor]);
      ++other_cursor;
    } else if (this_step > other_step) {
      // We can only subdivide none exprs.
      if (!other_unstripe.operands()[other_cursor]
               .isa<AccessPatternNoneExpr>()) {
        return AccessPatternExpr(nullptr);
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
  return AccessPatternUnStripeExpr::get(new_exprs, new_factors);
}

mlir::LogicalResult AccessPatternUnStripeExpr::UnificationConstraints(
    AccessPatternExpr other_expr,
    llvm::MutableArrayRef<AccessPatternExpr> constraints) const {
  if (other_expr.isa<AccessPatternNoneExpr>()) return mlir::success();
  AccessPatternUnStripeExpr other_unstripe =
      other_expr.dyn_cast<AccessPatternUnStripeExpr>();
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
      if (!operands()[this_cursor].isa<AccessPatternNoneExpr>()) {
        return mlir::failure();
      }
      ++other_cursor;
    } else if (this_step > other_step) {
      // We can only subdivide none exprs.
      if (!other_unstripe.operands()[other_cursor]
               .isa<AccessPatternNoneExpr>()) {
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

AccessPatternExpr AccessPatternUnStripeExpr::FindInInverse(
    llvm::ArrayRef<AccessPatternExpr> inverse) const {
  return operands()[0]
      .FindInInverse(inverse)
      .cast<AccessPatternStripeExpr>()
      .operand();
}

//===----------------------------------------------------------------------===//
// AccessPatternAttr
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::AccessPatternAttr. Instances
// of this class are allocated by MLIR type system in a dedicated arena. Not
// intended for direct use.
class impl::AccessPatternAttrStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifying AccessPatternAttrStorage for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = std::pair<int, llvm::ArrayRef<AccessPatternExpr>>;

  // Creates an AccessPatternAttrStorage using the provided allocator. Hook for
  // MLIR attribute system.
  static AccessPatternAttrStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<AccessPatternAttrStorage>())
        AccessPatternAttrStorage(
            std::make_pair(key.first, allocator.copyInto(key.second)));
  }

  // Compares the AccessPatternAttrStorage identification key with this object.
  bool operator==(const KeyTy &key) const {
    return key.first == use_domain_size_ && key.second == pattern_;
  }

  // Returns the number of dimensions in the use domain.
  int use_domain_size() const { return use_domain_size_; }

  // Returns the list of dimensions along which a variable is accessed.
  // Dimensions are identified by their position in the domain definition.
  llvm::ArrayRef<AccessPatternExpr> pattern() const { return pattern_; }

 private:
  // Constructs a storage object from the provided key. Such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit AccessPatternAttrStorage(KeyTy key)
      : use_domain_size_(key.first), pattern_(key.second) {}

  int use_domain_size_;

  // The list of dimensions along which a Sair variable is accessed.
  llvm::ArrayRef<AccessPatternExpr> pattern_;
};

// Verifies that the expressions form a valid access pattern
static mlir::LogicalResult VerifyPatternExprs(
    mlir::MLIRContext *context, int use_domain_size,
    llvm::ArrayRef<AccessPatternExpr> pattern_exprs) {
  llvm::SmallVector<AccessPatternExpr, 4> inverted_exprs(
      use_domain_size, AccessPatternNoneExpr::get(context));
  for (int i = 0, e = pattern_exprs.size(); i < e; ++i) {
    AccessPatternExpr dim_expr = AccessPatternDimExpr::get(i, context);
    if (mlir::failed(pattern_exprs[i].SetInverse(dim_expr, inverted_exprs))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

AccessPatternAttr AccessPatternAttr::get(
    mlir::MLIRContext *context, int use_domain_size,
    llvm::ArrayRef<AccessPatternExpr> pattern) {
  assert(
      mlir::succeeded(VerifyPatternExprs(context, use_domain_size, pattern)));
  return Base::get(context, std::make_pair(use_domain_size, pattern));
}

AccessPatternAttr AccessPatternAttr::getChecked(
    mlir::MLIRContext *context, int use_domain_size,
    llvm::ArrayRef<AccessPatternExpr> pattern) {
  if (mlir::failed(VerifyPatternExprs(context, use_domain_size, pattern))) {
    return nullptr;
  }
  return Base::get(context, std::make_pair(use_domain_size, pattern));
}

AccessPatternAttr AccessPatternAttr::GetIdentity(mlir::MLIRContext *context,
                                                 int num_dimensions,
                                                 int use_domain_size) {
  if (use_domain_size == -1) use_domain_size = num_dimensions;
  assert(use_domain_size >= num_dimensions);
  llvm::SmallVector<AccessPatternExpr, 4> pattern;
  pattern.reserve(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    pattern.push_back(AccessPatternDimExpr::get(i, context));
  }
  return AccessPatternAttr::get(context, use_domain_size, pattern);
}

AccessPatternAttr AccessPatternAttr::FromAffineMap(mlir::AffineMap map) {
  assert(map.isProjectedPermutation());
  llvm::SmallVector<AccessPatternExpr, 8> dimensions;

  dimensions.reserve(map.getNumResults());
  for (mlir::AffineExpr expr : map.getResults()) {
    dimensions.push_back(AccessPatternDimExpr::get(
        expr.cast<mlir::AffineDimExpr>().getPosition(), map.getContext()));
  }
  return get(map.getContext(), map.getNumDims(), dimensions);
}

llvm::ArrayRef<AccessPatternExpr> AccessPatternAttr::Dimensions() const {
  return getImpl()->pattern();
}

int AccessPatternAttr::UseDomainSize() const {
  return getImpl()->use_domain_size();
}

AccessPatternAttr AccessPatternAttr::Compose(AccessPatternAttr other) const {
  llvm::SmallVector<AccessPatternExpr, 4> new_access_pattern_dims;
  new_access_pattern_dims.reserve(other.size());
  for (AccessPatternExpr other_expr : other) {
    new_access_pattern_dims.push_back(other_expr.SubstituteDims(Dimensions()));
  }
  return AccessPatternAttr::get(getContext(), UseDomainSize(),
                                new_access_pattern_dims);
}

// TODO(b/339834234): move this to the memref introduction pass
static mlir::AffineExpr AsAffineExpr(AccessPatternExpr expr) {
  if (auto dim_expr = expr.dyn_cast<AccessPatternDimExpr>()) {
    return getAffineDimExpr(dim_expr.dimension(), dim_expr.getContext());
  } else if (auto stripe_expr = expr.dyn_cast<AccessPatternStripeExpr>()) {
    mlir::AffineExpr inner_expr = AsAffineExpr(stripe_expr.operand());
    auto step = mlir::getAffineConstantExpr(stripe_expr.step(),
                                            stripe_expr.getContext());
    auto floor_div = mlir::AffineExprKind::FloorDiv;
    return mlir::getAffineBinaryOpExpr(floor_div, inner_expr, step) * step;
  } else if (auto unstripe_expr = expr.dyn_cast<AccessPatternUnStripeExpr>()) {
    return AsAffineExpr(unstripe_expr.operands().back());
  }
  llvm_unreachable("unsupported expression");
}

mlir::AffineMap AccessPatternAttr::AsAffineMap() const {
  llvm::SmallVector<mlir::AffineExpr, 4> affine_exprs;
  affine_exprs.reserve(Dimensions().size());
  for (AccessPatternExpr expr : *this) {
    affine_exprs.push_back(AsAffineExpr(expr));
  }
  return mlir::AffineMap::get(UseDomainSize(), 0, affine_exprs, getContext());
}

mlir::AffineMap AccessPatternAttr::InverseAffineMap() const {
  return mlir::inversePermutation(AsAffineMap());
}

bool AccessPatternAttr::IsFullySpecified() const {
  return llvm::all_of(getImpl()->pattern(), [](AccessPatternExpr expr) {
    return expr.IsFullySpecified();
  });
}

AccessPatternAttr AccessPatternAttr::MakeFullySpecified() const {
  int num_dimensions = UseDomainSize();
  llvm::SmallVector<AccessPatternExpr, 4> new_exprs;
  new_exprs.reserve(size());
  for (AccessPatternExpr expr : Dimensions()) {
    new_exprs.push_back(expr.MakeFullySpecified(num_dimensions));
  }
  return AccessPatternAttr::get(getContext(), num_dimensions, new_exprs);
}

bool AccessPatternAttr::IsIdentity() const {
  for (auto en : llvm::enumerate(getImpl()->pattern())) {
    auto dim_expr = en.value().dyn_cast<AccessPatternDimExpr>();
    if (dim_expr == nullptr || dim_expr.dimension() != en.index()) return false;
  }
  return true;
}

llvm::SmallBitVector AccessPatternAttr::DependencyMask() const {
  llvm::SmallBitVector mask(UseDomainSize());
  for (AccessPatternExpr expr : *this) {
    expr.SetDependenciesInMask(mask);
  }
  return mask;
}

bool AccessPatternAttr::IsInjective(int num_dimensions) const {
  llvm::SmallBitVector mask = DependencyMask();
  mask.resize(num_dimensions);
  return mask.all();
}

AccessPatternAttr AccessPatternAttr::ResizeUseDomain(int new_size) const {
  int old_size = UseDomainSize();
  if (new_size == old_size) return *this;
  if (new_size >= old_size) {
    return AccessPatternAttr::get(getContext(), new_size, Dimensions());
  }

  mlir::MLIRContext *context = getContext();
  AccessPatternExpr none = AccessPatternNoneExpr::get(context);
  llvm::SmallVector<AccessPatternExpr, 4> substitutions(old_size, none);
  substitutions.reserve(old_size);
  for (int i = 0; i < new_size; ++i) {
    substitutions[i] = AccessPatternDimExpr::get(i, context);
  }

  llvm::SmallVector<AccessPatternExpr, 4> new_exprs;
  new_exprs.reserve(size());
  for (AccessPatternExpr expr : *this) {
    new_exprs.push_back(expr.SubstituteDims(substitutions));
  }

  return AccessPatternAttr::get(getContext(), new_size, new_exprs);
}

AccessPatternAttr AccessPatternAttr::Resize(int new_size) const {
  int current_size = Dimensions().size();
  if (new_size == current_size) return *this;
  if (new_size < current_size) {
    return AccessPatternAttr::get(getContext(), UseDomainSize(),
                                  Dimensions().take_front(new_size));
  }
  llvm::SmallVector<AccessPatternExpr, 4> dimensions(Dimensions().begin(),
                                                     Dimensions().end());
  dimensions.resize(new_size, AccessPatternNoneExpr::get(getContext()));
  return AccessPatternAttr::get(getContext(), UseDomainSize(), dimensions);
}

AccessPatternAttr AccessPatternAttr::ShiftRight(int offset,
                                                int start_from) const {
  mlir::MLIRContext *context = getContext();

  llvm::SmallVector<AccessPatternExpr, 4> substitutions;
  substitutions.reserve(UseDomainSize());
  for (int i = 0; i < start_from; ++i) {
    substitutions.push_back(AccessPatternDimExpr::get(i, context));
  }
  for (int i = start_from, e = UseDomainSize(); i < e; ++i) {
    substitutions.push_back(AccessPatternDimExpr::get(i + offset, context));
  }

  llvm::SmallVector<AccessPatternExpr, 8> new_dimensions;
  new_dimensions.reserve(Dimensions().size());
  for (AccessPatternExpr dim : Dimensions()) {
    new_dimensions.push_back(dim.SubstituteDims(substitutions));
  }

  return AccessPatternAttr::get(context, UseDomainSize() + offset,
                                new_dimensions);
}

AccessPatternAttr AccessPatternAttr::Inverse() const {
  mlir::MLIRContext *context = getContext();
  llvm::SmallVector<AccessPatternExpr, 4> inverted_exprs(
      UseDomainSize(), AccessPatternNoneExpr::get(context));
  for (int i = 0, e = size(); i < e; ++i) {
    AccessPatternExpr dim_expr = AccessPatternDimExpr::get(i, context);
    auto status = Dimension(i).SetInverse(dim_expr, inverted_exprs);
    assert(mlir::succeeded(status));
  }
  return AccessPatternAttr::get(context, size(), inverted_exprs);
}

//===----------------------------------------------------------------------===//
// DomainShapeDim
//===----------------------------------------------------------------------===//

DomainShapeDim::DomainShapeDim(RangeType type,
                               AccessPatternAttr dependency_pattern)
    : type_(type), dependency_pattern_(dependency_pattern) {
  assert(type != nullptr);
  assert(dependency_pattern != nullptr);
  assert(dependency_pattern.IsFullySpecified());
}

DomainShapeDim DomainShapeDim::Apply(AccessPatternAttr access_pattern) const {
  return DomainShapeDim(type_, access_pattern.Compose(dependency_pattern_));
}

bool operator==(const DomainShapeDim &lhs, const DomainShapeDim &rhs) {
  return lhs.type() == rhs.type() &&
         lhs.dependency_pattern() == rhs.dependency_pattern();
}

unsigned hash_value(const DomainShapeDim &shape_dim) {
  return llvm::hash_combine(shape_dim.type(), shape_dim.dependency_pattern());
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
    assert(dims[i].dependency_pattern().UseDomainSize() == i);
    (void) i;
  }

  return Base::get(context, dims);
}

DomainShapeAttr DomainShapeAttr::HyperRectangular(mlir::MLIRContext *context,
                                                  int rank) {
  DomainShapeAttr empty_shape = DomainShapeAttr::get(context);
  RangeType range_type = RangeType::get(context, empty_shape);
  llvm::SmallVector<DomainShapeDim, 4> dims;
  dims.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    AccessPatternAttr access_pattern = AccessPatternAttr::get(context, i, {});
    dims.emplace_back(range_type, access_pattern);
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

DomainShapeAttr DomainShapeAttr::AccessedShape(
    AccessPatternAttr access_pattern) const {
  llvm::SmallVector<DomainShapeDim, 4> shape;
  shape.reserve(access_pattern.size());
  AccessPatternAttr inverted_pattern = access_pattern.Inverse();
  for (int i = 0, e = access_pattern.size(); i < e; ++i) {
    DomainShapeDim shape_dim = access_pattern.Dimension(i).AccessedShape(
        Dimensions(), inverted_pattern.ResizeUseDomain(i));
    assert(shape_dim.dependency_pattern().UseDomainSize() == i);
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
  // dependency pattern.
  auto shape = llvm::to_vector<8>(Dimensions().take_front(pos));
  shape.reserve(NumDimensions() + other.NumDimensions());

  // All dimensions coming from the other domain must have their dependencies
  // shifted by `pos` to make sure their refer their new positions.
  for (const DomainShapeDim &dim : other.Dimensions()) {
    shape.emplace_back(dim.type(), dim.dependency_pattern().ShiftRight(pos));
  }

  // The remaining original dimensions must have their dependencies update to
  // account for the inserted dimensions.
  for (const DomainShapeDim &dim : Dimensions().drop_front(pos)) {
    shape.emplace_back(dim.type(), dim.dependency_pattern().ShiftRight(
                                       other.NumDimensions(), pos));
  }

  return DomainShapeAttr::get(getContext(), shape);
}

// Private implementation/storage class for sair::IteratorAttr. Instances
// of this class are allocated by MLIR type system in a dedicated arena. Not
// intended for direct use.
class impl::IteratorAttrStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifying IteratorAttrStorage for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = std::pair<int, int>;

  // Creates a IteratorAttrStorage using the provided allocator. Hook for
  // MLIR attribute system.
  static IteratorAttrStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<IteratorAttrStorage>())
        IteratorAttrStorage(key.first, key.second);
  }

  // Compares the identification key with this object.
  bool operator==(const KeyTy &key) const {
    return key.first == dimension_ && key.second == step_;
  }

  // Value to assign to dimension to represent the iterator that rematerialize
  // the computation.
  static constexpr int kRematerialize = -1;

  // Dimension to iterate on.
  int dimension() const { return dimension_; }
  // Size of the chunks to iterate on.
  int step() const { return step_; }

 private:
  // Constructs a storage object from the provided key. Such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit IteratorAttrStorage(int dimension, int step)
      : dimension_(dimension), step_(step) {
    assert(step > 0);
  }

  int dimension_;
  int step_;
};

IteratorAttr IteratorAttr::get(mlir::MLIRContext *context, int dimension,
                               int step) {
  assert(dimension >= 0);
  return Base::get(context, std::make_pair(dimension, step));
}

IteratorAttr IteratorAttr::get(mlir::MLIRContext *context) {
  return Base::get(
      context, std::make_pair(impl::IteratorAttrStorage::kRematerialize, 1));
}

bool IteratorAttr::Rematerialize() {
  return getImpl()->dimension() == impl::IteratorAttrStorage::kRematerialize;
}

int IteratorAttr::Step() const {
  assert(getImpl()->dimension() != impl::IteratorAttrStorage::kRematerialize);
  return getImpl()->step();
}

int IteratorAttr::Dimension() {
  int dimension = getImpl()->dimension();
  assert(dimension != impl::IteratorAttrStorage::kRematerialize);
  return dimension;
}

}  // namespace sair

#include "sair_structs.cc.inc"
