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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "sair_dialect.h"

namespace sair {

#include "sair_attr_interfaces.cc.inc"

mlir::InFlightDiagnostic AttrLocation::EmitError() const {
  if (name_ == nullptr) {
    return mlir::emitError(loc_) << "in " << kind_ << ": ";
  }
  return mlir::emitError(location()) << "in " << kind_ << " " << name() << ": ";
}

mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const AttrLocation &loc) {
  diag << loc.kind_;
  if (loc.name_ == nullptr) return diag;
  return diag << " " << loc.name_;
}

//===----------------------------------------------------------------------===//
// MappingExpr
//===----------------------------------------------------------------------===//

MappingExpr MappingExpr::SubstituteDims(
    mlir::ArrayRef<MappingExpr> exprs) const {
  return Map([&](MappingExpr sub_expr) -> MappingExpr {
    auto dim_expr = llvm::dyn_cast<MappingDimExpr>(sub_expr);
    if (dim_expr == nullptr) return sub_expr;
    if (dim_expr.dimension() >= exprs.size()) {
      return MappingNoneExpr::get(getContext());
    }
    return exprs[dim_expr.dimension()];
  });
}

llvm::SmallBitVector MappingExpr::DependencyMask(int domain_size) const {
  llvm::SmallBitVector mask(domain_size);
  SetDependenciesInMask(mask);
  return mask;
}

bool MappingExpr::HasNoneExprs() const {
  bool has_none_exprs = false;
  Walk([&](MappingExpr sub_expr) {
    has_none_exprs |= llvm::isa<MappingNoneExpr>(sub_expr);
  });
  return has_none_exprs;
}

bool MappingExpr::HasUnknownExprs() const {
  bool has_unknown_exprs = false;
  Walk([&](MappingExpr sub_expr) {
    has_unknown_exprs |= llvm::isa<MappingUnknownExpr>(sub_expr);
  });
  return has_unknown_exprs;
}

void MappingExpr::SetDependenciesInMask(llvm::SmallBitVector &mask) const {
  Walk([&](MappingExpr sub_expr) {
    auto dim_expr = llvm::dyn_cast<MappingDimExpr>(sub_expr);
    if (dim_expr == nullptr) return;
    mask.set(dim_expr.dimension());
  });
}

int MappingExpr::MinDomainSize() const {
  int min_domain_size = 0;
  Walk([&](MappingExpr sub_expr) {
    auto dim_expr = llvm::dyn_cast<MappingDimExpr>(sub_expr);
    if (dim_expr == nullptr) return;
    min_domain_size = std::max(min_domain_size, dim_expr.dimension() + 1);
  });
  return min_domain_size;
}

// Resolves unification of `lhs` and `rhs` for the case where one of the
// expression is `?` or `none`. Returns `nullptr` if unification fails.
static MappingExpr ResolveNoneAndUnknownUnification(MappingExpr lhs,
                                                    MappingExpr rhs) {
  if (llvm::isa<MappingNoneExpr>(lhs)) return rhs;
  if (llvm::isa<MappingNoneExpr>(rhs)) return lhs;
  if (llvm::isa<MappingUnknownExpr>(lhs)) return rhs;
  if (llvm::isa<MappingUnknownExpr>(rhs)) return lhs;
  return MappingExpr();
}

MappingExpr Unify(MappingExpr lhs, MappingExpr rhs) {
  return lhs.Unify(rhs, ResolveNoneAndUnknownUnification);
}

mlir::LogicalResult UnificationConstraints(
    MappingAttr lhs, MappingAttr rhs,
    llvm::MutableArrayRef<MappingExpr> constraints) {
  assert(lhs.size() == rhs.size());
  // Shift lhs so that all its dimensions are distinct from rhs.
  int shift = rhs.UseDomainSize();
  lhs = lhs.ShiftRight(shift);
  for (auto [lhs_expr, rhs_expr] : llvm::zip(lhs, rhs)) {
    MappingExpr result =
        lhs_expr.Unify(rhs_expr, [&](MappingExpr sub_lhs, MappingExpr sub_rhs) {
          auto trivial_resolve =
              ResolveNoneAndUnknownUnification(sub_lhs, sub_rhs);
          if (trivial_resolve != nullptr) return trivial_resolve;

          auto dim_expr = llvm::dyn_cast<MappingDimExpr>(sub_lhs);
          if (dim_expr == nullptr) return MappingExpr();

          MappingExpr &constraint = constraints[dim_expr.dimension() - shift];
          constraint = Unify(constraint, sub_rhs);
          if (constraint == nullptr) return MappingExpr();
          return sub_lhs;
        });
    if (result == nullptr) return mlir::failure();
  }
  return mlir::success();
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

MappingExpr MappingDimExpr::Map(
    llvm::function_ref<MappingExpr(MappingExpr)> function) const {
  return function(*this);
}

void MappingDimExpr::Walk(
    llvm::function_ref<void(MappingExpr)> function) const {
  function(*this);
}

mlir::LogicalResult MappingDimExpr::SetInverse(
    MappingExpr context_inverse,
    llvm::MutableArrayRef<MappingExpr> inverses) const {
  MappingExpr inverse = sair::Unify(inverses[dimension()], context_inverse);
  if (inverse == nullptr) return mlir::failure();
  inverses[dimension()] = inverse;
  return mlir::success();
}

MappingExpr MappingDimExpr::Unify(
    MappingExpr other_expr,
    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)> on_mismatch)
    const {
  if (other_expr == *this) return *this;
  return on_mismatch(*this, other_expr);
}

mlir::AffineExpr MappingDimExpr::AsAffineExpr() const {
  return mlir::getAffineDimExpr(dimension(), getContext());
}

//===----------------------------------------------------------------------===//
// MappingNoneExpr
//===----------------------------------------------------------------------===//

MappingNoneExpr MappingNoneExpr::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

MappingExpr MappingNoneExpr::Map(
    llvm::function_ref<MappingExpr(MappingExpr)> function) const {
  return function(*this);
}

void MappingNoneExpr::Walk(
    llvm::function_ref<void(MappingExpr)> function) const {
  function(*this);
}

MappingExpr MappingNoneExpr::Unify(
    MappingExpr other_expr,
    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)> on_mismatch)
    const {
  if (other_expr == *this) return *this;
  return on_mismatch(*this, other_expr);
}

//===----------------------------------------------------------------------===//
// MappingUnknownExpr
//===----------------------------------------------------------------------===//

MappingUnknownExpr MappingUnknownExpr::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

MappingExpr MappingUnknownExpr::Map(
    llvm::function_ref<MappingExpr(MappingExpr)> function) const {
  return function(*this);
}

void MappingUnknownExpr::Walk(
    llvm::function_ref<void(MappingExpr)> function) const {
  function(*this);
}

MappingExpr MappingUnknownExpr::Unify(
    MappingExpr other_expr,
    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)> on_mismatch)
    const {
  if (other_expr == *this) return *this;
  return on_mismatch(*this, other_expr);
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
  using KeyTy = std::pair<mlir::Attribute, llvm::ArrayRef<int>>;

  // Creates a MappingStripeExprStorage using the provided allocator. Hook
  // for MLIR attribute system.
  static MappingStripeExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<MappingStripeExprStorage>())
        MappingStripeExprStorage(cast<MappingExpr>(key.first),
                                 allocator.copyInto(key.second));
  }

  // Compares the MappingStripeExpr identification key with this object.
  bool operator==(const KeyTy &key) const {
    return key.first == operand_ && key.second == factors_;
  }

  // The striped expression.
  MappingExpr operand() const { return operand_; }

  // Stripe factors.
  llvm::ArrayRef<int> factors() const { return factors_; }

 private:
  // Constructs a storage object for the provided key. such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  explicit MappingStripeExprStorage(MappingExpr operand,
                                    llvm::ArrayRef<int> factors)
      : operand_(operand), factors_(factors) {}

  MappingExpr operand_;
  llvm::ArrayRef<int> factors_;
};

MappingStripeExpr MappingStripeExpr::get(MappingExpr operand,
                                         llvm::ArrayRef<int> factors) {
  assert(operand != nullptr);
  assert(!factors.empty());
  return Base::get(operand.getContext(), std::make_tuple(operand, factors));
}

MappingExpr MappingStripeExpr::operand() const { return getImpl()->operand(); }

llvm::ArrayRef<int> MappingStripeExpr::factors() const {
  return getImpl()->factors();
}

MappingExpr MappingStripeExpr::Map(
    llvm::function_ref<MappingExpr(MappingExpr)> function) const {
  MappingExpr new_operand = operand().Map(function);
  return function(MappingStripeExpr::get(new_operand, factors()));
}

void MappingStripeExpr::Walk(
    llvm::function_ref<void(MappingExpr)> function) const {
  operand().Walk(function);
  function(*this);
}

MappingExpr MappingStripeExpr::Unify(
    MappingExpr other_expr,
    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)> on_mismatch)
    const {
  MappingStripeExpr other_stripe =
      llvm::dyn_cast<MappingStripeExpr>(other_expr);
  if (other_stripe == nullptr || factors() != other_stripe.factors()) {
    return on_mismatch(*this, other_expr);
  }
  MappingExpr unified_operand =
      operand().Unify(other_stripe.operand(), on_mismatch);
  if (unified_operand == nullptr) return MappingExpr();
  return MappingStripeExpr::get(unified_operand, factors());
}

mlir::LogicalResult MappingStripeExpr::SetInverse(
    MappingExpr context_inverse,
    llvm::MutableArrayRef<MappingExpr> inverses) const {
  MappingExpr none = MappingNoneExpr::get(getContext());

  // Prefix unstripe operands by none for outer stripes.
  llvm::SmallVector<MappingExpr, 3> unstripe_operands(factors().size() - 1,
                                                      none);
  unstripe_operands.push_back(context_inverse);
  llvm::SmallVector<int, 2> unstripe_factors;
  llvm::append_range(unstripe_factors, factors());

  // Add a `none` operand for inner stripes.
  if (unstripe_factors.back() != 1) {
    unstripe_factors.push_back(1);
    unstripe_operands.push_back(none);
  }

  auto unstripe = MappingUnStripeExpr::get(unstripe_operands, unstripe_factors);
  return operand().SetInverse(unstripe, inverses);
}

MappingExpr MappingStripeExpr::FindInInverse(
    llvm::ArrayRef<MappingExpr> inverse) const {
  auto operand_inverse = operand().FindInInverse(inverse);
  if (llvm::isa<MappingUnknownExpr, MappingNoneExpr>(operand_inverse)) {
    return operand_inverse;
  }
  auto unstripe_expr = llvm::cast<MappingUnStripeExpr>(operand_inverse);
  return unstripe_expr.operands()[factors().size() - 1];
}

mlir::AffineExpr MappingStripeExpr::AsAffineExpr() const {
  int step = factors().back();
  return step * operand().AsAffineExpr().floorDiv(step);
}

static MappingExpr GetCanonicalStripe(MappingExpr canonical_operand,
                                      llvm::ArrayRef<int> factors) {
  if (factors.size() == 1 && factors.back() == 1) return canonical_operand;

  auto unstripe = llvm::dyn_cast<MappingUnStripeExpr>(canonical_operand);
  if (unstripe == nullptr) {
    return MappingStripeExpr::get(canonical_operand, factors);
  }

  auto it = std::mismatch(factors.begin(), factors.end(),
                          unstripe.factors().begin(), unstripe.factors().end());

  // If all factors match, stripe(unstripe) is the identity function.
  if (it.first == factors.end()) {
    return unstripe.operands()[factors.size() - 1];
  }

  // Otherwise, we trip common factors.
  int num_common = std::distance(factors.begin(), it.first);
  auto new_unstripe =
      MappingUnStripeExpr::get(unstripe.operands().drop_front(num_common),
                               unstripe.factors().drop_front(num_common));
  return MappingStripeExpr::get(new_unstripe, factors.drop_front(num_common));
}

MappingExpr MappingStripeExpr::Canonicalize() const {
  return GetCanonicalStripe(operand().Canonicalize(), factors());
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
  assert(stripes.size() == factors.size());
  assert(factors.back() == 1);
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

MappingExpr MappingUnStripeExpr::Map(
    llvm::function_ref<MappingExpr(MappingExpr)> function) const {
  auto new_operands = llvm::to_vector<4>(llvm::map_range(
      operands(), [&](MappingExpr expr) { return expr.Map(function); }));
  return function(MappingUnStripeExpr::get(new_operands, factors()));
}

void MappingUnStripeExpr::Walk(
    llvm::function_ref<void(MappingExpr)> function) const {
  for (MappingExpr operand : operands()) operand.Walk(function);
  function(*this);
}

mlir::LogicalResult MappingUnStripeExpr::SetInverse(
    MappingExpr context_inverse,
    llvm::MutableArrayRef<MappingExpr> inverses) const {
  for (int i = 0, e = factors().size(); i < e; ++i) {
    MappingExpr stripe_expr =
        MappingStripeExpr::get(context_inverse, factors().take_front(i + 1));
    if (mlir::failed(operands()[i].SetInverse(stripe_expr, inverses))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

MappingExpr MappingUnStripeExpr::Unify(
    MappingExpr other_expr,
    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)> on_mismatch)
    const {
  MappingUnStripeExpr other_unstripe =
      llvm::dyn_cast<MappingUnStripeExpr>(other_expr);
  if (other_unstripe == nullptr) return on_mismatch(*this, other_expr);

  llvm::SmallVector<MappingExpr> new_operands;
  llvm::ArrayRef<int> new_factors;

  // Operands and factors of the expression with a minimal number of factors
  // among this and other.
  llvm::ArrayRef<MappingExpr> min_operands;
  llvm::ArrayRef<int> min_factors;
  if (factors().size() >= other_unstripe.factors().size()) {
    llvm::append_range(new_operands, operands());
    new_factors = factors();
    min_operands = other_unstripe.operands();
    min_factors = other_unstripe.factors();
  } else {
    llvm::append_range(new_operands, other_unstripe.operands());
    new_factors = other_unstripe.factors();
    min_operands = operands();
    min_factors = factors();
  }

  // If the last operand is `none` or `?`, we can replace it by an arbitrary
  // number of operands.
  if (llvm::isa<MappingNoneExpr, MappingUnknownExpr>(min_operands.back())) {
    min_operands = min_operands.drop_back();
    min_factors = min_factors.drop_back();
  }

  // Ensure that the factors of one are a prefix of the factors of the other.
  if (min_factors != new_factors.take_front(min_factors.size())) {
    return on_mismatch(*this, other_expr);
  }

  for (int i = 0, e = min_operands.size(); i < e; ++i) {
    new_operands[i] = new_operands[i].Unify(min_operands[i], on_mismatch);
    if (new_operands[i] == nullptr) return MappingExpr();
  }

  return MappingUnStripeExpr::get(new_operands, new_factors);
}

MappingExpr MappingUnStripeExpr::FindInInverse(
    llvm::ArrayRef<MappingExpr> inverse) const {
  MappingExpr operand_inverse;
  for (int i = 0, e = operands().size(); i < e; ++i) {
    operand_inverse = operands()[i].FindInInverse(inverse);
    if (llvm::isa<MappingUnknownExpr, MappingNoneExpr>(operand_inverse)) continue;
    return llvm::cast<MappingStripeExpr>(operand_inverse).operand();
  }
  // Unstripe has at least one operand.
  return operand_inverse;
}

mlir::AffineExpr MappingUnStripeExpr::AsAffineExpr() const {
  return operands().back().AsAffineExpr();
}

MappingExpr MappingUnStripeExpr::Canonicalize() const {
  llvm::SmallVector<MappingExpr> new_operands;
  new_operands.reserve(operands().size());
  for (MappingExpr operand : operands()) {
    new_operands.push_back(operand.Canonicalize());
  }
  llvm::SmallVector<int> new_factors;
  llvm::append_range(new_factors, factors());

  // Use lambdas to break the control flow. Each lambda returns true if the
  // corresponding canonicalization rule was applied.

  // If the last argument is an unstripe, it can be collapsed in the current
  // expression.
  auto collapse_unstripes = [&]() {
    auto unstripe = llvm::dyn_cast<MappingUnStripeExpr>(new_operands.back());
    if (unstripe == nullptr) return false;
    // Stripe factors must be strictly decreasing.
    if (new_factors.size() > 1 &&
        new_factors[new_factors.size() - 2] <= unstripe.factors().front()) {
      return false;
    }
    new_operands.pop_back();
    new_factors.pop_back();
    llvm::append_range(new_operands, unstripe.operands());
    llvm::append_range(new_factors, unstripe.factors());
    return true;
  };

  // Stiches stripe expressions that have the same operand.
  auto stiche_stripes = [&]() {
    auto stripe = llvm::dyn_cast<MappingStripeExpr>(new_operands.back());
    if (stripe == nullptr) return false;
    int min_num_factors = std::min(new_factors.size(), stripe.factors().size());
    // Ensure factors are the same.
    if (llvm::ArrayRef(new_factors).take_back(min_num_factors) !=
        stripe.factors().take_back(min_num_factors)) {
      return false;
    }

    // Find how many stripes we can stich together.
    int first_stripe = new_operands.size() - 1;
    for (; first_stripe > 0; --first_stripe) {
      auto other_stripe =
          llvm::dyn_cast<MappingStripeExpr>(new_operands[first_stripe - 1]);
      if (other_stripe == nullptr ||
          other_stripe.operand() != stripe.operand()) {
        break;
      }
    }

    // Only one stripe, we can't stich anything.
    if (first_stripe == new_operands.size() - 1) return false;

    llvm::SmallVector<int> new_stripe_factors;
    llvm::append_range(
        new_stripe_factors,
        stripe.factors().drop_back(new_operands.size() - first_stripe));
    new_stripe_factors.push_back(1);

    new_operands.resize(first_stripe);
    new_factors.resize(first_stripe);
    new_operands.push_back(
        GetCanonicalStripe(stripe.operand(), new_stripe_factors));
    new_factors.push_back(1);
    return true;
  };

  // Apply canonicalization rules.
  while (collapse_unstripes() || stiche_stripes()) {
  }

  if (new_factors.size() == 1 && new_factors.back() == 1)
    return new_operands[0];
  return MappingUnStripeExpr::get(new_operands, new_factors);
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
        llvm::cast<mlir::AffineDimExpr>(expr).getPosition(), map.getContext()));
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

bool MappingAttr::HasNoneExprs() const {
  return llvm::any_of(getImpl()->mapping(),
                      [](MappingExpr expr) { return expr.HasNoneExprs(); });
}

bool MappingAttr::HasUnknownExprs() const {
  return llvm::any_of(getImpl()->mapping(),
                      [](MappingExpr expr) { return expr.HasUnknownExprs(); });
}

MappingAttr MappingAttr::MakeSurjective() const {
  int num_dimensions = UseDomainSize();
  llvm::SmallVector<MappingExpr, 4> new_exprs;
  new_exprs.reserve(size());
  for (MappingExpr expr : Dimensions()) {
    MappingExpr new_expr = expr.Map([&](MappingExpr sub_expr) -> MappingExpr {
      if (!llvm::isa<MappingNoneExpr>(sub_expr)) return sub_expr;
      return MappingDimExpr::get(num_dimensions++, getContext());
    });
    new_exprs.push_back(new_expr);
  }
  return MappingAttr::get(getContext(), num_dimensions, new_exprs);
}

MappingAttr MappingAttr::MakeFullySpecified() const {
  auto none = MappingNoneExpr::get(getContext());
  auto new_exprs =
      llvm::to_vector<4>(llvm::map_range(Dimensions(), [&](auto expr) {
        return expr.Map([&](MappingExpr sub_expr) -> MappingExpr {
          return llvm::isa<MappingUnknownExpr>(sub_expr) ? none : sub_expr;
        });
      }));
  return MappingAttr::get(getContext(), UseDomainSize(), new_exprs);
}

bool MappingAttr::IsIdentity() const {
  for (auto en : llvm::enumerate(getImpl()->mapping())) {
    auto dim_expr = llvm::dyn_cast<MappingDimExpr>(en.value());
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
    (void)status;
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

int MappingAttr::MinDomainSize() const {
  int min = 0;
  for (MappingExpr expr : Dimensions()) {
    min = std::max(min, expr.MinDomainSize());
  }
  return min;
}

MappingAttr MappingAttr::Unify(MappingAttr other) const {
  assert(size() == other.size());
  assert(UseDomainSize() == other.UseDomainSize());
  llvm::SmallVector<MappingExpr> exprs;
  exprs.reserve(size());
  for (auto [x, y] : llvm::zip(Dimensions(), other.Dimensions())) {
    exprs.push_back(sair::Unify(x, y));
    if (exprs.back() == nullptr) return nullptr;
  }
  return MappingAttr::get(getContext(), UseDomainSize(), exprs);
}

MappingAttr MappingAttr::UnifyUnknownExprs(MappingAttr other) const {
  assert(size() == other.size());
  assert(UseDomainSize() == other.UseDomainSize());
  llvm::SmallVector<MappingExpr> exprs;
  exprs.reserve(size());
  for (auto [lhs, rhs] : llvm::zip(Dimensions(), other.Dimensions())) {
    MappingExpr unified =
        lhs.Unify(rhs, [](MappingExpr sub_lhs, MappingExpr sub_rhs) {
          if (llvm::isa<MappingUnknownExpr>(sub_lhs)) return sub_rhs;
          if (llvm::isa<MappingUnknownExpr>(sub_rhs)) return sub_lhs;
          return MappingExpr();
        });
    if (unified == nullptr) return nullptr;
    exprs.push_back(unified);
  }
  return MappingAttr::get(getContext(), UseDomainSize(), exprs);
}

MappingAttr MappingAttr::AddPrefix(llvm::ArrayRef<MappingExpr> exprs) const {
  llvm::SmallVector<MappingExpr> new_exprs;
  new_exprs.reserve(exprs.size() + size());
  llvm::append_range(new_exprs, exprs);
  llvm::append_range(new_exprs, Dimensions());
  return MappingAttr::get(getContext(), UseDomainSize(), new_exprs);
}

MappingAttr MappingAttr::AddSuffix(llvm::ArrayRef<MappingExpr> exprs) const {
  llvm::SmallVector<MappingExpr> new_exprs;
  new_exprs.reserve(exprs.size() + size());
  llvm::append_range(new_exprs, Dimensions());
  llvm::append_range(new_exprs, exprs);
  return MappingAttr::get(getContext(), UseDomainSize(), new_exprs);
}

MappingAttr MappingAttr::Slice(int begin, int new_size) const {
  assert(begin >= 0);
  assert(new_size >= 0);
  assert(begin + new_size <= size());

  return MappingAttr::get(getContext(), UseDomainSize(),
                          Dimensions().slice(begin, new_size));
}

MappingAttr MappingAttr::DropFront(int num_drop) const {
  return Slice(num_drop, size() - num_drop);
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
      : names_(key.first), mapping_(llvm::cast<MappingAttr>(key.second)) {}

  llvm::ArrayRef<mlir::StringAttr> names_;
  MappingAttr mapping_;
};

NamedMappingAttr NamedMappingAttr::get(llvm::ArrayRef<mlir::StringAttr> names,
                                       MappingAttr mapping) {
  assert(names.size() == mapping.UseDomainSize());
  return Base::get(mapping.getContext(), names, mapping);
}
NamedMappingAttr NamedMappingAttr::get(llvm::ArrayRef<mlir::StringAttr> names,
                                       llvm::ArrayRef<MappingExpr> exprs,
                                       mlir::MLIRContext *context) {
  return get(names, MappingAttr::get(context, names.size(), exprs));
}

NamedMappingAttr NamedMappingAttr::GetIdentity(
    mlir::MLIRContext *context, llvm::ArrayRef<mlir::StringAttr> names) {
  auto mapping = MappingAttr::GetIdentity(context, names.size());
  return NamedMappingAttr::get(names, mapping);
}

llvm::ArrayRef<mlir::StringAttr> NamedMappingAttr::names() const {
  return getImpl()->names();
}

MappingAttr NamedMappingAttr::mapping() const { return getImpl()->mapping(); }

NamedMappingAttr NamedMappingAttr::DropUnusedDims() const {
  mlir::MLIRContext *context = getContext();

  auto none = MappingNoneExpr::get(context);
  llvm::SmallVector<MappingExpr> subsitutions(names().size(), none);
  llvm::SmallBitVector used_dims = mapping().DependencyMask();
  llvm::SmallVector<mlir::StringAttr> new_dim_names;

  new_dim_names.reserve(used_dims.count());
  for (int dim : used_dims.set_bits()) {
    subsitutions[dim] = MappingDimExpr::get(new_dim_names.size(), context);
    new_dim_names.push_back(names()[dim]);
  }

  auto new_mapping =
      MappingAttr::get(context, new_dim_names.size(), subsitutions)
          .Compose(mapping());
  return NamedMappingAttr::get(new_dim_names, new_mapping);
}

NamedMappingAttr NamedMappingAttr::Compose(MappingAttr other) const {
  return NamedMappingAttr::get(names(), mapping().Compose(other));
}

//===----------------------------------------------------------------------===//
// DomainShapeDim
//===----------------------------------------------------------------------===//

DomainShapeDim::DomainShapeDim(DimensionType type,
                               MappingAttr dependency_mapping)
    : type_(type), dependency_mapping_(dependency_mapping) {
  assert(type != nullptr);
  assert(dependency_mapping != nullptr);
}

DomainShapeDim DomainShapeDim::Apply(MappingAttr mapping) const {
  return DomainShapeDim(type_,
                        mapping.Compose(dependency_mapping_).Canonicalize());
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
#ifndef NDEBUG
  for (int i = 0, e = dims.size(); i < e; ++i) {
    MappingAttr mapping = dims[i].dependency_mapping();
    assert(mapping.UseDomainSize() == i);
    assert(mapping.IsSurjective());
    assert(mapping.IsFullySpecified());
  }
#endif

  return Base::get(context, dims);
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

// Computes the shape of a dimension resulting from applying expr to shape.
// `inverted_mapping` must be the inverse of the mapping expr is taken from.
static DomainShapeDim AccessedShape(MappingExpr expr,
                                    MappingAttr inverted_mapping,
                                    DomainShapeAttr shape);

DomainShapeAttr DomainShapeAttr::AccessedShape(MappingAttr mapping) const {
  assert(mapping.IsFullySpecified());
  assert(mapping.IsSurjective());

  llvm::SmallVector<DomainShapeDim, 4> shape;
  shape.reserve(mapping.size());
  MappingAttr inverted_mapping = mapping.Inverse();
  for (int i = 0, e = mapping.size(); i < e; ++i) {
    DomainShapeDim shape_dim = sair::AccessedShape(
        mapping.Dimension(i), inverted_mapping.ResizeUseDomain(i), *this);
    assert(shape_dim.dependency_mapping().UseDomainSize() == i);
    shape.push_back(shape_dim);
  }
  return DomainShapeAttr::get(getContext(), shape);
}

// Compute the shape of a dimension mapped by a stripe expression.
static DomainShapeDim StripeAccessedShape(MappingStripeExpr expr,
                                          DomainShapeDim inner_shape,
                                          MappingAttr inverted_mapping) {
  mlir::MLIRContext *context = expr.getContext();
  auto inverse_subexpr =
      expr.operand().FindInInverse(inverted_mapping.Dimensions());
  auto unstripe_expr = llvm::dyn_cast<MappingUnStripeExpr>(inverse_subexpr);

  // Append dependencies to larger stripes to the dependency mapping.
  llvm::SmallVector<MappingExpr, 4> dependency_mapping_exprs;
  llvm::append_range(dependency_mapping_exprs,
                     inner_shape.dependency_mapping());

  llvm::SmallVector<DomainShapeDim, 4> type_shape;
  llvm::append_range(type_shape, inner_shape.type().Shape().Dimensions());
  DimensionType type = inner_shape.type();

  // Handle the case where we are striping a static range and the result is also
  // a static range.
  if (expr.factors().size() == 1) {
    if (auto static_range = llvm::dyn_cast<StaticRangeType>(type)) {
      int new_step = static_range.getStep() * expr.factors().front();
      type = StaticRangeType::get(static_range.size(), new_step, context);
    }
  }

  for (int i = 0, e = expr.factors().size() - 1; i < e; ++i) {
    type_shape.emplace_back(
        type, MappingAttr::GetIdentity(context, type_shape.size()));
    type = DynRangeType::get(DomainShapeAttr::get(context, type_shape));
    if (unstripe_expr == nullptr) {
      dependency_mapping_exprs.push_back(inverse_subexpr);
    } else {
      dependency_mapping_exprs.push_back(unstripe_expr.operands()[i]);
    }
  }

  auto dependency_mapping = MappingAttr::get(
      context, inverted_mapping.UseDomainSize(), dependency_mapping_exprs);
  return DomainShapeDim(type, dependency_mapping);
}

// Compute the shape of a dimension mapped by an ustripe expression.
static DomainShapeDim UnStripeAccessedShape(MappingUnStripeExpr expr,
                                            DomainShapeDim inner_shape,
                                            MappingAttr inverted_mapping) {
  if (llvm::isa<DynRangeType>(inner_shape.type())) return inner_shape;
  auto type = llvm::cast<StaticRangeType>(inner_shape.type());
  int new_step = type.getStep() / expr.factors().front();
  return DomainShapeDim(
      StaticRangeType::get(type.size(), new_step, expr.getContext()),
      inner_shape.dependency_mapping());
}

static DomainShapeDim AccessedShape(MappingExpr expr,
                                    MappingAttr inverted_mapping,
                                    DomainShapeAttr shape) {
  return mlir::TypeSwitch<MappingExpr, DomainShapeDim>(expr)
      .Case<MappingDimExpr>([&](MappingDimExpr expr) {
        return shape.Dimension(expr.dimension()).Apply(inverted_mapping);
      })
      .Case<MappingStripeExpr>([&](MappingStripeExpr expr) {
        DomainShapeDim inner =
            AccessedShape(expr.operand(), inverted_mapping, shape);
        return StripeAccessedShape(expr, inner, inverted_mapping);
      })
      .Case<MappingUnStripeExpr>([&](MappingUnStripeExpr expr) {
        DomainShapeDim inner =
            AccessedShape(expr.operands().front(), inverted_mapping, shape);
        return UnStripeAccessedShape(expr, inner, inverted_mapping);
      });
}

// Ensure that expr and its sub-expressions have a valid shape. Inverse must be
// the inverse of the full mapping and shape the shape of the source domain.
// Stores the shape of the expression in `expr_shape` if it is known.
static mlir::LogicalResult VerifyMappingExprShape(
    AttrLocation loc, MappingExpr expr, MappingAttr inverse,
    DomainShapeAttr shape, std::optional<DomainShapeDim> &expr_shape);

static mlir::LogicalResult VerifyMappingExprShape(
    AttrLocation loc, MappingStripeExpr expr, MappingAttr inverse,
    DomainShapeAttr shape, std::optional<DomainShapeDim> &expr_shape) {
  std::optional<DomainShapeDim> inner_shape;
  if (mlir::failed(VerifyMappingExprShape(loc, expr.operand(), inverse, shape,
                                          inner_shape))) {
    return mlir::failure();
  }
  if (!inner_shape.has_value()) return mlir::success();
  expr_shape = StripeAccessedShape(expr, inner_shape.value(), inverse);
  return mlir::success();
}

static mlir::LogicalResult VerifyMappingExprShape(
    AttrLocation loc, MappingUnStripeExpr expr, MappingAttr inverse,
    DomainShapeAttr shape, std::optional<DomainShapeDim> &expr_shape) {
  for (auto index_operand : llvm::enumerate(expr.operands())) {
    int i = index_operand.index();
    std::optional<DomainShapeDim> inner_shape;
    if (mlir::failed(VerifyMappingExprShape(loc, index_operand.value(), inverse,
                                            shape, inner_shape))) {
      return mlir::failure();
    }
    if (!inner_shape.has_value()) continue;
    if (inner_shape->dependency_mapping().size() < i) {
      return loc.EmitError() << "operand " << i << " of unstripe in " << expr
                             << " has an invalid shape";
    }
    if (i == 0) expr_shape = inner_shape;
  }
  return mlir::success();
}

static mlir::LogicalResult VerifyMappingExprShape(
    AttrLocation loc, MappingExpr expr, MappingAttr inverse,
    DomainShapeAttr shape, std::optional<DomainShapeDim> &expr_shape) {
  return mlir::TypeSwitch<MappingExpr, mlir::LogicalResult>(expr)
      .Case<MappingDimExpr>([&](MappingDimExpr expr) {
        expr_shape = shape.Dimension(expr.dimension()).Apply(inverse);
        return mlir::success();
      })
      .Case<MappingStripeExpr>([&](MappingStripeExpr expr) {
        return VerifyMappingExprShape(loc, expr, inverse, shape, expr_shape);
      })
      .Case<MappingUnStripeExpr>([&](MappingUnStripeExpr expr) {
        return VerifyMappingExprShape(loc, expr, inverse, shape, expr_shape);
      })
      .Default([](auto) { return mlir::success(); });
}

mlir::LogicalResult VerifyMappingShape(const AttrLocation &loc,
                                       MappingAttr mapping,
                                       DomainShapeAttr shape) {
  MappingAttr inverse = mapping.Inverse();
  for (int i = 0, e = mapping.size(); i < e; ++i) {
    std::optional<DomainShapeDim> expr_shape;
    if (mlir::failed(VerifyMappingExprShape(loc, mapping.Dimension(i), inverse,
                                            shape, expr_shape))) {
      return mlir::failure();
    }
    if (!expr_shape.has_value()) continue;
    int min_domain_size = expr_shape->dependency_mapping().MinDomainSize();
    if (min_domain_size > i) {
      return loc.EmitError()
             << "dimension " << i << " of the mapping depends on dimension "
             << min_domain_size - 1 << " of the mapping";
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DecisionsAttr
//===----------------------------------------------------------------------===//

std::function<DecisionsAttr(DecisionsAttr)> MapLoopNest(
    std::function<mlir::ArrayAttr(mlir::ArrayAttr)> loop_nest_fn) {
  return [loop_nest_fn](DecisionsAttr decisions) -> DecisionsAttr {
    if (decisions == nullptr) return nullptr;
    return DecisionsAttr::get(
        decisions.sequence(), loop_nest_fn(decisions.loop_nest()),
        decisions.storage(), decisions.expansion(), decisions.copy_of(),
        decisions.operands(), decisions.getContext());
  };
}

std::function<DecisionsAttr(DecisionsAttr)> MapStorage(
    std::function<mlir::ArrayAttr(mlir::ArrayAttr)> storage_fn) {
  return [storage_fn](DecisionsAttr decisions) -> DecisionsAttr {
    if (decisions == nullptr) return nullptr;
    return DecisionsAttr::get(decisions.sequence(), decisions.loop_nest(),
                              storage_fn(decisions.storage()),
                              decisions.expansion(), decisions.copy_of(),
                              decisions.operands(), decisions.getContext());
  };
}

DecisionsAttr UpdateSequence(DecisionsAttr decisions, int new_sequence) {
  mlir::MLIRContext *context = decisions.getContext();
  mlir::IntegerAttr new_sequence_attr =
      OpBuilder(context).getI64IntegerAttr(new_sequence);
  return DecisionsAttr::get(new_sequence_attr, decisions.loop_nest(),
                            decisions.storage(), decisions.expansion(),
                            decisions.copy_of(), decisions.operands(), context);
}

DecisionsAttr UpdateOperands(DecisionsAttr decisions,
                             llvm::ArrayRef<mlir::Attribute> operands) {
  mlir::MLIRContext *context = decisions.getContext();
  return DecisionsAttr::get(decisions.sequence(), decisions.loop_nest(),
                            decisions.storage(), decisions.expansion(),
                            decisions.copy_of(),
                            mlir::ArrayAttr::get(context, operands), context);
}

mlir::ArrayAttr GetInstanceZeroOperands(mlir::MLIRContext *context,
                                        int num_operands) {
  llvm::SmallVector<mlir::Attribute> attributes(num_operands,
                                                InstanceAttr::get(context, 0));
  return mlir::ArrayAttr::get(context, attributes);
}

mlir::ArrayAttr GetInstanceZeroOperandsSingleInstance(
    mlir::MLIRContext *context, int num_operands) {
  auto decisions = DecisionsAttr::get(
      /*sequence=*/nullptr, /*loop_nest=*/nullptr, /*storage=*/nullptr,
      /*expansion=*/nullptr, /*copy_of=*/nullptr,
      GetInstanceZeroOperands(context, num_operands), context);
  return mlir::ArrayAttr::get(context, {decisions});
}

mlir::ArrayAttr EraseOperandFromArray(mlir::ArrayAttr old_operands,
                                      int operand) {
  if (old_operands == nullptr) return nullptr;
  assert(operand < old_operands.size());

  llvm::SmallVector<mlir::Attribute> operands;
  operands.reserve(old_operands.size() - 1);
  llvm::append_range(operands, old_operands.getValue().take_front(operand));
  llvm::append_range(operands, old_operands.getValue().drop_front(operand + 1));
  return mlir::ArrayAttr::get(old_operands.getContext(), operands);
}

DecisionsAttr EraseOperand(DecisionsAttr decisions, int operand) {
  return DecisionsAttr::get(
      decisions.sequence(), decisions.loop_nest(), decisions.storage(),
      decisions.expansion(), decisions.copy_of(),
      EraseOperandFromArray(decisions.operands(), operand),
      decisions.getContext());
}

mlir::ArrayAttr EraseOperandFromDecisions(mlir::ArrayAttr decisions,
                                          int operand) {
  if (decisions == nullptr) return decisions;

  auto range =
      llvm::map_range(decisions.getAsRange<DecisionsAttr>(),
                      [operand](DecisionsAttr decisions) -> mlir::Attribute {
                        return EraseOperand(decisions, operand);
                      });
  return mlir::ArrayAttr::get(decisions.getContext(),
                              llvm::to_vector<1>(range));
}

LoopAttr LoopAttr::get(mlir::StringAttr name, MappingExpr iter,
                       mlir::IntegerAttr unroll, mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::NamedAttribute, 3> fields;
  assert(name);
  auto name_id = mlir::StringAttr::get(context, "name");
  fields.emplace_back(name_id, name);

  assert(iter);
  auto iter_id = mlir::StringAttr::get(context, "iter");
  fields.emplace_back(iter_id, iter);

  if (unroll) {
    auto unroll_id = mlir::StringAttr::get(context, "unroll");
    fields.emplace_back(unroll_id, unroll);
  }

  mlir::Attribute dict = mlir::DictionaryAttr::get(context, fields);
  return llvm::dyn_cast<LoopAttr>(dict);
}

bool LoopAttr::classof(mlir::Attribute attr) {
  if (!attr) return false;
  auto derived = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
  if (!derived) return false;

  auto name = derived.get("name");
  if (!llvm::isa_and_nonnull<mlir::StringAttr>(name)) return false;

  auto iter = derived.get("iter");
  if (!llvm::isa_and_nonnull<sair::MappingExpr>(iter)) return false;

  auto unroll = derived.get("unroll");
  if (!unroll) return derived.size() == 2;

  auto intUnroll = llvm::dyn_cast<mlir::IntegerAttr>(unroll);
  if (!intUnroll || !intUnroll.getType().isSignlessInteger(64) ||
      !intUnroll.getValue().isStrictlyPositive()) {
    return false;
  }

  return derived.size() == 3;
}

mlir::StringAttr LoopAttr::name() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto name = derived.get("name");
  assert(name && "attribute not found.");
  assert(llvm::isa<mlir::StringAttr>(name) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::StringAttr>(name);
}

MappingExpr LoopAttr::iter() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto iter = derived.get("iter");
  assert(iter && "attribute not found.");
  assert(llvm::isa<MappingExpr>(iter) && "incorrect Attribute type found.");
  return llvm::cast<MappingExpr>(iter);
}

mlir::IntegerAttr LoopAttr::unroll() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto unroll = derived.get("unroll");
  if (!unroll) return nullptr;
  assert(llvm::isa<mlir::IntegerAttr>(unroll) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::IntegerAttr>(unroll);
}

BufferAttr BufferAttr::get(mlir::StringAttr space, mlir::StringAttr name,
                           NamedMappingAttr layout,
                           mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::NamedAttribute, 3> fields;

  assert(space);
  auto space_id = mlir::StringAttr::get(context, "space");
  fields.emplace_back(space_id, space);

  if (name) {
    auto name_id = mlir::StringAttr::get(context, "name");
    fields.emplace_back(name_id, name);
  }

  if (layout) {
    auto layout_id = mlir::StringAttr::get(context, "layout");
    fields.emplace_back(layout_id, layout);
  }

  mlir::Attribute dict = mlir::DictionaryAttr::get(context, fields);
  return llvm::dyn_cast<BufferAttr>(dict);
}

bool BufferAttr::classof(mlir::Attribute attr) {
  if (!attr) return false;
  auto derived = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
  if (!derived) return false;
  int num_absent_attrs = 0;

  auto space = derived.get("space");
  if (!llvm::isa_and_nonnull<mlir::StringAttr>(space)) return false;

  auto name = derived.get("name");
  if (!name) {
    ++num_absent_attrs;
  } else if (!llvm::isa<mlir::StringAttr>(name)) {
    return false;
  }

  auto layout = derived.get("layout");
  if (!layout) {
    ++num_absent_attrs;
  } else if (!llvm::isa<NamedMappingAttr>(layout)) {
    return false;
  }

  return derived.size() + num_absent_attrs == 3;
}

mlir::StringAttr BufferAttr::space() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto space = derived.get("space");
  assert(space && "attribute not found.");
  assert(llvm::isa<mlir::StringAttr>(space) && "incorrect Attribute type found.");
  return llvm::cast<mlir::StringAttr>(space);
}

mlir::StringAttr BufferAttr::name() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto name = derived.get("name");
  if (!name) return nullptr;
  assert(llvm::isa<mlir::StringAttr>(name) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::StringAttr>(name);
}

NamedMappingAttr BufferAttr::layout() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto layout = derived.get("layout");
  if (!layout) return nullptr;
  assert(llvm::isa<NamedMappingAttr>(layout) &&
         "incorrect Attribute type found.");
  return llvm::cast<NamedMappingAttr>(layout);
}

DecisionsAttr DecisionsAttr::get(mlir::IntegerAttr sequence,
                                 mlir::ArrayAttr loop_nest,
                                 mlir::ArrayAttr storage,
                                 mlir::StringAttr expansion,
                                 mlir::Attribute copy_of,
                                 mlir::ArrayAttr operands,
                                 mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::NamedAttribute, 6> fields;

  if (sequence) {
    auto sequence_id = mlir::StringAttr::get(context, "sequence");
    fields.emplace_back(sequence_id, sequence);
  }

  if (loop_nest) {
    auto loop_nest_id = mlir::StringAttr::get(context, "loop_nest");
    fields.emplace_back(loop_nest_id, loop_nest);
  }

  if (storage) {
    auto storage_id = mlir::StringAttr::get(context, "storage");
    fields.emplace_back(storage_id, storage);
  }

  if (expansion) {
    auto expansion_id = mlir::StringAttr::get(context, "expansion");
    fields.emplace_back(expansion_id, expansion);
  }

  if (copy_of) {
    auto copy_of_id = mlir::StringAttr::get(context, "copy_of");
    fields.emplace_back(copy_of_id, copy_of);
  }

  if (operands) {
    auto operands_id = mlir::StringAttr::get(context, "operands");
    fields.emplace_back(operands_id, operands);
  }

  mlir::Attribute dict = mlir::DictionaryAttr::get(context, fields);
  return llvm::dyn_cast<DecisionsAttr>(dict);
}

bool DecisionsAttr::classof(mlir::Attribute attr) {
  if (!attr) return false;
  auto derived = llvm::dyn_cast<mlir::DictionaryAttr>(attr);
  if (!derived) return false;
  int num_absent_attrs = 0;

  auto sequence = derived.get("sequence");
  if (!sequence) {
    ++num_absent_attrs;
  } else {
    auto int_sequence = llvm::dyn_cast<mlir::IntegerAttr>(sequence);
    if (!int_sequence || !int_sequence.getType().isSignlessInteger(64)) {
      return false;
    }
  }

  auto loop_nest = derived.get("loop_nest");
  if (!loop_nest) {
    ++num_absent_attrs;
  } else {
    auto loop_nest_attr = llvm::dyn_cast<mlir::ArrayAttr>(loop_nest);
    if (!loop_nest_attr) return false;
    if (llvm::any_of(loop_nest_attr, [](mlir::Attribute attr) {
          return !llvm::isa_and_nonnull<LoopAttr>(attr);
        })) {
      return false;
    }
  }

  auto storage = derived.get("storage");
  if (!storage) {
    ++num_absent_attrs;
  } else if (!llvm::isa<mlir::ArrayAttr>(storage)) {
    return false;
  }

  auto expansion = derived.get("expansion");
  if (!expansion) {
    ++num_absent_attrs;
  } else if (!llvm::isa<mlir::StringAttr>(expansion)) {
    return false;
  }

  auto copy_of = derived.get("copy_of");
  if (!copy_of) {
    ++num_absent_attrs;
  } else if (!llvm::isa<CopyAttr, InstanceAttr, mlir::UnitAttr>(copy_of)) {
    return false;
  }

  auto operands = derived.get("operands");
  if (!operands) {
    ++num_absent_attrs;
  } else {
    auto operands_attr = llvm::dyn_cast<mlir::ArrayAttr>(operands);
    if (llvm::any_of(operands_attr, [](mlir::Attribute attr) {
          return !llvm::isa_and_nonnull<CopyAttr, InstanceAttr, mlir::UnitAttr>(
              attr);
        })) {
      return false;
    }
  }

  return derived.size() + num_absent_attrs == 6;
}

mlir::IntegerAttr DecisionsAttr::sequence() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto sequence = derived.get("sequence");
  if (!sequence) return nullptr;
  assert(llvm::isa<mlir::IntegerAttr>(sequence) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::IntegerAttr>(sequence);
}

mlir::ArrayAttr DecisionsAttr::loop_nest() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto loop_nest = derived.get("loop_nest");
  if (!loop_nest) return nullptr;
  assert(llvm::isa<mlir::ArrayAttr>(loop_nest) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::ArrayAttr>(loop_nest);
}

mlir::ArrayAttr DecisionsAttr::storage() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto storage = derived.get("storage");
  if (!storage) return nullptr;
  assert(llvm::isa<mlir::ArrayAttr>(storage) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::ArrayAttr>(storage);
}

mlir::StringAttr DecisionsAttr::expansion() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto expansion = derived.get("expansion");
  if (!expansion) return nullptr;
  assert(llvm::isa<mlir::StringAttr>(expansion) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::StringAttr>(expansion);
}

mlir::Attribute DecisionsAttr::copy_of() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto copy_of = derived.get("copy_of");
  if (!copy_of) return nullptr;
  assert(llvm::isa<mlir::Attribute>(copy_of) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::Attribute>(copy_of);
}

mlir::ArrayAttr DecisionsAttr::operands() const {
  auto derived = llvm::cast<mlir::DictionaryAttr>(*this);
  auto operands = derived.get("operands");
  if (!operands) return nullptr;
  assert(llvm::isa<mlir::ArrayAttr>(operands) &&
         "incorrect Attribute type found.");
  return llvm::cast<mlir::ArrayAttr>(operands);
}

}  // namespace sair

#define GET_ATTRDEF_CLASSES
#include "sair_attributes.cc.inc"

mlir::OptionalParseResult sair::detail::ParseGeneratedAttribute(
    mlir::MLIRContext *context, mlir::AsmParser &parser,
    llvm::StringRef *mnemonic, mlir::Type type, mlir::Attribute &attribute) {
  return generatedAttributeParser(parser, mnemonic, type, attribute);
}

mlir::LogicalResult sair::detail::PrintGeneratedAttribute(
    mlir::Attribute attribute, mlir::AsmPrinter &printer) {
  return generatedAttributePrinter(attribute, printer);
}

//===----------------------------------------------------------------------===//
// SairDialect
//===----------------------------------------------------------------------===//

namespace sair {
void SairDialect::registerAttributes() {
  addAttributes<DomainShapeAttr, MappingAttr, NamedMappingAttr, MappingDimExpr,
                MappingNoneExpr, MappingUnknownExpr, MappingStripeExpr,
                MappingUnStripeExpr>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "sair_attributes.cc.inc"
      >();
}
}  // namespace sair
