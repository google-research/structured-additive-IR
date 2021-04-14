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
#include "sair_dialect.h"
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
// MappingExpr
//===----------------------------------------------------------------------===//

MappingExpr MappingExpr::SubstituteDims(
    mlir::ArrayRef<MappingExpr> exprs) const {
  return Map([&](MappingExpr sub_expr) -> MappingExpr {
    auto dim_expr = sub_expr.dyn_cast<MappingDimExpr>();
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
    has_none_exprs |= sub_expr.isa<MappingNoneExpr>();
  });
  return has_none_exprs;
}

void MappingExpr::SetDependenciesInMask(llvm::SmallBitVector &mask) const {
  Walk([&](MappingExpr sub_expr) {
    auto dim_expr = sub_expr.dyn_cast<MappingDimExpr>();
    if (dim_expr == nullptr) return;
    mask.set(dim_expr.dimension());
  });
}

int MappingExpr::MinDomainSize() const {
  int min_domain_size = 0;
  Walk([&](MappingExpr sub_expr) {
    auto dim_expr = sub_expr.dyn_cast<MappingDimExpr>();
    if (dim_expr == nullptr) return;
    min_domain_size = std::max(min_domain_size, dim_expr.dimension() + 1);
  });
  return min_domain_size;
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
    mlir::Location loc, llvm::ArrayRef<ValueAccess> domain,
    MappingAttr inverse_mapping, mlir::OpBuilder &builder,
    MapArguments &map_arguments) const {
  const ValueAccess &dim_access = domain[dimension()];
  auto range_op = mlir::cast<RangeOp>(dim_access.value.getDefiningOp());
  auto mapping =
      dim_access.mapping.ResizeUseDomain(map_arguments.Indices().size());
  assert(mapping.IsSurjective());
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

MappingExpr MappingNoneExpr::Map(
    llvm::function_ref<MappingExpr(MappingExpr)> function) const {
  return function(*this);
}

void MappingNoneExpr::Walk(
    llvm::function_ref<void(MappingExpr)> function) const {
  function(*this);
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
        MappingStripeExprStorage(key.first, allocator.copyInto(key.second));
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

MappingExpr MappingStripeExpr::Unify(MappingExpr other_expr) const {
  if (other_expr.isa<MappingNoneExpr>()) return *this;
  MappingStripeExpr other_stripe = other_expr.dyn_cast<MappingStripeExpr>();
  if (other_stripe == nullptr || factors() != other_stripe.factors()) {
    return MappingExpr();
  }
  MappingExpr unified_operand = operand().Unify(other_stripe.operand());
  if (unified_operand == nullptr) return MappingExpr();
  return MappingStripeExpr::get(unified_operand, factors());
}

mlir::LogicalResult MappingStripeExpr::UnificationConstraints(
    MappingExpr other_expr,
    llvm::MutableArrayRef<MappingExpr> constraints) const {
  if (other_expr.isa<MappingNoneExpr>()) return mlir::success();
  auto other_stripe = other_expr.dyn_cast<MappingStripeExpr>();
  if (other_stripe == nullptr || factors() != other_stripe.factors()) {
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

  for (int i = 0, e = factors().size() - 1; i < e; ++i) {
    type_shape.emplace_back(
        type, MappingAttr::GetIdentity(context, type_shape.size()));
    type = RangeType::get(DomainShapeAttr::get(context, type_shape));
    dependency_mapping_exprs.push_back(inverse_subexpr.operands()[i]);
  }

  auto dependency_mapping = MappingAttr::get(
      context, inverted_mapping.UseDomainSize(), dependency_mapping_exprs);
  return DomainShapeDim(type, dependency_mapping);
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
  auto unstripe_expr =
      operand().FindInInverse(inverse).cast<MappingUnStripeExpr>();
  return unstripe_expr.operands()[factors().size() - 1];
}

mlir::AffineExpr MappingStripeExpr::AsAffineExpr() const {
  int step = factors().back();
  return step * operand().AsAffineExpr().floorDiv(step);
}

static MappingExpr GetCanonicalStripe(MappingExpr canonical_operand,
                                        llvm::ArrayRef<int> factors) {
  if (factors.size() == 1 && factors.back() == 1) return canonical_operand;

  auto unstripe = canonical_operand.dyn_cast<MappingUnStripeExpr>();
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

RangeParameters MappingStripeExpr::GetRangeParameters(
    mlir::Location loc, llvm::ArrayRef<ValueAccess> domain,
    MappingAttr inverse_mapping, mlir::OpBuilder &builder,
    MapArguments &map_arguments) const {
  // Compute range parameters for the operand.
  RangeParameters operand_parameters = operand().GetRangeParameters(
      loc, domain, inverse_mapping, builder, map_arguments);
  int step = factors().back() * operand_parameters.step;

  // If the stripe covers the entire operand range, no additional computation is
  // needed.
  if (factors().size() == 1) {
    return {operand_parameters.begin, operand_parameters.end, step};
  }
  int size = factors()[factors().size() - 2];

  // Compute the begin index. For this, look for the unstripe operation
  // corresponding to `this` in the inverse mapping, and find the expression of
  // the outer stripe dimension.
  auto inverse_expr = operand()
                          .FindInInverse(inverse_mapping.Dimensions())
                          .cast<MappingUnStripeExpr>();
  auto begin_map = mlir::AffineMap::get(
      map_arguments.Indices().size(), 0,
      inverse_expr.operands()[factors().size() - 2].AsAffineExpr());
  mlir::Value begin = builder.create<mlir::AffineApplyOp>(
      loc, begin_map, map_arguments.Indices());

  // Compute the end index as `min(begin + size, operand_size)`.
  mlir::Type index_type = builder.getIndexType();
  auto size_op = builder.create<mlir::ConstantOp>(
      loc, index_type, builder.getIndexAttr(size * operand_parameters.step));
  auto uncapped_end =
      builder.create<mlir::AddIOp>(loc, index_type, begin, size_op);
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
  for (int i = 0, e = factors().size(); i < e; ++i) {
    MappingExpr stripe_expr =
        MappingStripeExpr::get(context_inverse, factors().take_front(i + 1));
    if (mlir::failed(operands()[i].SetInverse(stripe_expr, inverses))) {
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

  // If the last operand is `none`, we can replace it by an arbitrary number of
  // operands.
  if (min_operands.back().isa<MappingNoneExpr>()) {
    min_operands = min_operands.drop_back();
    min_factors = min_factors.drop_back();
  }

  // Ensure that the factors of one are a prefix of the factors of the other.
  if (min_factors != new_factors.take_front(min_factors.size())) {
    return MappingExpr();
  }

  for (int i = 0, e = min_operands.size(); i < e; ++i) {
    new_operands[i] = new_operands[i].Unify(min_operands[i]);
    if (new_operands[i] == nullptr) return MappingExpr();
  }

  return MappingUnStripeExpr::get(new_operands, new_factors);
}

mlir::LogicalResult MappingUnStripeExpr::UnificationConstraints(
    MappingExpr other_expr,
    llvm::MutableArrayRef<MappingExpr> constraints) const {
  if (other_expr.isa<MappingNoneExpr>()) return mlir::success();
  MappingUnStripeExpr other_unstripe =
      other_expr.dyn_cast<MappingUnStripeExpr>();
  if (other_unstripe == nullptr) return mlir::failure();

  int num_common;
  llvm::ArrayRef<MappingExpr> min_operands;
  if (factors().size() >= other_unstripe.factors().size()) {
    num_common = other_unstripe.factors().size();
    min_operands = other_unstripe.operands();
  } else {
    num_common = factors().size();
    min_operands = operands();
  }

  // If the last operand is `none`, we can replace it by an arbitrary number of
  // operands.
  if (min_operands.back().isa<MappingNoneExpr>()) --num_common;

  // Ensure that the factors of one are a prefix of the factors of the other.
  if (factors().take_front(num_common) !=
      other_unstripe.factors().take_front(num_common)) {
    return mlir::failure();
  }

  for (int i = 0; i < num_common; ++i) {
    if (mlir::failed(operands()[i].UnificationConstraints(
            other_unstripe.operands()[i], constraints))) {
      return mlir::failure();
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
    auto unstripe = new_operands.back().dyn_cast<MappingUnStripeExpr>();
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
    auto stripe = new_operands.back().dyn_cast<MappingStripeExpr>();
    if (stripe == nullptr) return false;
    int min_num_factors =
        std::min(new_factors.size(), stripe.factors().size());
    // Ensure factors are the same.
    if (llvm::makeArrayRef(new_factors).take_back(min_num_factors) !=
        stripe.factors().take_back(min_num_factors)) {
      return false;
    }


    // Find how many stripes we can stich together.
    int first_stripe = new_operands.size() - 1;
    for(; first_stripe > 0; --first_stripe) {
      auto other_stripe =
          new_operands[first_stripe - 1].dyn_cast<MappingStripeExpr>();
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
  while (collapse_unstripes() || stiche_stripes()) { }

  if (new_factors.size() == 1 && new_factors.back() == 1)
    return new_operands[0];
  return MappingUnStripeExpr::get(new_operands, new_factors);
}

RangeParameters MappingUnStripeExpr::GetRangeParameters(
    mlir::Location loc, llvm::ArrayRef<ValueAccess> domain,
    MappingAttr inverse_mapping, mlir::OpBuilder &builder,
    MapArguments &map_arguments) const {
  RangeParameters inner_parameters = operands()[0].GetRangeParameters(
      loc, domain, inverse_mapping, builder, map_arguments);
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

bool MappingAttr::HasNoneExprs() const {
  return llvm::any_of(getImpl()->mapping(),
                      [](MappingExpr expr) { return expr.HasNoneExprs(); });
}

MappingAttr MappingAttr::MakeSurjective() const {
  int num_dimensions = UseDomainSize();
  llvm::SmallVector<MappingExpr, 4> new_exprs;
  new_exprs.reserve(size());
  for (MappingExpr expr : Dimensions()) {
    MappingExpr new_expr = expr.Map([&](MappingExpr sub_expr) -> MappingExpr {
      if (!sub_expr.isa<MappingNoneExpr>()) return sub_expr;
      return MappingDimExpr::get(num_dimensions++, getContext());
    });
    new_exprs.push_back(new_expr);
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
    exprs.push_back(x.Unify(y));
    if (exprs.back() == nullptr) return nullptr;
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
    assert(dims[i].dependency_mapping().IsSurjective());
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

//===----------------------------------------------------------------------===//
// SairDialect
//===----------------------------------------------------------------------===//

namespace sair {
void SairDialect::registerAttributes() {
  addAttributes<DomainShapeAttr, MappingAttr, NamedMappingAttr, MappingDimExpr,
                MappingNoneExpr, MappingStripeExpr, MappingUnStripeExpr>();
}
}  // namespace sair
