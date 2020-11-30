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

// Private implementation/storage class for sair::AccessPatternDimExpr.
// Instances of this class are allocate by MLIR type system in a dedicated
// arena. Not intended for direct use.
class impl::AccessPatternDimExprStorage : public mlir::AttributeStorage {
 public:
  // Key type uniquely identifies DomainShapeAttrStorage for MLIR attribute
  // unique-ing. This specific name is required by mlir::AttributeUniquer.
  using KeyTy = int;

  // Creates a AccessPatternDimExprStorage using the provided allocator. Hook
  // for MLIR attribute system.
  static AccessPatternDimExprStorage *construct(
      mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<AccessPatternDimExprStorage>())
        AccessPatternDimExprStorage(key);
  }

  // Compares the DomainShapeAttrStorage identification key with this object.
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

DomainShapeDim AccessPatternDimExpr::AccessedShape(
    llvm::ArrayRef<DomainShapeDim> accessing_shape,
    AccessPatternAttr inversed_pattern) const {
  return accessing_shape[dimension()].Apply(inversed_pattern);
}

AccessPatternNoneExpr AccessPatternNoneExpr::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

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

AccessPatternAttr AccessPatternAttr::get(
    mlir::MLIRContext *context, int use_domain_size,
    llvm::ArrayRef<AccessPatternExpr> pattern) {
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

mlir::AffineMap AccessPatternAttr::AsAffineMap() const {
  llvm::SmallVector<mlir::AffineExpr, 4> affine_exprs;
  affine_exprs.reserve(Dimensions().size());
  for (AccessPatternExpr expr : *this) {
    // TODO(ulysse): support more cases or remove this method
    int dimension = expr.cast<AccessPatternDimExpr>().dimension();
    affine_exprs.push_back(mlir::getAffineDimExpr(dimension, getContext()));
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

  AccessPatternExpr none_expr = AccessPatternNoneExpr::get(getContext());
  llvm::SmallVector<AccessPatternExpr, 4> exprs;
  exprs.reserve(size());
  for (AccessPatternExpr expr : Dimensions()) {
    llvm::SmallBitVector dimensions(old_size);
    expr.SetDependenciesInMask(dimensions);
    if (dimensions.find_last() >= new_size) {
      exprs.push_back(none_expr);
    } else {
      exprs.push_back(expr);
    }
  }

  return AccessPatternAttr::get(getContext(), new_size, exprs);
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
  llvm::SmallVector<AccessPatternExpr, 4> inversed_exprs(
      UseDomainSize(), AccessPatternNoneExpr::get(context));
  for (int i = 0, e = size(); i < e; ++i) {
    AccessPatternExpr dim_expr = AccessPatternDimExpr::get(i, context);
    Dimension(i).SetInverse(dim_expr, inversed_exprs);
  }
  return AccessPatternAttr::get(context, size(), inversed_exprs);
}

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
  AccessPatternAttr inversed_pattern = access_pattern.Inverse();
  for (int i = 0, e = access_pattern.size(); i < e; ++i) {
    DomainShapeDim shape_dim = access_pattern.Dimension(i).AccessedShape(
        Dimensions(), inversed_pattern.ResizeUseDomain(i));
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
