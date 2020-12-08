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

#ifndef SAIR_SAIR_ATTRIBUTES_H_
#define SAIR_SAIR_ATTRIBUTES_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "sair_types.h"

namespace sair {

namespace impl {
// Private implementation for sair::MappingDimExpr
class MappingDimExprStorage;
// Private implementation for sair::MappingStripeExpr
class MappingStripeExprStorage;
// Private implementation for sair::MappingUnStripeExpr
class MappingUnStripeExprStorage;
// Private implementation for sair::MappingAttr
class MappingAttrStorage;
// Private implementation for sair::NamedMappingAttr
class NamedMappingAttrStorage;
// Private implementation for sair::DomainShapeAttr
class DomainShapeAttrStorage;
}  // namespace impl

class MappingExpr;

// MappingAttr describes how a Sair variable is accessed within an
// iteration domain. It indicates how dimensions of the current domain map to
// the dimensions of the variable domain. Dimensions are identified by their
// position in the domain definition.
//
// For example the mapping (0, 2) accesses a three-dimensional variable, with
// the first dimension of the domain iterating on the first dimension of the
// variable, and the third dimension of the domain on the second dimension of
// the variable.
class MappingAttr
    : public mlir::Attribute::AttrBase<MappingAttr, mlir::Attribute,
                                       impl::MappingAttrStorage> {
 public:
  using Base::Base;

  // Constructs an instance of MappingAttr in the given context.
  static MappingAttr get(mlir::MLIRContext *context, int domain_size,
                         llvm::ArrayRef<MappingExpr> mapping);

  // Constructs an instance of MappingAttr and checks it is valid. Returns
  // nullptr in case of failure.
  static MappingAttr getChecked(mlir::MLIRContext *context, int domain_size,
                                llvm::ArrayRef<MappingExpr> mapping);

  // Returns the mapping that accesses `num_dimensions` pointwise without
  // transposing any of them, and has the given size of the use domain. If use
  // domain size is `-1`, it is considered equal to `num_dimensions`.
  static MappingAttr GetIdentity(mlir::MLIRContext *context, int num_dimensions,
                                 int use_domain_size = -1);

  // Returns the mapping that corresponds to the given affine map.
  // Expects the map to be a permutation.
  static MappingAttr FromAffineMap(mlir::AffineMap map);

  // Returns the attribute itself. This is a hook for MLIR.
  MappingAttr getValue() { return *this; }

  // Returns the number of dimensiom in the calling context.
  int UseDomainSize() const;

  // Returns the dimensions along which the variable is accessed.
  llvm::ArrayRef<MappingExpr> Dimensions() const;

  // Returns the dimension at the given position.
  const MappingExpr &Dimension(int position) const {
    return Dimensions()[position];
  }

  // Indicates if the mapping contains no dimensions.
  bool empty() const { return Dimensions().empty(); }

  // Indicates the number of dimensions.
  int size() const { return Dimensions().size(); }

  // Returns the mapping resulting from applying `this` and then `other`
  // to the current set of indices.
  MappingAttr Compose(MappingAttr other) const;

  // Converts the mapping into an affine map from the domain of the
  // operation to the domain of the access value. The mapping must be
  // fully specified.
  mlir::AffineMap AsAffineMap() const;

  // Returns `true` if mapping expression does not contain `none`.
  bool IsFullySpecified() const;

  // Completes the mapping to make it fully specified by allocating new
  // dimensions in the use domain.
  MappingAttr MakeFullySpecified() const;

  // Indicates whether the mapping is an identity, e.g. does not
  // transpose or otherwise modify any dimension.
  bool IsIdentity() const;

  // A bit mask that indicates which dimensions of the domain the mapping
  // depends on.
  llvm::SmallBitVector DependencyMask() const;

  // Indicates if the mapping accesses a single element of the def
  // domain per element of the use domain, when considering only the first
  // `num_dimensions` of the use domain.
  bool IsInjective(int num_dimensions) const;

  // Returns this mapping with a different use domain size. Replaces
  // expressions that are invalid in the new domain by `none`.
  MappingAttr ResizeUseDomain(int new_size) const;

  // Returns this mapping a different number of dimensions. If the new
  // size is greater than the current one, appends `kNoDimensions` at the end of
  // the mapping.
  MappingAttr Resize(int new_size) const;

  // Returns this mapping with dimensions shifted right by `offset`. If
  // `start_from` is set, only starting from the given position will be shifted.
  // For example:
  //   (d0,d1,d2).ShiftRight(2)  =>  (d2,d3,d4)
  //   (d0,d1,d2).ShiftRight(2,1) => (d0,d1,d4)
  MappingAttr ShiftRight(int offset, int start_from = 0) const;

  // Inverse the mapping.
  MappingAttr Inverse() const;

  using iterator = llvm::ArrayRef<MappingExpr>::iterator;
  iterator begin() const { return Dimensions().begin(); }
  iterator end() const { return Dimensions().end(); }
};

// A MappingAttr, with named dimensions in the use domain. Format is
// ```
// [d0:<name[0]>, ..., dn:<name[n]>] -> (<mapping>)
// ```
class NamedMappingAttr
    : public mlir::Attribute::AttrBase<NamedMappingAttr, mlir::Attribute,
                                       impl::NamedMappingAttrStorage> {
 public:
  using Base::Base;

  // Constructs an instance of NamedMappingAttr.
  static NamedMappingAttr get(llvm::ArrayRef<mlir::StringAttr> names,
                              MappingAttr mapping);

  llvm::ArrayRef<mlir::StringAttr> names() const;

  MappingAttr mapping() const;
};

// The shape of an iteration dimension of a Sair domain.
class DomainShapeDim {
 public:
  DomainShapeDim(RangeType type, MappingAttr dependency_mapping);

  // Expected type for the dimension.
  const RangeType &type() const { return type_; }

  // Mapping for the dimension, with regard to previous dimensions in the
  // domain.
  MappingAttr dependency_mapping() const { return dependency_mapping_; }

  // Composes the mapping with the shape dimension dependency mapping.
  DomainShapeDim Apply(MappingAttr mapping) const;

  // A mask that indicates which dimensions this dimension depends on. A
  // dimension can only depend on dimensions that occure before this one.
  llvm::SmallBitVector DependencyMask() const {
    return dependency_mapping_.DependencyMask();
  }

 private:
  RangeType type_;
  MappingAttr dependency_mapping_;
};

// Tests the equality of two domain shape dimensions.
bool operator==(const DomainShapeDim &lhs, const DomainShapeDim &rhs);

// Compute a hash for a domain shape dimension.
unsigned hash_value(const DomainShapeDim &shape_dim);

// DomainShapeAttr describes the shape of a Sair iteration domain. A Sair
// iteration domain is a product of iteration dimensions, with some dimensions
// depending on others. The shape of a domain specifies the type of each
// dimension and how they depend on each other.
//
// The general syntax for DomainShapeAttr is the following.
// ```
// <domain-shape> ::= () | <dim-shape> | <dim-shape> 'x' <domain-shape>
//    <dim-shape> ::= <name> ':' 'range' ( '(' <int>,* ')' )?
// ```
// For example the following shape describes a two-dimensional domain where the
// second dimension depends on the first.
// ```mlir
//   d0:range x d1:range(d0)
// ```
class DomainShapeAttr
    : public mlir::Attribute::AttrBase<DomainShapeAttr, mlir::Attribute,
                                       impl::DomainShapeAttrStorage> {
 public:
  using Base::Base;

  // Constructs an instance of DomainShapeAttr in the given context. Defaults to
  // the zero-dimensional shape when no dimenion shapes are given.
  static DomainShapeAttr get(mlir::MLIRContext *context,
                             llvm::ArrayRef<DomainShapeDim> dims = {});
  // Returns the shape of an hyper-rectangular domain with 'rank' iteration
  // dimensions.  An hyper-rectangular domain is a domain composed of range
  // iteration dimensions with no dependencies between them.
  static DomainShapeAttr HyperRectangular(mlir::MLIRContext *context, int rank);

  // Returns a prefix of this shape of the given size.
  DomainShapeAttr Prefix(int size);

  // Returns the attribute itself. This is a hook for TableGen.
  DomainShapeAttr getValue() { return *this; }

  // Returns the number of dimensions in the domain.
  int NumDimensions() const;

  // Indicates if the domain contains exactly one point.
  bool Is0d() const;

  // Returns the shape of the iteration dimension that compose the domain.
  llvm::ArrayRef<DomainShapeDim> Dimensions() const;

  // Returns the dimension at the given index.
  const DomainShapeDim &Dimension(int index) const {
    return Dimensions()[index];
  }

  // Indicates if all the dimensions of the domain are independent.
  bool IsHyperRectangular() const;

  // Indicates if this DomainShapeAttr is a prefix of another DomainShapeAttr.
  bool IsPrefixOf(DomainShapeAttr other);

  // Returns the expected shape of the sair value the mapping refers to.
  DomainShapeAttr AccessedShape(MappingAttr mapping) const;

  // Returns a new domain that is a product of this and `other` domains, i.e.,
  // is a concatenation of the dimensions of both.
  DomainShapeAttr Product(DomainShapeAttr other) const;

  // Returns a new domain that is a product of this and `other` domains, with
  // dimensions of the `other` domain starting at `pos` in the result. In other
  // words, the result is a concatenation of the first `pos` dimensions of this
  // domain, the dimensions of `other`, and the remaining dimensions of this
  // domain.
  DomainShapeAttr ProductAt(int pos, DomainShapeAttr other) const;
};

#include "sair_attr_interfaces.h.inc"

// Mapping expression that maps to a dimension of the domain.
class MappingDimExpr
    : public mlir::Attribute::AttrBase<MappingDimExpr, mlir::Attribute,
                                       impl::MappingDimExprStorage,
                                       MappingExpr::Trait> {
 public:
  using Base::Base;

  // Creates an MappingDimExpr representing the given dimension.
  static MappingDimExpr get(int dimension, mlir::MLIRContext *context);

  // Returns the dimensions represented by the expression
  int dimension() const;

  void SetDependenciesInMask(llvm::SmallBitVector &mask) const {
    mask.set(dimension());
  }

  bool IsFullySpecified() const { return true; }

  MappingExpr MakeFullySpecified(int &num_dimensions) const { return *this; }

  MappingExpr SubstituteDims(mlir::ArrayRef<MappingExpr> exprs) const;

  DomainShapeDim AccessedShape(llvm::ArrayRef<DomainShapeDim> accessing_shape,
                               MappingAttr inverted_mapping) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const;

  MappingExpr Unify(MappingExpr other_expr) const;

  mlir::LogicalResult UnificationConstraints(
      MappingExpr other_expr,
      llvm::MutableArrayRef<MappingExpr> constraints) const;

  int MinDomainSize() const { return dimension() + 1; }

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const {
    return inverse[dimension()];
  }

  mlir::AffineExpr AsAffineExpr() const;
};

// Mapping expression that maps to no dimensions. This used when
// computing dependencies and is invalid in types and operations.
class MappingNoneExpr
    : public mlir::Attribute::AttrBase<MappingNoneExpr, mlir::Attribute,
                                       mlir::AttributeStorage,
                                       MappingExpr::Trait> {
 public:
  using Base::Base;

  static constexpr llvm::StringRef kAttrName = "none";

  static MappingNoneExpr get(mlir::MLIRContext *context);

  bool IsFullySpecified() const { return false; }

  MappingExpr MakeFullySpecified(int &num_dimensions) const;

  void SetDependenciesInMask(llvm::SmallBitVector &mask) const {}

  MappingExpr SubstituteDims(mlir::ArrayRef<MappingExpr> exprs) const {
    return *this;
  }

  DomainShapeDim AccessedShape(llvm::ArrayRef<DomainShapeDim> accessing_shape,
                               MappingAttr inversed_mapping) const {
    llvm_unreachable(
        "'none' mapping expression cannot be used to access values");
  }

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const {
    return mlir::success();
  }

  MappingExpr Unify(MappingExpr other_expr) const { return other_expr; }

  mlir::LogicalResult UnificationConstraints(
      MappingExpr other_expr,
      llvm::MutableArrayRef<MappingExpr> constraints) const {
    return mlir::success();
  }

  int MinDomainSize() const { return 0; }

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const {
    llvm_unreachable(
        "cannot call `FindInInverse` on partially-specified expressions");
  }

  mlir::AffineExpr AsAffineExpr() const {
    llvm_unreachable(
        "cannot call `AsAffineExpr` on partially specified expressions");
  }
};

// Applies stripe-mining to an expression. Iterates on its operand with step
// `step()`, on a strip of size `size()`. If it iterates of the full expression,
// `size()` is none.
class MappingStripeExpr
    : public mlir::Attribute::AttrBase<MappingStripeExpr, mlir::Attribute,
                                       impl::MappingStripeExprStorage,
                                       MappingExpr::Trait> {
 public:
  using Base::Base;

  static constexpr llvm::StringRef kAttrName = "stripe";

  static MappingStripeExpr get(MappingExpr operand, int step,
                               llvm::Optional<int> size);

  // The striped expression.
  MappingExpr operand() const;

  // Iteration step. This is 1 for point expressions.
  int step() const;

  // The range of the expression. This is the stripe factor for point
  // expressions and none for outer stripe expressions.
  llvm::Optional<int> size() const;

  bool IsFullySpecified() const { return operand().IsFullySpecified(); }

  MappingExpr MakeFullySpecified(int &num_dimensions) const;

  int MinDomainSize() const { return operand().MinDomainSize(); }

  void SetDependenciesInMask(llvm::SmallBitVector &mask) const {
    operand().SetDependenciesInMask(mask);
  }

  MappingExpr SubstituteDims(mlir::ArrayRef<MappingExpr> exprs) const;

  DomainShapeDim AccessedShape(llvm::ArrayRef<DomainShapeDim> accessing_shape,
                               MappingAttr inverted_mapping) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const;

  MappingExpr Unify(MappingExpr other_expr) const;

  mlir::LogicalResult UnificationConstraints(
      MappingExpr other_expr,
      llvm::MutableArrayRef<MappingExpr> constraints) const;

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const;

  mlir::AffineExpr AsAffineExpr() const;
};

// Stiches together stripe expressions to iterate on a full dimension. Specifies
// the step of stripe expressions, except for the innermost which always has
// step 1.
class MappingUnStripeExpr
    : public mlir::Attribute::AttrBase<MappingUnStripeExpr, mlir::Attribute,
                                       impl::MappingUnStripeExprStorage,
                                       MappingExpr::Trait> {
 public:
  using Base::Base;

  static constexpr llvm::StringRef kAttrName = "unstripe";

  // Constructs an unstripe mapping expression. `stripes` must not be empty.
  static MappingUnStripeExpr get(llvm::ArrayRef<MappingExpr> stripes,
                                 llvm::ArrayRef<int> factors);

  // The stripe expressions that are combined to obtain the unstriped
  // expression.
  llvm::ArrayRef<MappingExpr> operands() const;

  // Stripe expression sizes.
  llvm::ArrayRef<int> factors() const;

  bool IsFullySpecified() const;

  MappingExpr MakeFullySpecified(int &num_dimensions) const;

  void SetDependenciesInMask(llvm::SmallBitVector &mask) const;

  int MinDomainSize() const;

  MappingExpr SubstituteDims(mlir::ArrayRef<MappingExpr> exprs) const;

  DomainShapeDim AccessedShape(llvm::ArrayRef<DomainShapeDim> accessing_shape,
                               MappingAttr inverted_mapping) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const;

  MappingExpr Unify(MappingExpr other_expr) const;

  mlir::LogicalResult UnificationConstraints(
      MappingExpr other_expr,
      llvm::MutableArrayRef<MappingExpr> constraints) const;

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const;

  mlir::AffineExpr AsAffineExpr() const;
};

}  // namespace sair

// Include the definition of struct attributes generated by MLIR. The using
// namespace is required by the generated code.
using namespace mlir;  // NOLINT
#include "sair_structs.h.inc"

#endif  // SAIR_SAIR_ATTRIBUTES_H_
