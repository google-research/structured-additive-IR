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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "sair_types.h"

namespace sair {

namespace impl {
// Private implementation for sair::AccessPatternDimExpr
class AccessPatternDimExprStorage;
// Private implementation for sair::AccessPatternAttr
class AccessPatternAttrStorage;
// Private implementation for sair::DomainShapeAttr
class DomainShapeAttrStorage;
// Private implementation for sair::IteratorAttr
class IteratorAttrStorage;
}  // namespace impl

class AccessPatternExpr;

// AccessPatternAttr describes how a Sair variable is accessed within an
// iteration domain. It indicates how dimensions of the current domain map to
// the dimensions of the variable domain. Dimensions are identified by their
// position in the domain definition.
//
// For example the pattern (0, 2) accesses a three-dimensional variable, with
// the first dimension of the domain iterating on the first dimension of the
// variable, and the third dimension of the domain on the second dimension of
// the variable.
class AccessPatternAttr
    : public mlir::Attribute::AttrBase<AccessPatternAttr, mlir::Attribute,
                                       impl::AccessPatternAttrStorage> {
 public:
  using Base::Base;

  // Constructs an instance of AccessPatternAttr in the given context.
  static AccessPatternAttr get(mlir::MLIRContext *context, int domain_size,
                               llvm::ArrayRef<AccessPatternExpr> pattern);
  // Returns the access pattern that accesses `num_dimensions` pointwise without
  // transposing any of them, and has the given size of the use domain. If use
  // domain size is `-1`, it is considered equal to `num_dimensions`.
  static AccessPatternAttr GetIdentity(mlir::MLIRContext *context,
                                       int num_dimensions,
                                       int use_domain_size = -1);

  // Returns the access pattern that corresponds to the given affine map.
  // Expects the map to be a permutation.
  static AccessPatternAttr FromAffineMap(mlir::AffineMap map);

  // Returns the attribute itself. This is a hook for MLIR.
  AccessPatternAttr getValue() { return *this; }

  // Returns the number of dimensiom in the calling context.
  int UseDomainSize() const;

  // Returns the dimensions along which the variable is accessed.
  llvm::ArrayRef<AccessPatternExpr> Dimensions() const;

  // Returns the dimension at the given position.
  const AccessPatternExpr &Dimension(int position) const {
    return Dimensions()[position];
  }

  // Indicates if the pattern contains no dimensions.
  bool empty() const { return Dimensions().empty(); }

  // Indicates the number of dimensions.
  int size() const { return Dimensions().size(); }

  // Returns the access pattern resulting from applying `this` and then `other`
  // to the current set of indices.
  AccessPatternAttr Compose(AccessPatternAttr other) const;

  // Converts the access pattern into an affine map from the domain of the
  // operation to the domain of the access value. The access pattern must be
  // fully specified.
  mlir::AffineMap AsAffineMap() const;

  // Inverse the access pattern and returns it as an affine map. Returns the
  // null affine map if the pattern is not invertible.
  mlir::AffineMap InverseAffineMap() const;

  // Returns `true` if access pattern expression does not contain `none`.
  bool IsFullySpecified() const;

  // Indicates whether the access pattern is an identity, e.g. does not
  // transpose or otherwise modify any dimension.
  bool IsIdentity() const;

  // A bit mask that indicates which dimensions of the domain the access pattern
  // depends on.
  llvm::SmallBitVector DependencyMask() const;

  // Indicates if the access pattern accesses a single element of the def
  // domain per element of the use domain, when considering only the first
  // `num_dimensions` of the use domain.
  bool IsInjective(int num_dimensions) const;

  // Returns this access pattern with a different use domain size. Replaces
  // expressions that are invalid in the new domain by `none`.
  AccessPatternAttr ResizeUseDomain(int new_size) const;

  // Returns this access pattern a different number of dimensions. If the new
  // size is greater than the current one, appends `kNoDimensions` at the end of
  // the access pattern.
  AccessPatternAttr Resize(int new_size) const;

  // Returns this access pattern with dimensions shifted right by `offset`. If
  // `start_from` is set, only starting from the given position will be shifted.
  // For example:
  //   (d0,d1,d2).ShiftRight(2)  =>  (d2,d3,d4)
  //   (d0,d1,d2).ShiftRight(2,1) => (d0,d1,d4)
  AccessPatternAttr ShiftRight(int offset, int start_from = 0) const;

  // Inverse the access pattern.
  AccessPatternAttr Inverse() const;

  using iterator = llvm::ArrayRef<AccessPatternExpr>::iterator;
  iterator begin() const { return Dimensions().begin(); }
  iterator end() const { return Dimensions().end(); }
};

// The shape of an iteration dimension of a Sair domain.
class DomainShapeDim {
 public:
  DomainShapeDim(RangeType type, AccessPatternAttr dependency_pattern);

  // Expected type for the dimension.
  const RangeType &type() const { return type_; }

  // Access pattern for the dimension, with regard to previous dimensions in the
  // domain.
  AccessPatternAttr dependency_pattern() const { return dependency_pattern_; }

  // Composes the access pattern with the shape dimension dependency pattern.
  DomainShapeDim Apply(AccessPatternAttr access_pattern) const;

  // A mask that indicates which dimensions this dimension depends on. A
  // dimension can only depend on dimensions that occure before this one.
  llvm::SmallBitVector DependencyMask() const {
    return dependency_pattern_.DependencyMask();
  }

 private:
  RangeType type_;
  AccessPatternAttr dependency_pattern_;
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

  // Returns the expected shape of the sair value the access pattern refers to.
  DomainShapeAttr AccessedShape(AccessPatternAttr access_pattern) const;

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

// Base trait for access pattern expressions.
struct AccessPatternExprInterfaceTraits {
  struct Concept {
    virtual ~Concept() {}

    // Returns `true` if the expression do not contain `none`.
    virtual bool IsFullySpecified(mlir::Attribute attr) const = 0;

    // Sets the dimensions referenced by the expression in `mask`.
    virtual void SetDependenciesInMask(mlir::Attribute attr,
                                       llvm::SmallBitVector &mask) const = 0;

    // Substitutes dimension expressions by the given expressions.
    virtual mlir::Attribute SubstituteDims(
        mlir::Attribute attr,
        mlir::ArrayRef<AccessPatternExpr> exprs) const = 0;

    // Returns the shape of the dimension accessed by the given expression,
    // given the shape of the current domain, the inversed access pattern and
    // the number of dimensions the final shape should have.
    virtual DomainShapeDim AccessedShape(
        mlir::Attribute attr, llvm::ArrayRef<DomainShapeDim> accessing_shape,
        AccessPatternAttr inversed_pattern) const = 0;

    // Sets `inverses[i]` to the inverse of the current expression (and its
    // context) with regard to `di`. `context_inverse` is the inverse of the
    // context of the current expression.
    virtual void SetInverse(
        mlir::Attribute attr, AccessPatternExpr &context_inverse,
        llvm::MutableArrayRef<AccessPatternExpr> inverses) const = 0;
  };

  template <typename ConcreteAttr>
  struct Model : public Concept {
    bool IsFullySpecified(mlir::Attribute attr) const final {
      return attr.cast<ConcreteAttr>().IsFullySpecified();
    }

    void SetDependenciesInMask(mlir::Attribute attr,
                               llvm::SmallBitVector &mask) const final {
      attr.cast<ConcreteAttr>().SetDependenciesInMask(mask);
    }

    mlir::Attribute SubstituteDims(
        mlir::Attribute attr,
        mlir::ArrayRef<AccessPatternExpr> exprs) const final {
      return attr.cast<ConcreteAttr>().SubstituteDims(exprs);
    }

    DomainShapeDim AccessedShape(
        mlir::Attribute attr, llvm::ArrayRef<DomainShapeDim> accessing_shape,
        AccessPatternAttr inversed_pattern) const final {
      return attr.cast<ConcreteAttr>().AccessedShape(accessing_shape,
                                                     inversed_pattern);
    }

    void SetInverse(
        mlir::Attribute attr, AccessPatternExpr &context_inverse,
        llvm::MutableArrayRef<AccessPatternExpr> inverses) const final {
      return attr.cast<ConcreteAttr>().SetInverse(context_inverse, inverses);
    }
  };
};

// Base interface for access pattern expressions.
class AccessPatternExpr
    : public mlir::AttributeInterface<AccessPatternExpr,
                                      AccessPatternExprInterfaceTraits> {
 public:
  using mlir::AttributeInterface<
      AccessPatternExpr, AccessPatternExprInterfaceTraits>::AttributeInterface;

  // Returns `true` if the expression do not contain `none`.
  bool IsFullySpecified() const { return getImpl()->IsFullySpecified(*this); }

  // Sets the dimensions referenced by the expression in `mask`.
  void SetDependenciesInMask(llvm::SmallBitVector &mask) const {
    return getImpl()->SetDependenciesInMask(*this, mask);
  }

  // Substitutes dimension expressions by the given expressions.
  AccessPatternExpr SubstituteDims(
      mlir::ArrayRef<AccessPatternExpr> exprs) const {
    return getImpl()->SubstituteDims(*this, exprs).cast<AccessPatternExpr>();
  }

  // Returns the shape of the dimension accessed by the given expression, given
  // the shape of the current domain, the inversed access pattern and the number
  // of dimensions the final shape should have.
  DomainShapeDim AccessedShape(llvm::ArrayRef<DomainShapeDim> accessing_shape,
                               AccessPatternAttr inversed_pattern) const {
    return getImpl()->AccessedShape(*this, accessing_shape, inversed_pattern);
  }

  // Sets `inverses[i]` to the inverse of the current expression (and its
  // context) with regard to `di`. `context_inverse` is the inverse of the
  // context of the current expression.
  void SetInverse(AccessPatternExpr context_inverse,
                  llvm::MutableArrayRef<AccessPatternExpr> inverses) const {
    return getImpl()->SetInverse(*this, context_inverse, inverses);
  }
};

// Access pattern expression that maps to a dimension of the domain.
class AccessPatternDimExpr
    : public mlir::Attribute::AttrBase<AccessPatternDimExpr, mlir::Attribute,
                                       impl::AccessPatternDimExprStorage,
                                       AccessPatternExpr::Trait> {
 public:
  using Base::Base;

  // Creates an AccessPatternDimExpr representing the given dimension.
  static AccessPatternDimExpr get(int dimension, mlir::MLIRContext *context);

  // Returns the dimensions represented by the expression
  int dimension() const;

  void SetDependenciesInMask(llvm::SmallBitVector &mask) const {
    mask.set(dimension());
  }

  bool IsFullySpecified() const { return true; }

  mlir::Attribute SubstituteDims(
      mlir::ArrayRef<AccessPatternExpr> exprs) const {
    return exprs[dimension()];
  }

  DomainShapeDim AccessedShape(llvm::ArrayRef<DomainShapeDim> accessing_shape,
                               AccessPatternAttr inversed_pattern) const;

  void SetInverse(AccessPatternExpr context_inverse,
                  llvm::MutableArrayRef<AccessPatternExpr> inverses) const {
    inverses[dimension()] = context_inverse;
  }
};

// Access pattern expression that maps to no dimensions. This used when
// computing dependencies and is invalid in types and operations.
class AccessPatternNoneExpr
    : public mlir::Attribute::AttrBase<AccessPatternNoneExpr, mlir::Attribute,
                                       mlir::AttributeStorage,
                                       AccessPatternExpr::Trait> {
 public:
  using Base::Base;

  static AccessPatternNoneExpr get(mlir::MLIRContext *context);

  bool IsFullySpecified() const { return false; }

  void SetDependenciesInMask(llvm::SmallBitVector &mask) const {}

  mlir::Attribute SubstituteDims(
      mlir::ArrayRef<AccessPatternExpr> exprs) const {
    return *this;
  }

  DomainShapeDim AccessedShape(llvm::ArrayRef<DomainShapeDim> accessing_shape,
                               AccessPatternAttr inversed_pattern) const {
    llvm_unreachable(
        "'none' access pattern expression cannot be used to access values");
  }

  void SetInverse(AccessPatternExpr context_inverse,
                  llvm::ArrayRef<AccessPatternExpr> inverses) const {}
};

// An iterator on a Sair iteration dimension.
// TODO(ulysse): replace by AccessPatternExprAttr
class IteratorAttr
    : public mlir::Attribute::AttrBase<IteratorAttr, mlir::Attribute,
                                       impl::IteratorAttrStorage> {
 public:
  using Base::Base;

  // Returns the `IteratorAttr` that iterates along `dimension` with the given
  // step. This is a hook for MLIR.
  static IteratorAttr get(mlir::MLIRContext *context, int dimension,
                          int step = 1);
  // Returns the `IteratorAttr` that rematerializes the computation in the
  // dimension of another operation. This is a hook for MLIR.
  static IteratorAttr get(mlir::MLIRContext *context);

  // Returns the attribute itself. This is a hook for MLIR.
  IteratorAttr getValue() { return *this; }

  // Indicates that `IteratorAttr` rematerializes the computation in the
  // dimension of another operation.
  bool Rematerialize();

  // Dimension to iterate on. Only defined if `Rematerialize` returns false.
  int Dimension();
  // Size of the chunks to iterate on. Only defined if `Dimensions` is.
  int Step();
};

}  // namespace sair

// Include the definition of struct attributes generated by MLIR. The using
// namespace is required by the generated code.
using namespace mlir;  // NOLINT
#include "sair_structs.h.inc"

#endif  // SAIR_SAIR_ATTRIBUTES_H_
