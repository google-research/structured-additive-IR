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
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
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

  // HasNoneExprs indicates if any sub-expression is `none`.
  bool HasNoneExprs() const;
  bool IsSurjective() const { return !HasNoneExprs(); }

  // Replaces `none` expressions by new dimensions to make the mapping
  // surjective.
  MappingAttr MakeSurjective() const;

  // HasUnknownExprs indicates if any sub-expression is `?`.
  bool HasUnknownExprs() const;
  bool IsFullySpecified() const { return !HasUnknownExprs(); }

  // Replaces `?` expressions by `none` expressions.
  MappingAttr MakeFullySpecified() const;

  // Indicates whether the mapping is an identity, e.g. does not
  // transpose or otherwise modify any dimension.
  bool IsIdentity() const;

  // A bit mask that indicates which dimensions of the domain the mapping
  // depends on.
  llvm::SmallBitVector DependencyMask() const;

  // Minimal use domain size the mapping could have while remaining valid.
  int MinDomainSize() const;

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
  // Use domain size is also increased by `offset`.
  // For example:
  //   (d0,d1,d2).ShiftRight(2)  =>  (d2,d3,d4)
  //   (d0,d1,d2).ShiftRight(2,1) => (d0,d1,d4)
  MappingAttr ShiftRight(int offset, int start_from = 0) const;

  // Adds mapping expressions in front or at the end of the mapping.
  MappingAttr AddPrefix(llvm::ArrayRef<MappingExpr> exprs) const;
  MappingAttr AddSuffix(llvm::ArrayRef<MappingExpr> exprs) const;

  // Drops expressions in front of the mapping.
  MappingAttr DropFront(int num_drop) const;

  // Take a slice of the expressions of the mapping.
  MappingAttr Slice(int begin, int new_size) const;

  // Inverse the mapping.
  MappingAttr Inverse() const;

  // Canonicalize dimension expressions.
  MappingAttr Canonicalize() const;

  // Unifies this mapping with `other` by substituting `none` and `?`
  // expressions. In case `none` conflicts with a `?`, keeps the `none`
  // expression. Returns `nullptr` if unification fails in one of the
  // dimensions. Both expressions must have the same number of dimensions and
  // domain size.
  MappingAttr Unify(MappingAttr other) const;

  // Unifies this mapping with `other` by substituting ``?` expressions. Returns
  // `nullptr` if unification fails in one of the dimensions. Both expressions
  // must have the same number of dimensions and domain size.
  MappingAttr UnifyUnknownExprs(MappingAttr other) const;

  using iterator = llvm::ArrayRef<MappingExpr>::iterator;
  iterator begin() const { return Dimensions().begin(); }
  iterator end() const { return Dimensions().end(); }
};

// Unifies two mapping expressions by substituting `none` and `?` expressions.
// In the case where `none` conflicts with `?`, keeps `none`.  Returns nullptr
// in case of failure.
MappingExpr Unify(MappingExpr lhs, MappingExpr rhs);

// Fills `constraints` with expression such that
// `lhs.SubstituteDims(constraints).Unify(rhs)` succeeds. Expects `contraints`
// to be initially filled with `none`. Leaves `constraints[i]` untouched if
// dimension i generates no constraints. Returns a failure if expressions cannot
// be unified.
mlir::LogicalResult UnificationConstraints(
    MappingAttr lhs, MappingAttr rhs,
    llvm::MutableArrayRef<MappingExpr> constraints);

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
  static NamedMappingAttr get(llvm::ArrayRef<mlir::StringAttr> names,
                              llvm::ArrayRef<MappingExpr> exprs,
                              mlir::MLIRContext *context);

  // Constructs an instance of NamedMappingAttr with an identity mapping.
  static NamedMappingAttr GetIdentity(mlir::MLIRContext *context,
                                      llvm::ArrayRef<mlir::StringAttr> names);

  llvm::ArrayRef<mlir::StringAttr> names() const;

  MappingAttr mapping() const;

  // Drop dimensions from the domain that are unused.
  NamedMappingAttr DropUnusedDims() const;

  // Returns the mapping resulting from applying `this` and then `other`.
  NamedMappingAttr Compose(MappingAttr other) const;
};

// The shape of an iteration dimension of a Sair domain.
class DomainShapeDim {
 public:
  DomainShapeDim(DimensionType type, MappingAttr dependency_mapping);

  // Expected type for the dimension.
  const DimensionType &type() const { return type_; }

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
  DimensionType type_;
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

  // Returns the expected shape of the sair value the mapping refers to.
  DomainShapeAttr AccessedShape(MappingAttr mapping) const;
};

// Location in a Sair attribute.
//
// The location is defined by an MLIR location, an attribute kind and an
// attribute name. The attribute name may be null.
class AttrLocation {
 public:
  AttrLocation(mlir::Location loc, llvm::StringRef kind,
               mlir::StringAttr name = nullptr)
      : loc_(loc), kind_(kind), name_(name) {}

  // MLIR location.
  mlir::Location location() const { return loc_; }

  // Attribute name.
  mlir::StringAttr name() const { return name_; }

  // Context in which the attribute is defined.
  mlir::MLIRContext *context() const { return name_.getContext(); }

  // Emits an error at the attribute location. Error has format
  // `error in <kind> <name>: <msg>`
  mlir::InFlightDiagnostic EmitError() const;

 private:
  friend mlir::Diagnostic &operator<<(mlir::Diagnostic &diag,
                                      const AttrLocation &loc);

  mlir::Location loc_;
  llvm::StringRef kind_;
  mlir::StringAttr name_;
};

// Appends the attribute kind and name to the error message.
mlir::Diagnostic &operator<<(mlir::Diagnostic &diag, const AttrLocation &loc);

// Verifies that the mapping can be applied to a domain with the given shape.
// Does not emits errors.
mlir::LogicalResult VerifyMappingShape(const AttrLocation &loc,
                                       MappingAttr mapping,
                                       DomainShapeAttr shape);

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

  MappingExpr Map(llvm::function_ref<MappingExpr(MappingExpr)> function) const;

  void Walk(llvm::function_ref<void(MappingExpr)> function) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const;

  MappingExpr Unify(MappingExpr other_expr,
                    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)>
                        on_mismatch) const;

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const {
    return inverse[dimension()];
  }

  mlir::AffineExpr AsAffineExpr() const;

  MappingExpr Canonicalize() const { return *this; }
};

// Mapping expression that maps to no dimensions.
class MappingNoneExpr
    : public mlir::Attribute::AttrBase<MappingNoneExpr, mlir::Attribute,
                                       mlir::AttributeStorage,
                                       MappingExpr::Trait> {
 public:
  using Base::Base;

  static constexpr llvm::StringRef kAttrName = "none";

  static MappingNoneExpr get(mlir::MLIRContext *context);

  MappingExpr Map(llvm::function_ref<MappingExpr(MappingExpr)> function) const;

  void Walk(llvm::function_ref<void(MappingExpr)> function) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const {
    return mlir::success();
  }

  MappingExpr Unify(MappingExpr other_expr,
                    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)>
                        on_mismatch) const;

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const {
    return *this;
  }

  mlir::AffineExpr AsAffineExpr() const {
    llvm_unreachable("cannot call `AsAffineExpr` on none expressions");
  }

  MappingExpr Canonicalize() const { return *this; }
};

// To be specified mapping expression.
class MappingUnknownExpr
    : public mlir::Attribute::AttrBase<MappingUnknownExpr, mlir::Attribute,
                                       mlir::AttributeStorage,
                                       MappingExpr::Trait> {
 public:
  using Base::Base;

  static constexpr llvm::StringRef kAttrName = "?";

  static MappingUnknownExpr get(mlir::MLIRContext *context);

  MappingExpr Map(llvm::function_ref<MappingExpr(MappingExpr)> function) const;

  void Walk(llvm::function_ref<void(MappingExpr)> function) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const {
    return mlir::success();
  }

  MappingExpr Unify(MappingExpr other_expr,
                    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)>
                        on_mismatch) const;

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const {
    return *this;
  }

  mlir::AffineExpr AsAffineExpr() const {
    llvm_unreachable("cannot call `AsAffineExpr` on unknown expressions");
  }

  MappingExpr Canonicalize() const { return *this; }
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

  static MappingStripeExpr get(MappingExpr operand,
                               llvm::ArrayRef<int> factors);

  // The striped expression.
  MappingExpr operand() const;

  // Successive stripe factors to apply to `operand` to obtain this dimension.
  // Cannot be empty.
  llvm::ArrayRef<int> factors() const;

  MappingExpr Map(llvm::function_ref<MappingExpr(MappingExpr)> function) const;

  void Walk(llvm::function_ref<void(MappingExpr)> function) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const;

  MappingExpr Unify(MappingExpr other_expr,
                    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)>
                        on_mismatch) const;

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const;

  mlir::AffineExpr AsAffineExpr() const;

  MappingExpr Canonicalize() const;
};

// Stiches together stripe expressions to iterate on a full dimension. Specifies
// the step of stripe expression. The last step must be 1.
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

  MappingExpr Map(llvm::function_ref<MappingExpr(MappingExpr)> function) const;

  void Walk(llvm::function_ref<void(MappingExpr)> function) const;

  mlir::LogicalResult SetInverse(
      MappingExpr context_inverse,
      llvm::MutableArrayRef<MappingExpr> inverses) const;

  MappingExpr Unify(MappingExpr other_expr,
                    llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)>
                        on_mismatch) const;

  MappingExpr FindInInverse(llvm::ArrayRef<MappingExpr> inverse) const;

  mlir::AffineExpr AsAffineExpr() const;

  MappingExpr Canonicalize() const;
};

}  // namespace sair

// Include the definition of struct attributes generated by MLIR. The using
// namespace is required by the generated code.
using namespace mlir;  // NOLINT
#include "sair_structs.h.inc"

namespace sair {

// Below are helper functions to manipulate DecisionsAttr. Each helper takes a
// function and returns a function that applies the first function to a field of
// a DecisionsAttr. Helpers return functions rather than directly applying the
// transformation so that it is easier to combine transformations.

// Takes a function that updates a loop nest and returns a function that updates
// the loop nest field of a DecisionsAttr.
std::function<DecisionsAttr(DecisionsAttr)> MapLoopNest(
    std::function<mlir::ArrayAttr(mlir::ArrayAttr)> loop_nest_fn);

// Takes a function that updates a list of storages and returns a function that
// updates the storage field of a DecisionsAttr.
std::function<DecisionsAttr(DecisionsAttr)> MapStorage(
    std::function<mlir::ArrayAttr(mlir::ArrayAttr)> storage_fn);

}  // namespace sair

#endif  // SAIR_SAIR_ATTRIBUTES_H_
