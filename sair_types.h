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

#ifndef SAIR_SAIR_TYPES_H_
#define SAIR_SAIR_TYPES_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

namespace sair {

namespace impl {
// Private implementation for sair::RangeType.
class ShapedTypeStorage;
// Private implementation for sair::ValueType.
class ValueTypeStorage;
}  // end namespace impl

class DomainShapeAttr;

// Base type for Sair objects (values or iteration dimensions) that are defined
// for each point in an iteration domain. This type exposes the shape of the
// domain. Unknown shapes are not supported.
class ShapedType : public mlir::Type {
 public:
  // This a hook for the MLIR type system.
  using Type::Type;

  // Returns the shape of the type.
  DomainShapeAttr Shape() const;
};

// Range type is used for values that define a dimension in a Sair iteration
// domain. A range type may dependent on some number of other iteration
// dimensions. The syntax for the range type is as follows:
//
//   sair-range-type ::= `!` dialect-namespace `.` `range` ('<' dom-shape '>')?
//
class RangeType : public mlir::Type::TypeBase<RangeType, ShapedType,
                                              impl::ShapedTypeStorage> {
 public:
  // Constructs RangeType from opaque types in MLIR TypeBase.
  using Base::Base;

  // Constructs an instance of RangeType in the provided context. This is a hook
  // for MLIR Builders.
  static RangeType get(DomainShapeAttr shape);

  // Returns the name of this type as it appears in the textual format without
  // the dialect prefix.
  static llvm::StringRef Name() { return "range"; }

  // Range domain shape.
  DomainShapeAttr Shape() const;
};

class MappingAttr;

// Types for n-dimensional values produced and consumed by sair operators. A
// value type specifies the shape of the domain of the value and its element
// type. The syntax for this type is the following.
//
//   value-type ::= `!` dialect-namespace `.` `value` `<` dom-shape `,` type `>`
//
class ValueType : public mlir::Type::TypeBase<ValueType, ShapedType,
                                              impl::ValueTypeStorage> {
 public:
  // Construct ValueType from opaque types in MLIR TypeBase.
  using Base::Base;

  // Construct an instance of ValueType in the provided context. This is a hook
  // for MLIR Builders.
  static ValueType get(DomainShapeAttr domain, mlir::Type element_type);

  // Construct a 0-dimensional instance of ValueType in the provided context.
  // This is a hook for MLIR Builders.
  static ValueType get(mlir::Type element_type);

  // Returns the name of this type as it appears in the textual format, without
  // the dialect prefix.
  static llvm::StringRef Name() { return "value"; }

  // Returns the type of the value elements.
  mlir::Type ElementType() const;

  // Value domain shape.
  DomainShapeAttr Shape() const;

  // Converts the type from the use domain to the def domain of the mapping.
  ValueType AccessedType(MappingAttr mapping) const;
};

}  // end namespace sair

#endif  // SAIR_SAIR_TYPES_H_
