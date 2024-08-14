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
// Private implementation for sair::StaticRange.
class StaticRangeTypeStorage;
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

  // TODO(b/267597147): Replace this stub with correct implementation.
  static bool classof(mlir::Type) { return true; }
};

// Base type for sair dimensions.
class DimensionType : public ShapedType {
 public:
  // Hook for MLIR type system.
  using ShapedType::ShapedType;
};

// Range type is used for values that define a dimension in a Sair iteration
// domain. A range type may dependent on some number of other iteration
// dimensions. The syntax for the range type is as follows:
//
//   sair-range-type ::= `!` dialect-namespace `.` `dyn_range` ('<' dom-shape
//   '>')?
//
class DynRangeType : public mlir::Type::TypeBase<DynRangeType, DimensionType,
                                                 impl::ShapedTypeStorage> {
 public:
  // Constructs RangeType from opaque types in MLIR TypeBase.
  using Base::Base;

  static constexpr mlir::StringLiteral name = "sair.dyn_range";

  // Constructs an instance of RangeType in the provided context. This is a hook
  // for MLIR Builders.
  static DynRangeType get(DomainShapeAttr shape);

  // Returns the name of this type as it appears in the textual format without
  // the dialect prefix.
  static llvm::StringRef Name() { return "dyn_range"; }

  // Range domain shape.
  DomainShapeAttr Shape() const;
};

// Type for ranges with a static size. The syntax for the range is the
// following.
//
//   `!sair.static_range` `<` size (`,` step)? `>`
//
class StaticRangeType
    : public mlir::Type::TypeBase<StaticRangeType, DimensionType,
                                  impl::StaticRangeTypeStorage> {
 public:
  using Base::Base;

  static constexpr mlir::StringLiteral name = "sair.static_range";

  static StaticRangeType get(int size, int step, mlir::MLIRContext *context);
  static StaticRangeType getChecked(
      llvm::function_ref<mlir::InFlightDiagnostic()> emit_error, int size,
      int step, mlir::MLIRContext *context);

  // Returns the name of this type as it appears in the textual format without
  // the dialect prefix.
  static llvm::StringRef Name() { return "static_range"; }

  // Range size.
  int size() const;

  // Range step.
  int getStep() const;

  static mlir::LogicalResult verifyInvariants(
      llvm::function_ref<mlir::InFlightDiagnostic()> emit_error, int size,
      int step);
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

  static constexpr mlir::StringLiteral name = "sair.value_type";

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
