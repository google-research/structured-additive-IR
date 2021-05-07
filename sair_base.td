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

#ifndef SAIR_SAIR_BASE_TD_
#define SAIR_SAIR_BASE_TD_

include "mlir/IR/OpBase.td"

def SairDialect : Dialect {
  let name = "sair";

  let summary = "Structured additive program representation";

  let description = [{
    Program representation designed to discover optimization opportunities early
    in the compilation process and to express implementation decisions without
    loosing the high-level structure of the code (hence the "additive" part of
    the name).
  }];

  let cppNamespace = "::sair";
}

//===----------------------------------------------------------------------===//
// Sair Attributes
//===----------------------------------------------------------------------===//

def MappingExpr : AttrInterface<"MappingExpr"> {
  let description = [{
    Interface class for mapping expressions. Mapping expressions
    specify how to access a dimension of another domain given the index along
    dimensions of the current domain.
  }];

  let methods = [
    InterfaceMethod<
      [{Applies a function to all sub-expressions and constructs a result in
        post-order}],
      "MappingExpr", "Map",
      (ins "llvm::function_ref<MappingExpr(MappingExpr)>":$function)
    >,
    InterfaceMethod<
      "Applies a function to all sub-expressions in post-order.",
      "void", "Walk",
      (ins "llvm::function_ref<void(MappingExpr)>":$function)
    >,
    InterfaceMethod<
      [{Merges two mapping expressions into a single one. Calls $on_leaves to
        merge expressions that do not have a sub-expression. Returns nullptr on
        failure.}],
      "MappingExpr", "Unify",
      (ins "MappingExpr":$other,
           "llvm::function_ref<MappingExpr(MappingExpr, MappingExpr)>":$on_leaves)
    >,
    InterfaceMethod<
      [{Sets `inverses[i]` to the inverse of the current expression (and its
      context) with regard to the i-th dimension of the domain. `inverses` size
      must be equal to the size of the domain.

      `context_inverse` is the inverse of the surrounding sub-expression. If
      the current sub-expression is `e` in `C(e)`, then `context_inverse` is
      the inverse of `C(x)`.

      Returns a failure if the inverse of this expression cannot be unified
      with inverses already set in `inverses`.}],
      "mlir::LogicalResult", "SetInverse", (ins
        "MappingExpr":$context_inverse,
        "llvm::MutableArrayRef<MappingExpr>":$inverses
      )
    >,
    InterfaceMethod<
      [{Finds the subexpression corresponding to the inverse of this expression
        in `inverse`. Expects the expression to be fully specified. `inverse`
        must be the inverse of the mapping using this sub-expression.}],
      "MappingExpr", "FindInInverse",
      (ins "llvm::ArrayRef<MappingExpr>":$inverse)
    >,
    InterfaceMethod<
      [{Converts the expression into an affine expression. The expression must
        be fully specified}],
      "mlir::AffineExpr", "AsAffineExpr"
    >,
    InterfaceMethod<
      [{Canonicalizes the expression.}], "MappingExpr", "Canonicalize"
    >,
  ];

  let extraClassDeclaration = [{
    // Substitute dimensions by the given expressions.
    MappingExpr SubstituteDims(llvm::ArrayRef<MappingExpr> exprs) const;

    // Return a mask of the dimensions appearing in the expression.
    llvm::SmallBitVector DependencyMask(int domain_size) const;

    // Sets the dimensions appearing in the expression in `mask`.
    void SetDependenciesInMask(llvm::SmallBitVector& mask) const;

    // Returns `true` if any subexpression is `none`.
    bool HasNoneExprs() const;

    // Returns `true` if any subexpression is `?`.
    bool HasUnknownExprs() const;

    // Returns the minimal domain size for the expression to be valid.
    int MinDomainSize() const;

  }];
}

// Tests if an attribute is an MappingAttr.
def IsSairMappingAttr : CPred<"$_self.isa<MappingAttr>()">;

// Specifies how a Sair value is accessed in the domain of another operation.
def SairMappingAttr : Attr<IsSairMappingAttr, "mapping"> {
  let storageType = [{MappingAttr}];
  let returnType = storageType;
}

def IsSairNamedMappingAttr : CPred<"$_self.isa<NamedMappingAttr>()">;

def SairNamedMappingAttr : Attr<IsSairNamedMappingAttr, "named_mapping"> {
  let storageType = [{NamedMappingAttr}];
  let returnType = storageType;
}

// Tests if an attribute is a DomainShapeAttr.
def IsSairDomainShapeAttr : CPred<"$_self.isa<DomainShapeAttr>()">;

// Specifies the shape of a Sair iteration domain.
def SairDomainShapeAttr : Attr<IsSairDomainShapeAttr, "domain shape"> {
  let storageType = [{DomainShapeAttr}];
  let returnType = storageType;
}

// Derived attribute that returns a domain shape with 0 dimensions.
def SairEmptyDomainShapeAttr :
  DerivedAttr<"DomainShapeAttr",
              [{return DomainShapeAttr::get(getContext());}]> {
  let convertFromStorage = [{$_self}];
}

// Derived attribute that returns the shape of results.
def SairResultDomainShapeAttr :
    DerivedAttr<"DomainShapeAttr", [{
      mlir::Type type = getOperation()->getResult(0).getType();
      return type.cast<ShapedType>().Shape();
    }]> {
  let convertFromStorage = [{$_self}];
}

// Tests if an attribute is an MappingExpr.
def IsSairMappingExpr : CPred<"$_self.isa<sair::MappingExpr>()">;

// Defines an iterator on a dimension of the domain of the current operation.
def SairMappingExpr : Attr<IsSairMappingExpr, "mapping_expr"> {
  let storageType = [{MappingExpr}];
  let returnType = storageType;
}

// Helper to define a struct attribute with some additional constraints. The
// attribute passed to constraints has the struct type.
class SairStructAttr<string name,
                     list<StructFieldAttr> fields,
                     list<AttrConstraint> constraints = []>
  : Confined<
      StructAttr<name, SairDialect, fields>,
      !foreach(c, constraints, AttrConstraint<
        SubstLeaves<"$_self", "$_self.cast<" # name # ">()", c.predicate>,
        c.summary>
      )>;

// An attribute describing a loop to generate when lowering an operation.
def SairLoopAttr : SairStructAttr<"LoopAttr", [
  // Name of the loop to generate.
  StructFieldAttr<"name", StrAttr>,
  StructFieldAttr<"iter", SairMappingExpr>
]>;

// A loop nest to generate when lowering an operation.
def SairLoopNestAttr
  : OptionalAttr<TypedArrayAttrBase<SairLoopAttr, "array of LoopAttr">>;

// An attribute that specifies how a value is stored in a buffer.
def SairBufferAttr : SairStructAttr<"BufferAttr", [
  StructFieldAttr<"space", StrAttr>,
  StructFieldAttr<"name", OptionalAttr<StrAttr>>,
  StructFieldAttr<"layout", OptionalAttr<SairNamedMappingAttr>>,
]>;

// An attribute that specifies how results of a sair operation are stored in
// memory.
def SairStorageAttr : OptionalAttr<ArrayAttr>;

//===----------------------------------------------------------------------===//
// Sair Types
//===----------------------------------------------------------------------===//

// Type constraint for ranges.
def SairRange : Type<CPred<"$_self.isa<RangeType>()">, "range">;

// Type constraint for Sair values.
def SairValue : Type<CPred<"$_self.isa<ValueType>()">, "value">;

// Predicate that checks the element type of a Sair value.
class SairElementTypePred<Type type>
  : SubstLeaves<"$_self", "$_self.cast<ValueType>().ElementType()",
    type.predicate>;

// Type constraint for Sair values with a specific element type.
class SairValueOf<Type type>
  : Type<And<[SairValue.predicate, SairElementTypePred<type>]>,
      "sair.value with " # type.summary # " elements">;

// Type for arrays of mappings.
def SairMappingArrayAttr :
  TypedArrayAttrBase<SairMappingAttr, "mappings array">;

//===----------------------------------------------------------------------===//
// Sair Traits
//===----------------------------------------------------------------------===//

// Ensures that two named values have the same element type. Values can be Sair
// values or mlir shaped values.
class SairSameElementType<string lhs, string rhs>
  : PredOpTrait<lhs # " and " # rhs # " have the same element type",
      CPred<"IsSameElementType(" # lhs # "(), " # rhs # "())">>;

//===----------------------------------------------------------------------===//
// Sair Interfaces
//===----------------------------------------------------------------------===//

def SairOpInterface : OpInterface<"SairOp"> {
  let description = [{
    Common interface for Sair operations, that requires them to expose their
    iteration domain.

    The operation must have an "mapping_attr" attribute unless
    ValueOperands returns an empty range.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the shape of the iteration domain of the operation",
      "DomainShapeAttr", "shape"
    >,
    InterfaceMethod<
      "Returns the dimensions that compose the domain of the operation",
      "mlir::Operation::operand_range", "domain", (ins), [{}], [{
        return mlir::OperandRange(nullptr, 0);
      }]
    >,
    InterfaceMethod<
      "Returns the number of dimensions in the domains of produced Sair values",
      "int", "results_rank", (ins), [{}], [{
        return cast<ConcreteOp>(this->getOperation()).shape().NumDimensions();
      }]
    >,
    InterfaceMethod<
      "Returns the mappings for the !sair.value operands",
      "mlir::ArrayAttr", "mapping_array", (ins), [{}], [{
        return mlir::ArrayAttr::get(this->getOperation()->getContext(), {});
      }]
    >,
    InterfaceMethod<
      "Sets the mapping for the given Sair value operand",
      "void", "SetMapping",
      (ins "int":$position, "MappingAttr":$mapping), [{}], [{
        ::sair::SetMapping(ConcreteOp($_op), position, mapping);
      }]
    >,
    InterfaceMethod<
      "Returns the Sair values accessed by the operation",
      "ValueOperandRange", "ValueOperands", (ins), [{}], [{
        return ::sair::ValueOperands(ConcreteOp($_op));
      }]
    >,
    InterfaceMethod<
      [{Returns a mask of the dimensions of the current operation that depend on
        `sair_operand`. The operand must be executed before entering these
        dimensions.}],
      "llvm::SmallBitVector", "DimsDependingOnOperand",
      (ins "int":$sair_operand), [{}], [{
        int size = cast<ConcreteOp>(this->getOperation()).domain().size();
        return llvm::SmallBitVector(size);
      }]
    >,
    InterfaceMethod<
      [{Returns the dimensions that must finish before using results}],
      "llvm::SmallBitVector", "ResultsDimDependencies",
      (ins), [{}], [{
        int size = cast<ConcreteOp>(this->getOperation()).domain().size();
        return llvm::SmallBitVector(size);
      }]
    >,
    InterfaceMethod<
      [{Indicates a Sair operand can be definied after this operation}],
      "bool", "AllowUseBeforeDef", (ins "int":$sair_operand), [{}],
      [{return false;}]
    >,
    InterfaceMethod<
      [{Returns a mask of the dimensions of the current operation that carry
        dependencies accross iterations for the given !sair.value operand}],
      "llvm::SmallBitVector", "CarryingDimensions",
      (ins "int":$sair_operand), [{}], [{
        int size = cast<ConcreteOp>(this->getOperation()).domain().size();
        return llvm::SmallBitVector(size);
      }]
    >,
    InterfaceMethod<
      [{Creates a copy of the operation with a new domain. May steal the body of
        the current operation but does not erase it.}],
      "SairOp", "ReCreateWithNewDomain", (ins
        "llvm::ArrayRef<llvm::SmallVector<mlir::Value>>":$new_domains,
        "DomainShapeAttr":$new_shape,
        "MappingAttr":$new_to_old_mapping,
        "mlir::OpBuilder &":$builder
      )
    >,
    InterfaceMethod<
      [{Indicates the size of each sub-domain. The first sub-domain is always
        the parallel domain}],
      "llvm::SmallVector<int, 2>", "SubDomains", (ins), [{}]
    >
  ];

  let verify = [{return VerifySairOp(op);}];
}

// Base class for all Sair operators.
class SairOp<string mnemonic,
             list<OpTrait> traits = [],
             Dialect dialect = SairDialect>
  : Op<
      dialect, mnemonic,
      !listconcat(traits, [
        SairOpInterface,
        DeclareOpInterfaceMethods<SairOpInterface>
      ])
    > {

  let hasCanonicalizer = 1;
}

def SairComputeOp : OpInterface<"ComputeOp"> {
  let description = [{
    A Sair operation that performs a computation and results in actual code.
    This is as oppposition to operations defining iteration dimensions and
    casting sair to/from Sair values.

    Assumes that the operation has an optional `loop_nest` attribute of type
    `SairLoopNestAttr`. Checks dimension names are only used once in the loop
    nest, that dimensions referenced by the loop nest are dimensions of the
    domain, and that the loop nest iterates on the full domain.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the loop nest to generate when lowering the operation",
      "Optional<mlir::ArrayAttr>", "loop_nest"
    >,
    InterfaceMethod<
      "Sets the loop nest to generate when lowering the operation",
      "void", "setLoopNest", (ins "mlir::ArrayAttr":$loop_nest), [{
        $_op.loop_nestAttr(loop_nest);
      }]
    >,
    InterfaceMethod<
      [{Returns the loops in the loop_nest attribute. Returns the empty loop nest
        if the attribute is missing"}],
      "llvm::ArrayRef<mlir::Attribute>", "LoopNestLoops", (ins), [{}], [{
        if (!$_op.loop_nest().hasValue()) {
          return llvm::ArrayRef<mlir::Attribute>();
        }
        return $_op.loop_nest().getValue().getValue();
      }]
    >,
    InterfaceMethod<
      "Returns the storage of the values produced by the operation",
      "Optional<mlir::ArrayAttr>", "storage"
    >,
    InterfaceMethod<
      "Returns the storage of the values produced by the operation",
      "BufferAttr", "Storage", (ins "int":$result), [{}], [{
        if (!$_op.storage().hasValue()) return nullptr;
        mlir::Attribute attr = $_op.storage().getValue().getValue()[result];
        if (attr.isa<mlir::UnitAttr>()) return nullptr;
        return attr.template cast<BufferAttr>();
      }]
    >,
    InterfaceMethod<
      "Sets the storage of the values produced by the operation",
      "void", "SetStorage", (ins "int":$result, "BufferAttr":$buffer), [{}], [{
        SetInArrayAttr(
          $_op, kStorageAttrName, $_op->getNumResults(), result, buffer);
      }]
    >
  ];

  let verify = [{return sair::VerifyComputeOp(op);}];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef kLoopNestAttrName = "loop_nest";
    static constexpr llvm::StringRef kStorageAttrName = "storage";
  }];
}

def SairInferIterationSpaceOp : OpInterface<"InferIterationSpaceOp"> {
  let description = [{
    A Sair operation whose loop nest is infered from the loop nest of an
    operand.
  }];

  let methods = [
    InterfaceMethod<
      "Position of the Sair value operand to inherit the iteration space from.",
      "int", "infer_iteration_space_operand"
    >
  ];
}

def SairRangeOp : OpInterface<"RangeOp"> {
  let description = [{A Sair operation that defines a !sair.range}];

  let methods = [
    InterfaceMethod<"Returns the range step", "llvm::APInt", "step">,
    InterfaceMethod<
      "Returns the upper bound of the range", "ValueOrConstant", "LowerBound"
    >,
    InterfaceMethod<
      "Returns the lower bound of the range", "ValueOrConstant", "UpperBound"
    >,
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef kStepAttrName = "step";
  }];

  let verify = [{return sair::VerifyRangeOp(op);}];
}

def SairFromToMemRefOp : OpInterface<"FromToMemRefOp"> {
  let description = [{Sair operation that accesses an external memref}];

  let methods = [
    InterfaceMethod<"Domain dimensions not mapping to memref dimensions",
                    "ValueRange", "parallel_domain">,
    InterfaceMethod<"Domain dimensions mapping to memref dimensions",
                    "ValueRange", "memref_domain">,
    InterfaceMethod<"Memref accessed", "ValueOperand", "MemRef">,
    InterfaceMethod<"Buffer name", "llvm::StringRef", "buffer_name">,
    InterfaceMethod<"Memref type", "mlir::MemRefType", "MemRefType", (ins),
                    [{}], [{
                      return $_op.MemRef()
                          .GetType()
                          .ElementType()
                          .template cast<mlir::MemRefType>();
                    }]>,
    InterfaceMethod<"Mapping from value domain to layout", "MappingAttr",
                    "Layout", (ins), [{}], [{
                      return MappingAttr::GetIdentity(
                                 $_op.getContext(), $_op.memref_domain().size())
                          .ShiftRight($_op.parallel_domain().size());
                    }]>
  ];
}

#endif  // SAIR_SAIR_BASE_TD_
