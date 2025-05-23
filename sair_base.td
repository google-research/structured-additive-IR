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

include "mlir/IR/AttrTypeBase.td"
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

  let useDefaultAttributePrinterParser = 0;
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
        in `inverse`. May return `none` or `unknown` if the expression is not
        surjective and fully specified. `inverse` must be the inverse of the
        mapping using this sub-expression.}],
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
def IsSairMappingAttr : CPred<"isa<MappingAttr>($_self)">;

// Specifies how a Sair value is accessed in the domain of another operation.
def SairMappingAttr : Attr<IsSairMappingAttr, "mapping"> {
  let storageType = [{MappingAttr}];
  let returnType = storageType;
}

def IsSairNamedMappingAttr : CPred<"isa<NamedMappingAttr>($_self)">;

def SairNamedMappingAttr : Attr<IsSairNamedMappingAttr, "named_mapping"> {
  let storageType = [{NamedMappingAttr}];
  let returnType = storageType;
}

// Tests if an attribute is a DomainShapeAttr.
def IsSairDomainShapeAttr : CPred<"isa<DomainShapeAttr>($_self)">;

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
      return llvm::cast<ShapedType>(type).Shape();
    }]> {
  let convertFromStorage = [{$_self}];
}

// Tests if an attribute is an MappingExpr.
def IsSairMappingExpr : CPred<"isa<sair::MappingExpr>($_self)">;

// Defines an iterator on a dimension of the domain of the current operation.
def SairMappingExpr : Attr<IsSairMappingExpr, "mapping_expr"> {
  let storageType = [{MappingExpr}];
  let returnType = storageType;
}

// Identifies a numbered copy of a value to use as operand.
def SairCopyAttr : AttrDef<SairDialect, "Copy"> {
  let mnemonic = "copy";
  let parameters = (ins "unsigned":$value);

  let assemblyFormat = "`<` $value `>`";
}

// Identifies a numbered instance of an operation whose result to use as
// operand.
def SairInstanceAttr : AttrDef<SairDialect, "Instance"> {
  let mnemonic = "instance";
  let parameters = (ins "unsigned":$value);

  let assemblyFormat = "`<` $value `>`";
}

// A predicate that matches either a copy or an instance attribute.
def IsSairCopyOrInstanceAttr : Or<[SairCopyAttr.predicate,
                                   SairInstanceAttr.predicate,
                                   UnitAttr.predicate]>;
def SairCopyOrInstanceAttr : Attr<IsSairCopyOrInstanceAttr,
                                  "Sair copy or instance attribute">;

// Identifies a concrete value to use as operand.
def SairOperandsAttr
    : TypedArrayAttrBase<SairCopyOrInstanceAttr,
                         "array of Sair copy or instance attributes">;

// An attribute describing a loop to generate when lowering an operation.
def SairLoopAttr
  : Attr<CPred<"isa<::sair::LoopAttr>($_self)">, "LoopAttr">;

// A loop nest to generate when lowering an operation.
def SairLoopNestAttr
  : OptionalAttr<TypedArrayAttrBase<SairLoopAttr, "array of LoopAttr">>;

// An attribute that specifies how a value is stored in a buffer.
def SairBufferAttr
  : Attr<CPred<"isa<::sair::BufferAttr>($_self)">, "BufferAttr">;

// An attribute that specifies how results of a sair operation are stored in
// memory.
def SairStorageAttr : OptionalAttr<ArrayAttr>;

// An attribute that specifies how to implement an operation.
def SairDecisionsAttr
  : Attr<CPred<"isa<::sair::DecisionsAttr>($_self)">, "DecisionsAttr">;

// An attribute that specifies instances of an operation.
def SairInstancesAttr
  : TypedArrayAttrBase<SairDecisionsAttr, "array of Sair decisions">;

// An attribute that specifies copies of the result of an operation.
def SairCopiesAttr
  : TypedArrayAttrBase<SairInstancesAttr, "array of array of Sair decisions">;

//===----------------------------------------------------------------------===//
// Sair Types
//===----------------------------------------------------------------------===//

// Type constraint for dimensions.
def SairDimension : Type<CPred<"isa<DimensionType>($_self)">, "dimension">;

// Type constraint for ranges.
def SairDynRange : Type<CPred<"isa<DynRangeType>($_self)">, "range">;

// Type constraint for static ranges.
def SairStaticRange
  : Type<CPred<"isa<StaticRangeType>($_self)">, "static_range">;

// Type constraint for Sair values.
def SairValue : Type<CPred<"isa<ValueType>($_self)">, "value">;

// Predicate that checks the element type of a Sair value.
class SairElementTypePred<Type type>
  : SubstLeaves<"$_self", "llvm::cast<ValueType>($_self).ElementType()",
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

// Ensures that a Sair op has at most one instance specified.
def SingleInstance: PredOpTrait<
    "op has at most one instance",
    CPred<"!cast<SairOp>($_op).getInstances().has_value() || "
          "cast<SairOp>($_op).getInstances()->getValue().size() <= 1">>;

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
      "DomainShapeAttr", "getShape"
    >,
    InterfaceMethod<
      "Returns the dimensions that compose the domain of the operation",
      "mlir::Operation::operand_range", "getDomain", (ins), [{}], [{
        return mlir::OperandRange(nullptr, 0);
      }]
    >,
    InterfaceMethod<
      "Returns the number of dimensions in the domains of produced Sair values",
      "int", "results_rank", (ins), [{}], [{
        return cast<ConcreteOp>(this->getOperation()).getShape().NumDimensions();
      }]
    >,
    InterfaceMethod<
      "Returns the mappings for the !sair.value operands",
      "mlir::ArrayAttr", "getMappingArray", (ins), [{}], [{
        return mlir::ArrayAttr::get(this->getOperation()->getContext(), {});
      }]
    >,
    InterfaceMethod<
      "Sets the mapping for the given Sair value operand",
      "void", "SetMapping",
      (ins "int":$position, "MappingAttr":$mapping), [{}], [{
        ::sair::SetMapping($_op, position, mapping);
      }]
    >,
    InterfaceMethod<
      "Returns the Sair values accessed by the operation",
      "ValueOperandRange", "ValueOperands", (ins), [{}], [{
        return ::sair::ValueOperands(ConcreteOp($_op));
      }]
    >,
    InterfaceMethod<
      "Returns the operation domain with dependency mappings between dimensions",
      "llvm::SmallVector<ValueAccess>", "DomainWithDependencies", (ins), [{}], [{
        ConcreteOp op($_op);
        llvm::SmallVector<ValueAccess> result;
        result.reserve(op.getDomain().size());
        for (auto [value, shape_dim] :
             llvm::zip(op.getDomain(), op.getShape().Dimensions())) {
          result.push_back({value, shape_dim.dependency_mapping()});
        }
        return result;
      }]
    >,
    InterfaceMethod<
      [{Returns a mask of the dimensions of the current operation that depend on
        `sair_operand`. The operand must be executed before entering these
        dimensions.}],
      "llvm::SmallBitVector", "DimsDependingOnOperand",
      (ins "int":$sair_operand), [{}], [{
        int size = cast<ConcreteOp>(this->getOperation()).getDomain().size();
        return llvm::SmallBitVector(size);
      }]
    >,
    InterfaceMethod<
      [{Returns the dimensions that must finish before using results}],
      "llvm::SmallBitVector", "ResultsDimDependencies",
      (ins), [{}], [{
        int size = cast<ConcreteOp>(this->getOperation()).getDomain().size();
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
        return llvm::SmallBitVector($_op.getDomain().size());
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
    >,
    InterfaceMethod<
      [{Indicates if the operation has exactly one instance and no copy.}],
      "bool", "HasExactlyOneInstance", (ins), [{}], [{
        return sair::HasExactlyOneInstance(cast<SairOp>(this->getOperation()));
      }]
    >,
    InterfaceMethod<
      [{Returns the instances attribute if present, std::nullopt otherwise.}],
      "std::optional<mlir::ArrayAttr>", "getInstances", (ins), [{}],
      [{
        // PlacehoderOp is a special op that gets introduced during code
        // generation and is not expected to have external decisions.
        static_assert(std::is_same<ConcreteOp, sair::SairPlaceholderOp>::value,
                      "Sair ops must have the 'instances' attribute");
        return std::nullopt;
      }]
    >,
    InterfaceMethod<
      "Returns the number of instances of the operation",
      "int", "NumInstances", (ins), [{}], [{
        std::optional<mlir::ArrayAttr> instances = $_op.getInstances();
        if (!instances.has_value()) return 0;
        return instances->size();
      }]>,
    InterfaceMethod<
      "Returns lowering decisions for all instances as an array attribute",
      "mlir::ArrayAttr", "GetAllDecisions", (ins), [{}], [{
        std::optional<mlir::ArrayAttr> instances = $_op.getInstances();
        if (!instances) return mlir::ArrayAttr();
        return *instances;
      }]
    >,
    InterfaceMethod<
      "Returns lowering decisions for the given operation instance",
      "DecisionsAttr", "GetDecisions", (ins "int":$instance), [{}], [{
        mlir::ArrayAttr instances = *$_op.getInstances();
        return llvm::cast<DecisionsAttr>(instances.getValue()[instance]);
      }]
    >,
    InterfaceMethod<
      "Sets lowering decisions for the operation",
      "void", "SetDecisions", (ins "int":$instance, "DecisionsAttr":$value),
      [{}], [{
        auto instances = llvm::to_vector<4>(
          $_op.getInstances()->getValue());
        instances[instance] = value;
        $_op.setInstancesAttr(mlir::ArrayAttr::get($_op.getContext(), instances));
      }]
    >,
    InterfaceMethod<
      "Appends an instance to the list of instances",
      "void", "AddInstance", (ins "DecisionsAttr":$value),
      [{}], [{
        llvm::SmallVector<mlir::Attribute> instances;
        if ($_op.getInstances().has_value()) {
          instances = llvm::to_vector<4>(
            $_op.getInstances()->getValue());
        }
        instances.push_back(value);
        $_op.setInstancesAttr(mlir::ArrayAttr::get($_op.getContext(), instances));
      }]
    >
  ];

  let verify = [{return VerifySairOp(op);}];

  let extraClassDeclaration = [{
    // The string identifier used for mapping attribute in Sair ops.
    static constexpr llvm::StringRef kMappingAttrName = "mapping_array";

    // The string identifier used for instances attribute in Sair ops.
    static constexpr llvm::StringRef kInstancesAttrName = "instances";
  }];
}

// Base class for all Sair operators.
class SairOp<string mnemonic,
             list<Trait> traits = [],
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
}

def SairValueProducerOp : OpInterface<"ValueProducerOp"> {
  let description = [{
    A Sair operation that produces Sair values.

    Value producer operations have a `copies` attribute that lists copy
    operations to insert to make copies of the operation results. The copy
    attribute is an array of arrays, with one array for each result of the
    operation. The array corresponding to a result contains `DecisionAttr`
    attributes that describes decisions for the copy operation to insert.
  }];

  let methods = [
    InterfaceMethod<
      "List of copies of the given result. This is an array of DecisionAttr.",
      "llvm::ArrayRef<mlir::Attribute>", "GetCopies", (ins "int":$result), [{}], [{
          auto all_copies = $_op.getCopiesAttr();
          if (all_copies == nullptr) return {};
          return llvm::cast<mlir::ArrayAttr>(all_copies.getValue()[result]).getValue();
      }]>,
    InterfaceMethod<
      "Indicates if the operation has any copy set in its `copies` attribute`",
      "bool", "HasCopies", (ins), [{}], [{
        auto all_copies = $_op.getCopiesAttr();
        if (all_copies == nullptr) return false;
        return llvm::any_of(all_copies.getValue(), [](mlir::Attribute attr) {
          return !llvm::cast<mlir::ArrayAttr>(attr).empty();
        });
      }]>,
    InterfaceMethod<
      "Set decisions for the given copy of the given result.",
      "void", "SetCopy",
      (ins "int":$result, "int":$copy, "DecisionsAttr":$decisions), [{}], [{
        auto all_copies = llvm::to_vector<4>(*$_op.getCopies());
        auto result_copies_attr = llvm::cast<mlir::ArrayAttr>(all_copies[result]);
        auto result_copies = llvm::to_vector<4>(result_copies_attr.getValue());

        result_copies[copy] = decisions;
        mlir::MLIRContext *context = $_op.getContext();
        all_copies[result] = mlir::ArrayAttr::get(context, result_copies);
        $_op.setCopiesAttr(mlir::ArrayAttr::get(context, all_copies));
      }]>
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef kCopiesAttrName = "copies";
  }];

  let verify = [{return sair::VerifyValueProducerOp(op);}];
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
    InterfaceMethod<"Returns the range step", "int", "Step">,
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
}

def SairFromToMemRefOp : OpInterface<"FromToMemRefOp"> {
  let description = [{Sair operation that accesses an external memref}];

  let methods = [
    InterfaceMethod<"Domain dimensions not mapping to memref dimensions",
                    "ValueRange", "getParallelDomain">,
    InterfaceMethod<"Domain dimensions mapping to memref dimensions",
                    "ValueRange", "getMemrefDomain">,
    InterfaceMethod<"Memref accessed", "ValueOperand", "MemRef">,
    InterfaceMethod<"Buffer name", "llvm::StringRef", "getBufferName">,
    InterfaceMethod<"Memref type", "mlir::MemRefType", "MemRefType", (ins),
                    [{}], [{
                      return llvm::cast<mlir::MemRefType>(
                          $_op.MemRef().GetType().ElementType());
                    }]>,
    InterfaceMethod<"Mapping from value domain to layout", "MappingAttr",
                    "Layout", (ins), [{}], [{
                      return MappingAttr::GetIdentity(
                                 $_op.getContext(), $_op.getMemrefDomain().size())
                          .ShiftRight($_op.getParallelDomain().size());
                    }]>
  ];
}

#endif  // SAIR_SAIR_BASE_TD_
