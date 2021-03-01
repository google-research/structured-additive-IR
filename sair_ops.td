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

// Defines Sair operations.
#ifndef SAIR_SAIR_OPS_TD_
#define SAIR_SAIR_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "sair_base.td"

def SairDynRangeOp : SairOp<"dyn_range",
    [NoSideEffect, SairRangeOp, AttrSizedOperandSegments]> {
  let summary = "Defines an iteration dimension as the interval [0, $size-1].";

  let description = [{
    Defines a range dimension that iterates from 0 to $size-1 for each point
    of the domain. The general syntax is:

    ```
    <var-name> = sair.dyn_range[<domain>] <size> : !sair.range< <domain-shape> >
    ```

    For example, the following statement defines a range %n of size %n_size for
    each point of %m.

    ```mlir
    %n = sair.dyn_range[%m] %n_size(0) : !sair.range<range>
    ```

    The brackets around the domain can be ommited when the it is empty:

    ```mlir
    %k = sair.dyn_range 42 : sair.range
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairMappingArrayAttr:$mapping_array,
    Optional<SairValueOf<Index>>:$lower_bound,
    SairValueOf<Index>:$upper_bound,
    DefaultValuedAttr<IndexAttr, "1">:$step
  );

  let results = (outs SairRange:$range);

  let parser = [{return ParseDynRangeOp(parser, result);}];
  let printer = [{return Print(*this, p);}];

  DerivedAttr shape = SairResultDomainShapeAttr;

  let extraClassDeclaration = [{
    ValueOrConstant LowerBound() {
      if (lower_bound() != nullptr) return ValueOperands()[0].Get();
      return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), 0);
    }

    ValueOrConstant UpperBound() {
      return ValueOperands().back().Get();
    }
  }];
}

def SairStaticRangeOp : SairOp<"static_range", [NoSideEffect, SairRangeOp]> {
  let summary = "Defines an iteration dimension as a fixed-size interval.";

  let description = [{
    Defines a range iteration dimension, iterating from 0 to $size-1, where
    $size is a constant. The general syntax is:

    ```
    <var-name> = sair.static_range <size> : !sair.range
    ```
  }];

  let arguments = (ins
    IndexAttr:$size,
    DefaultValuedAttr<IndexAttr, "1">:$step
  );
  let results = (outs SairRange:$range);

  let parser = [{return ParseStaticRangeOp(parser, result);}];
  let printer = [{return Print(*this, p);}];

  DerivedAttr shape = SairEmptyDomainShapeAttr;

  let extraClassDeclaration = [{
    ValueOrConstant LowerBound() {
      return mlir::IntegerAttr::get(sizeAttr().getType(), 0);
    }

    ValueOrConstant UpperBound() { return sizeAttr(); }
  }];
}

def SairCopyOp : SairOp<"copy", [
    NoSideEffect,
    SairComputeOp,
    SairValueProducerOp,
    SairSameElementType<"value", "result">]> {
  let summary = "Creates a copy of a Sair value in a new iteration domain";

  let description = [{
    Copies the content of a Sair value into a new iteration domain. For example,
    the following code broadcasts a value along dimension %n.

    ```mlir
    %1 = sair.copy[d0:%m, d1:%n] %1(d0) : !sair.value<d0:range x d1:range, f32>
    ```

    The general syntax for the copy operation is the following.

    ```
    <var-name> = sair.copy[<domain>] <var-name> : <sair-value-type>
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairMappingArrayAttr:$mapping_array,
    SairValue:$value,
    SairLoopNestAttr:$loop_nest,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs SairValue:$result);

  let parser = [{return ParseCopyOp(parser, result);}];
  let printer = [{return Print(*this, p);}];

  DerivedAttr shape = SairResultDomainShapeAttr;

  let extraClassDeclaration = [{
    ValueOperand Value() { return ValueOperands()[0]; }
  }];
}

def SairFromScalarOp : SairOp<"from_scalar", [
    NoSideEffect,
    SairValueProducerOp]> {
  let description = [{Converts an SSA value into a 0-dimensional Sair value.}];

  let arguments = (ins AnyType:$value);
  let results = (outs SairValue:$result);

  let parser = [{return ParseFromScalarOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
  let verifier = [{return Verify(*this);}];

  let builders = [
    OpBuilderDAG<(ins "Value":$value),
    [{
      $_state.addOperands(value);
      $_state.addTypes(ValueType::get(value.getType()));
    }]>
  ];

  DerivedAttr shape = SairEmptyDomainShapeAttr;

  // The memory space is set to kRegister.
  DerivedAttr memory_space = DerivedAttr<"llvm::Optional<ArrayAttr>", [{
    mlir::Builder builder(getContext());
    return builder.getArrayAttr(
      {builder.getI64IntegerAttr(ValueProducerOp::kRegister)});
  }], [{$_self.getValueOr(nullptr)}]>;
}

class SairFromMemRefOpBase<string mnemonic,
                           list<OpTrait> extraTraits = [],
                           dag extraArgs = (ins)> : SairOp<mnemonic,
   !listconcat([AttrSizedOperandSegments,
                SairValueProducerOp], extraTraits)> {
  let arguments = !con((ins
    Variadic<SairRange>:$parallel_domain,
    Variadic<SairRange>:$memref_domain,
    SairMappingArrayAttr:$mapping_array,
    SairValueOf<AnyMemRef>:$memref,
    OptionalAttr<AffineMapAttr>:$access_map
  ), extraArgs);

  let results = (outs SairValue:$result);

  let parser = [{return ParseFromMemRefLike(parser, result);}];
  let printer = [{return PrintFromMemRefLike(*this, p);}];

  code commonVerifier = [{
    return VerifyFromToMemRef(
      getOperation(), parallel_domain().size(), shape(), memref(), result(),
      AccessMap());
  }];
  let verifier = commonVerifier;

  code commonClassDeclaration = [{
    ValueOperand MemRef() { return ValueOperands()[0]; }

    OperandRange domain() {
      return ChainOperandRanges(parallel_domain(), memref_domain());
    }

    llvm::SmallBitVector DimsDependingOnOperand(int sair_operand);

    mlir::AffineMap AccessMap() {
      if (access_map().hasValue()) return access_map().getValue();
      return mlir::AffineMap::getMultiDimIdentityMap(memref_domain().size(),
                                                     getContext());
    }

    mlir::MemRefType MemRefType() {
      return memref()
          .getType()
          .cast<ValueType>()
          .ElementType()
          .cast<mlir::MemRefType>();
    }
  }];
  let extraClassDeclaration = commonClassDeclaration;

  DerivedAttr shape = SairResultDomainShapeAttr;
}

def SairFromMemRefOp : SairFromMemRefOpBase<"from_memref",
    [NoSideEffect, SairInferIterationSpaceOp]> {
  let summary = "Interprets a memref as a Sair value";

  let description = [{
    Interprets a memref as a Sair value. The Sair value must have an
    hyper-rectangular domain with a number of dimensions equal to the rank of
    the memref.

    It can occur anywhere in the Sair program but before any other operation may
    write to the memref it reinterprets. Not requiring these operations to be at
    the entry of the program allows Sair to support dependent dimensions (whose
    size is provided in another memref), but it still needs a clear boundary
    between memrefs and values that would otherwise alias each other.

    This operation is only intended to connect Sair to built-in types in the
    input programs. Sair is not expected to produce such operations, and those
    present in the input are expected to be no-op in the final code, achieved by
    mapping the memref and the Sair value to the same location, with the same
    layout.

    The general syntax for the from_memref operation is the following.

    ```
    <var-name> =
      sair.from_memeref[<parallel-domain>] <var-name> shape[<memref-domain>]
        : <memref-type> -> <sair-value-type>
    ```
  }];

  let extraClassDeclaration = !strconcat([{
    int infer_iteration_space_operand() { return 0; }
  }], commonClassDeclaration);

  // The memory space is set to kMemory.
  DerivedAttr memory_space = DerivedAttr<"llvm::Optional<ArrayAttr>", [{
    mlir::Builder builder(getContext());
    return builder.getArrayAttr(
      {builder.getI64IntegerAttr(ValueProducerOp::kMemory)});
  }], [{$_self.getValueOr(nullptr)}]>;
}

def SairLoadFromMemRefOp : SairFromMemRefOpBase<"load_from_memref",
    [SairComputeOp],
    (ins SairLoopNestAttr:$loop_nest,
         SairMemorySpaceAttr:$memory_space)> {
  let summary = "Loads data from a memref to a Sair value";

  let description = [{
    Loads data from a memref into a Sair value. The Sair value must have an
    hyper-rectangular domain with a number of dimensions equal to the rank of
    the memref.

    This operation is introduced by Sair during its lowering process and is NOT
    expected to be present in the input. It is implemented as an actual load
    in the final code.

    The syntax for the load_from_memref operation is as follows.

    ```
    <var-name> =
      sair.load_from_memeref[<parallel-domain>] <var-name>
                       shape[<memref-domain>]
          : <memref-type> -> <sair-value-type>
    ```
  }];

  let verifier = !strconcat([{
    if (failed(VerifyDomainLoopNestIsIdentity(getOperation(), loop_nestAttr(),
                                              parallel_domain().size(),
                                              domain().size()))) {
      return mlir::failure();
    }
  }], commonVerifier);
}

class SairToMemRefOpBase<string mnemonic,
                         list<OpTrait> extraTraits = [],
                         dag extraArgs = (ins)> : SairOp<mnemonic,
    !listconcat([AttrSizedOperandSegments], extraTraits)> {
  let arguments = !con((ins
    Variadic<SairRange>:$parallel_domain,
    Variadic<SairRange>:$memref_domain,
    SairMappingArrayAttr:$mapping_array,
    SairValueOf<AnyMemRef>:$memref,
    SairValue:$value,
    SairDomainShapeAttr:$shape,
    OptionalAttr<AffineMapAttr>:$access_map
  ), extraArgs);

  let parser = [{return ParseToMemRefLike(parser, result);}];
  let printer = [{return PrintToMemRefLike(*this, p);}];

  code commonVerifier = [{
    return VerifyFromToMemRef(
      getOperation(), parallel_domain().size(), shape(), memref(), value(),
      AccessMap());
  }];
  let verifier = commonVerifier;

  code commonClassDeclaration = [{
    ValueOperand MemRef() { return ValueOperands()[0]; }

    ValueOperand Value() { return ValueOperands()[1]; }

    OperandRange domain() {
      return ChainOperandRanges(parallel_domain(), memref_domain());
    }

    llvm::SmallBitVector DimsDependingOnOperand(int sair_operand);

    mlir::AffineMap AccessMap() {
      if (access_map().hasValue()) return access_map().getValue();
      return mlir::AffineMap::getMultiDimIdentityMap(memref_domain().size(),
                                                     getContext());
    }

    mlir::MemRefType MemRefType() {
      return memref()
          .getType()
          .cast<ValueType>()
          .ElementType()
          .cast<mlir::MemRefType>();
    }
  }];
  let extraClassDeclaration = commonClassDeclaration;
}

def SairToMemRefOp : SairToMemRefOpBase<"to_memref",
    [SairInferIterationSpaceOp]> {
  let summary = "Interprets a Sair value as a memref";

  let description = [{
    Interprets a Sair value as a memref. The operation has an hyper-rectangular
    domain with a number of dimensions equal to the rank of the memref.

    This operation can appear anywhere in the program, but further accesses to
    the memref it writes to are not allowed. That is, the IR outside Sair will
    observe this memref elements written by this operation.

    This operation is only intended to enforce layout and location constraints
    on its operand. It is expected to be implemented as a no-op in the final
    code and is not produced by Sair.

    The general syntax for the to_memref operation is the following.

    ```
    sair.to_memref[<parallel-domain>] <memref> memref[<memref-domain>] <value>
      : <shape>, <memref-type>
    ```
  }];

  let extraClassDeclaration = !strconcat(commonClassDeclaration, [{
    int infer_iteration_space_operand() { return 0; }
  }]);
}

def SairStoreToMemRefOp : SairToMemRefOpBase<"store_to_memref",
    [SairComputeOp], (ins SairLoopNestAttr:$loop_nest)> {
  let summary = "Storse a Sair value in a memref";

  let description = [{
    Stores a Sair value in a memref. The operation has an hyper-rectangular
    domain with a number of dimensions equal to the rank of the memref.

    This operation is introduced by Sair during its lowering process and is NOT
    expected to be present in the input. It is implemented as an actual store
    in the final code.

    The syntax for the store_memref operation is as follows.

    ```
    sair.store_to_memref[<parallel-domain>] <memref>
                  memref[<memref-domain>] <value>
        : <shape>, <memref-type>
    ```
  }];

  let verifier = !strconcat([{
    if (failed(VerifyDomainLoopNestIsIdentity(getOperation(), loop_nestAttr(),
                                              parallel_domain().size(),
                                              domain().size()))) {
      return mlir::failure();
    }
  }], commonVerifier);
}

def SairReturnOp : Op<SairDialect, "return", [Terminator]> {
  let summary = "Returns scalar values from the body of a Sair operation";

  let description = [{
    In Sair, computations are expressed in terms of operations on scalars that
    are replicated across the domain of the values. This operation allows one to
  }];

  let builders = [
      OpBuilderDAG<(ins),
      [{ return build($_builder, $_state, ValueRange()); }]>];

  let arguments = (ins Variadic<AnyType>:$operands);
  let parser = [{return ParseReturnOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
}

def SairMapOp : SairOp<"map", [
    AttrSizedOperandSegments,
    SairComputeOp,
    SairValueProducerOp,
    IsolatedFromAbove]> {
  let summary = "Applies the computation across the given domain";

  let description = [{
    Applies the computation defined on scalar elements in the 'body' region to
    define a set of Sair values with the given 'domain'. The region is executed
    as many times as the domain has elements. Each execution of the body has
    access to the current indices of the domain dimensions and, additionally, to
    scalar values extracted from 'inputs' by transforming the current indices
    using the mappings provided in 'mapping_array'. The body must
    contain a single block terminated with SairReturnOp that takes as many
    operands as the SairMapOp has results. The types of thoes operands must
    match the element types of the SairMapOp results.

    The custom syntax for the operation as follows.

    ```
    sair.map[<domain>] <input-list> attributes <attr-dict> <region>
                       : <sair-shape-attr>, <function-type>
    ```
    where `<input-list>` is a potentially empty comma-separated list of
    `<value-name> <mapping>`.
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairMappingArrayAttr:$mapping_array,
    Variadic<SairValue>:$inputs,
    SairDomainShapeAttr:$shape,
    SairLoopNestAttr:$loop_nest,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs Variadic<SairValue>:$results);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    // Builds a sair.map operation and setups its block with the right arguments.
    // Input values must have !sair.value types.
    OpBuilderDAG<(ins "mlir::TypeRange":$result_types,
      "mlir::ValueRange":$domain, "mlir::ArrayAttr":$mappings_array,
      "mlir::ValueRange":$inputs, "DomainShapeAttr":$shape,
      CArg<"/*optional*/ ArrayAttr", "nullptr">:$loop_nest,
      CArg<"/*optional*/ ArrayAttr", "nullptr">:$memory_space)>
  ];

  let extraClassDeclaration = [{
    // Returns the arguments of the body block holding inputs values.
    llvm::MutableArrayRef<mlir::BlockArgument> block_inputs() {
      return block().getArguments().drop_front(domain().size());
    }

    mlir::Block& block() { return body().front(); }
  }];

  let parser = [{return ParseMapOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
  let verifier = [{return Verify(*this);}];
}

def SairMapReduceOp : SairOp<"map_reduce", [
    AttrSizedOperandSegments,
    SairComputeOp,
    SairValueProducerOp,
    IsolatedFromAbove]> {
  let summary = "Applies the computation to the domain and reduces the results";

  let description = [{
    Applies the computation defined on scalar elements in the 'body' region to
    define a set of Sair values. The computation is applied across a domain
    consisting of a parallel part and a reduction part. Only the parallel part
    forms the domain of the resulting values. The values are reduced along the
    dimensions of the reduction domain. The operation takes as operands a list
    of Sair values that serve as inputs, and another list of Sair values that
    serve as initial values of reductions. The mappings of the latter
    must not involve dimensions from the reduction domain. The body contains a
    single block with the following arguments. First, "index" type arguments
    corresponding to the coordinates in the parallel domain and the reduction
    domain. Then, as many scalar values as the operaiton has "inits", using
    their elemental type. During the iteration, these values contain partially
    reduced values. Finally, as many scalar values as the operaiton has
    "inputs", using the scalar type of "inputs". The body is expected to
    terminate with "sair.return" taking as many operands as the operation has
    "inits", of the same type as the elemental type of "inits". These values
    correspond to the partially reduced values updated in the body.

    The custom syntax for the operaiton is as follows.

    ```
    sair.map_reduce[<domain>] <input-list> reduce[<domain>] <input-list>
                              attributes <attr-dict> <region>
                              : <sair-shape-attr>, <function-type>
    ```
    where `<input-list>` is a potentially empty comma-separated list of
    `<value-name> <mapping>`.
  }];

  let arguments = (ins
    Variadic<SairRange>:$parallel_domain,
    Variadic<SairRange>:$reduction_domain,
    SairMappingArrayAttr:$mapping_array,
    Variadic<SairValue>:$inits,
    Variadic<SairValue>:$inputs,
    SairDomainShapeAttr:$shape,
    SairLoopNestAttr:$loop_nest,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs Variadic<SairValue>:$results);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    constexpr static ::llvm::StringRef kReduceKeyword = "reduce";

    OperandRange domain() {
      return ChainOperandRanges(parallel_domain(), reduction_domain());
    }

    // Indicates if the operand at the given position is used to initialize the
    // reduction.
    bool IsInitOperand(int position);

    llvm::SmallBitVector DimsDependingOnOperand(int sair_operand);

    int results_rank() { return parallel_domain().size(); }

    mlir::Block& block() { return body().front(); }
  }];

  let parser = [{return ParseMapReduceOp(parser, result);}];
  let printer = [{Print(*this, p);}];
  let verifier = [{return Verify(*this);}];
}

class SairProjectionOp<string name> : SairOp<name, [
    AttrSizedOperandSegments,
    NoSideEffect,
    SairInferIterationSpaceOp,
    SairSameElementType<"value", "result">,
    SairValueProducerOp]> {
  let arguments = (ins
    Variadic<SairRange>:$parallel_domain,
    Variadic<SairRange>:$projection_domain,
    SairMappingArrayAttr:$mapping_array,
    SairValue:$value,
    SairDomainShapeAttr:$shape,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs SairValue:$result);

  let parser = [{return ParseProjection(parser, result);}];
  let printer = [{Print(*this, p);}];

  code commonDeclarations = [{
    int results_rank() { return parallel_domain().size(); }

    OperandRange domain() {
      return ChainOperandRanges(parallel_domain(), projection_domain());
    }

    ValueOperand Value() { return ValueOperands()[0]; }

    int infer_iteration_space_operand() { return 0; }
  }];
}

def SairProjAnyOp : SairProjectionOp<"proj_any"> {
  let summary = "Takes any value of a variable along dimensions";

  let description = [{
    Projects a !sair.value along a set of projection dimensions and returns
    any element along each of these dimensions. Parallel dimensions are left
    untouched. This operation if the inverse of a broadcast.

    The custom syntax for the operation is the following.
    ```
    sair.proj_any[<parallel domain>] of[<projection domain>] <operand>
      <attr-dict> : <shape>, <element-type>
    ```
  }];

  let extraClassDeclaration = commonDeclarations;
}

def SairProjLastOp : SairProjectionOp<"proj_last"> {
  let summary = "Takes the last value of a variable along dimensions";

  let description = [{
    Projects a !sair.value along a set of projection dimensions and returns the
    last element along each of these dimensions. Parallel dimensions are left
    untouched. This operation if the inverse of a broadcast.

    The custom syntax for the operation is the following.
    ```
    sair.proj_last[<parallel domain>] of[<projection domain>] <operand>
      <attr-dict> : <shape>, <element-type>
    ```
  }];

  let extraClassDeclaration = commonDeclarations # [{
    llvm::SmallBitVector ResultsDimDependencies();
  }];
}

def SairFbyOp : SairOp<"fby", [
    AttrSizedOperandSegments,
    NoSideEffect,
    SairInferIterationSpaceOp,
    SairSameElementType<"value", "init">,
    SairSameElementType<"value", "result">,
    SairValueProducerOp]> {
  let summary = "Creates a loop-carried variable.";

  let description = [{
    For each iteration of the parallel domain, returns `init` at the first
    iteration of the sequential domain and the last value of `value` at
    following iterations of the sequantial domain.

    The custom syntax for the operation is the following
    ```
    sair.fby[<parallel domain>] <init> then[<sequential domain>] <value>
      <attr-dict> : <value-type>
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$parallel_domain,
    Variadic<SairRange>:$sequential_domain,
    SairMappingArrayAttr:$mapping_array,
    SairValue:$init,
    SairValue:$value,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs SairValue:$result);

  let parser = [{return ParseFbyOp(parser, result);}];
  let printer = [{Print(*this, p);}];

  DerivedAttr shape = SairResultDomainShapeAttr;

  let extraClassDeclaration = [{
    constexpr static ::llvm::StringRef kThenKeyword = "then";

    OperandRange domain() {
      return ChainOperandRanges(parallel_domain(), sequential_domain());
    }

    llvm::SmallBitVector DimsDependingOnOperand(int sair_operand);

    bool AllowUseBeforeDef(int index) { return index == 1; }

    llvm::SmallBitVector CarryingDimensions(int sair_operand);

    ValueOperand Init() { return ValueOperands()[0]; }

    ValueOperand Value() { return ValueOperands()[1]; }

    int infer_iteration_space_operand() { return 1; }
  }];
}

def SairProgramOp : Op<SairDialect, "program", [RegionKindInterface]> {
  let summary = "Sair program contains other Sair operations";

  let description = [{
    All Sair operations are designed to be contained in a single straight-line
    block and are not allowed to interleave with non-Sair operations.
    SairProgramOp provides the isolation necessary to ensure this property by
    serving as a single-block container for other Sair operations. All
    operations with SairOpTrait are only allowed in SairProgramOp region and,
    conversely, this region is only allowed to contain such Ops. Note that
    SairReturnOp terminator is an exception and does not have the trait anyway.

    The body region of this operation must terminate with SairExitOp.

    The custom syntax for the operation is as follows.

    ```
    sair.program (attributes <attr-dict>) <region>
    ```
  }];

  let arguments = (ins StrArrayAttr:$loop_name_table);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$results);

  let parser = [{return ParseProgramOp(parser, result);}];
  let printer = [{Print(*this, p);}];
  let verifier = [{return Verify(*this);}];

  let skipDefaultBuilders = 1;
  let builders = [
    // Builds a sair.program operation. Allocates the region its block.
    OpBuilderDAG<(ins CArg<"mlir::TypeRange", "{}">:$result_types)>
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef kLoopNameTable = "loop_name_table";

    // Generates a loop name, unique in the scope of the sair.program operation,
    // with the given prefix.
    mlir::StringAttr GenLoopName(llvm::StringRef prefix);

    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }
  }];
}

def SairExitOp : SairOp<"exit", [Terminator]> {
  let summary = "Terminates sair.program bodies";

  let description = [{
    SairExitOp terminates the body of Sair programs. It has 0-dimensional
    domain. It takes any number of Sair values that it unwrap and return to the
    calling context.

    The custom syntax for the operation is the following.
    ```
    sair.exit (<ssa-value> <mapping>),* <attr-dict>
      (':' <element-type> (, <element-type>)*)?
    ```
  }];

  let arguments = (ins
    SairMappingArrayAttr:$mapping_array,
    Variadic<SairValue>:$inputs
  );

  let parser = [{return ParseExitOp(parser, result);}];
  let printer = [{Print(*this, p);}];
  let verifier = [{return Verify(*this);}];

  DerivedAttr shape = SairEmptyDomainShapeAttr;

  let builders = [
    OpBuilderDAG<(ins CArg<"mlir::ValueRange", "{}">:$operands)>
  ];
}

// Note: this is not defined as SairOp because it is not expected inside
// SairProgramOp and produces values of types other than sair::ValueType.
def SairUndefOp : Op<SairDialect, "undef", [NoSideEffect]> {
  let summary = "Produces an undefined value of any type";

  let description = [{
    SairUndefOp produces an undefined value of any type, i.e. not necessarily a
    SairValue. This is a glue operation that maps directly to LLVM's `undef`
    value and has the same semantics. Unlike other Sair operations, SairUndefOp
    is not expected inside a Sair program and persists throuought the lowering.

    The custom syntax is as follows.
    ```
      %0 = sair.undef <attr-dict>? : <type>
    ```

    It is intended for use as reduction initalizer as follows.
    ```
      %0 = sair.undef : f32
      sair.program {
        %1 = sair.from_scalar %0 : !sair.value<(), f32>
        %2 = sair.copy[...] %0 : !sair.value<..., f32>
        sair.map_reduce[...] %2 reduce[...] ...
      }
    ```
  }];

  let results = (outs AnyType:$result);

  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def SairAllocOp : SairOp<"alloc", [
    AttrSizedOperandSegments,
    SairComputeOp,
    SairValueProducerOp]> {
  let summary = "Creates a Sair value containing allocated memrefs.";

  let description = [{
    Defines a Sair value with the given domain where each element of the value
    is a newly allocated memref of any type. If the memref has dynamic sizes,
    the allocation must be provided with Sair values containing `index`-typed
    sizes for each memref in the domain.

    The custom syntax for the operation is as follows.
    ```
    sair.alloc[<domain>] <dynamic-sizes> <attr-dict> : <value-type>
    where <dynamic-sizes> is a potentially empty comma-separated list of
    <value> <mapping>.
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairMappingArrayAttr:$mapping_array,
    Variadic<SairValueOf<Index>>:$dynamic_sizes,
    SairLoopNestAttr:$loop_nest,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs SairValueOf<AnyMemRef>:$result);

  let parser = [{return ParseAllocOp(parser, result); }];
  let printer = [{Print(*this, p);}];
  let verifier = [{return Verify(*this);}];

  DerivedAttr shape = SairResultDomainShapeAttr;

  let extraClassDeclaration = [{
    MemRefType MemType() {
      return getType().cast<ValueType>().ElementType().cast<MemRefType>();
    }
  }];
}

def SairFreeOp : SairOp<"free", [
    SairComputeOp]> {
  let summary = "Deallocates memref contained in the given Sair value.";

  let description = [{
    Frees the memrefs contained in the given Sair value. It is invalid to use
    the value after this operation.

    The custom syntax for the operation is as follows.
    ```
    sair.free[domain] <value> <mapping> <attr-dict> : <value-type>
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairMappingArrayAttr:$mapping_array,
    SairValueOf<AnyMemRef>:$value,
    SairLoopNestAttr:$loop_nest);

  let parser = [{return ParseFreeOp(parser, result); }];
  let printer = [{Print(*this, p);}];

  let extraClassDeclaration = [{
    ValueOperand Value() { return ValueOperands()[0]; }
  }];
}

#endif  // SAIR_SAIR_OPS_TD_
