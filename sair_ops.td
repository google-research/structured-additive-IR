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

def SairRangeOp : SairOp<"range",
    [NoSideEffect, SairOneResult]> {
  let summary = "Defines an iteration dimension as the interval [0, $size-1].";

  let description = [{
    Defines a range dimension that iterates from 0 to $size-1 for each point
    of the domain. The general syntax is:

    ```
    <var-name> = sair.range[<domain>] <size> : !sair.range< <domain-shape> >
    ```

    For example, the following statement defines a range %n of size %n_size for
    each point of %m.

    ```mlir
    %n = sair.range[%m] %n_size(0) : !sair.range<range>
    ```

    The brackets around the domain can be ommited when the it is empty:

    ```mlir
    %k = sair.range 42 : sair.range
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairAccessPatternArrayAttr:$access_pattern_array,
    SairValueOf<Index>:$size
  );

  let results = (outs SairRange:$range);

  let parser = [{return ParseRangeOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
}

def SairStaticRangeOp : SairOp<"static_range", [NoSideEffect]> {
  let summary = "Defines an iteration dimension as a fixed-size interval.";

  let description = [{
    Defines a range iteration dimension, iterating from 0 to $size-1, where
    $size is a constant. The general syntax is:

    ```
    <var-name> = sair.static_range <size> : !sair.range
    ```
  }];

  let arguments = (ins IndexAttr:$size);
  let results = (outs SairRange:$range);

  let parser = [{return ParseStaticRangeOp(parser, result);}];
  let printer = [{return Print(*this, p);}];

  DerivedAttr shape = SairEmptyDomainShapeAttr;
}

def SairCopyOp : SairOp<"copy", [
    NoSideEffect,
    SairComputeOp,
    SairValueProducerOp,
    SairOneResult,
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
    SairAccessPatternArrayAttr:$access_pattern_array,
    SairValue:$value,
    SairLoopNestAttr:$loop_nest,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs SairValue:$result);

  let parser = [{return ParseCopyOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
}

def SairFromScalarOp : SairOp<"from_scalar", [
    NoSideEffect,
    SairOneResult,
    SairValueProducerOp]> {
  let description = [{Converts an SSA value into a 0-dimensional Sair value.}];

  let arguments = (ins AnyType:$value);
  let results = (outs SairValue:$result);

  let parser = [{return ParseFromScalarOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
  let verifier = [{return Verify(*this);}];

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Value value", [{
      state.addOperands(value);
      state.addTypes(builder.getType<ValueType>(value.getType()));
    }]>
  ];

  // The memory space is set to kRegister.
  DerivedAttr memory_space = DerivedAttr<"llvm::Optional<ArrayAttr>", [{
    mlir::Builder builder(getContext());
    return builder.getArrayAttr(
      {builder.getI64IntegerAttr(ValueProducerOp::kRegister)});
  }], [{$_self.getValueOr(nullptr)}]>;
}

def SairFromMemRefOp : SairOp<"from_memref", [
    NoSideEffect,
    SairHyperRectangularDomain,
    SairOneResult,
    SairSameElementType<"memref", "result">,
    SairSameRank<"result", "memref">,
    SairValueProducerOp]> {
  let summary = "Interprets a memref as a Sair value";

  let description = [{
    Interprets a memref as a Sair value. The Sair value must have an
    hyper-rectangular domain with a number of dimensions equal to the rank of
    the memref.

    This instruction is only intended to translate values from one dialect to
    the other and is expected to be implemented as a no-op in the final code, by
    mapping the memref and the Sair value to the same location, with the same
    layout.

    The general syntax for the from_memref operation is the following.

    ```
    <var-name> = sair.from_memeref[<domain>] <var-name>
      : <memref-type> -> <sair-value-type>
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    AnyMemRef:$memref
  );

  let results = (outs SairValue:$result);

  let parser = [{return ParseFromMemRefOp(parser, result);}];
  let printer = [{return Print(*this, p);}];

  // The memory space is set to kMemory.
  DerivedAttr memory_space = DerivedAttr<"llvm::Optional<ArrayAttr>", [{
    mlir::Builder builder(getContext());
    return builder.getArrayAttr(
      {builder.getI64IntegerAttr(ValueProducerOp::kMemory)});
  }], [{$_self.getValueOr(nullptr)}]>;
}

def SairToMemRefOp : SairOp<"to_memref", [
    SairHyperRectangularDomain,
    SairSameElementType<"value", "memref">]> {
  let summary = "Indicates that a value should be stored in a memref.";

  let description = [{
    Indicates that a Sair value should be stored inside a memref. The operation
    has an hyper-rectangular domain with a number of dimensions equal to the
    rank of the memref.

    This instruction is only intended to enforce layout and location constraints
    on its operand. It is expected to be implemented as a no-op in the final
    code.

    The general syntax for the to_memref operation is the following.

    ```
    sair.to_memref[<domain>] <value-name> <access-pattern>, <memref-name>
      : <memref-type>
    ```
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairAccessPatternArrayAttr:$access_pattern_array,
    SairValue:$value,
    AnyMemRef:$memref
  );

  let parser = [{return ParseToMemRefOp(parser, result);}];
  let printer = [{return Print(*this, p);}];

  let extraClassDeclaration = [{DomainShapeAttr shape();}];
}

def SairReturnOp : Op<SairDialect, "return", [Terminator]> {
  let summary = "Returns scalar values from the body of a Sair operation";

  let description = [{
    In Sair, computations are expressed in terms of operations on scalars that
    are replicated across the domain of the values. This operation allows one to
  }];

  let builders = [
      OpBuilder<"OpBuilder &builder, OperationState &result",
                [{ return build(builder, result, ValueRange()); }]>];

  let arguments = (ins Variadic<AnyType>:$operands);
  let parser = [{return ParseReturnOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
}

def SairMapOp : SairOp<"map", [
    AttrSizedOperandSegments,
    SairComputeOp,
    SairOpWithBody,
    SairValueProducerOp,
    IsolatedFromAbove]> {
  let summary = "Applies the computation across the given domain";

  let description = [{
    Applies the computation defined on scalar elements in the 'body' region to
    define a set of Sair values with the given 'domain'. The region is executed
    as many times as the domain has elements. Each execution of the body has
    access to the current indices of the domain dimensions and, additionally, to
    scalar values extracted from 'inputs' by transforming the current indices
    using the access patterns provided in 'access_pattern_array'. The body must
    contain a single block terminated with SairReturnOp that takes as many
    operands as the SairMapOp has results. The types of thoes operands must
    match the element types of the SairMapOp results.

    The custom syntax for the operation as follows.

    ```
    sair.map[<domain>] <input-list> attributes <attr-dict> <region>
                       : <sair-shape-attr>, <function-type>
    ```
    where `<input-list>` is a potentially empty comma-separated list of
    `<value-name> <access-pattern>`.
  }];

  let arguments = (ins
    Variadic<SairRange>:$domain,
    SairAccessPatternArrayAttr:$access_pattern_array,
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
    OpBuilder<[{mlir::OpBuilder &builder,
                mlir::OperationState &result,
                mlir::TypeRange result_types,
                mlir::ValueRange domain,
                mlir::ArrayAttr access_patterns_array,
                mlir::ValueRange inputs,
                DomainShapeAttr shape,
                /*optional*/ ArrayAttr loop_nest = nullptr,
                /*optional*/ ArrayAttr memory_space = nullptr}]>
  ];

  let extraClassDeclaration = [{
    // Returns the arguments of the body block holding inputs values.
    llvm::MutableArrayRef<mlir::BlockArgument> block_inputs() {
      return block().getArguments().drop_front(domain().size());
    }
  }];

  let parser = [{return ParseMapOp(parser, result);}];
  let printer = [{return Print(*this, p);}];
  let verifier = [{return Verify(*this);}];
}

def SairMapReduceOp : SairOp<"map_reduce", [
    AttrSizedOperandSegments,
    SairComputeOp,
    SairOpWithBody,
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
    serve as initial values of reductions. The access patterns of the latter
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
    `<value-name> <access-pattern>`.
  }];

  let arguments = (ins
    Variadic<SairRange>:$parallel_domain,
    Variadic<SairRange>:$reduction_domain,
    SairAccessPatternArrayAttr:$access_pattern_array,
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
  }];

  let parser = [{return ParseMapReduceOp(parser, result);}];
  let printer = [{Print(*this, p);}];
  let verifier = [{return Verify(*this);}];
}

class SairProjectionOp<string name> : SairOp<name, [
    AttrSizedOperandSegments,
    NoSideEffect,
    SairSameElementType<"value", "result">,
    SairValueProducerOp]> {
  let arguments = (ins
    Variadic<SairRange>:$parallel_domain,
    Variadic<SairRange>:$projection_domain,
    SairAccessPatternArrayAttr:$access_pattern_array,
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
    SairOneResult,
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
    SairAccessPatternArrayAttr:$access_pattern_array,
    SairValue:$init,
    SairValue:$value,
    SairMemorySpaceAttr:$memory_space
  );

  let results = (outs SairValue:$result);

  let parser = [{return ParseFbyOp(parser, result);}];
  let printer = [{Print(*this, p);}];

  let extraClassDeclaration = [{
    constexpr static ::llvm::StringRef kThenKeyword = "then";

    OperandRange domain() {
      return ChainOperandRanges(parallel_domain(), sequential_domain());
    }

    llvm::SmallBitVector DimsDependingOnOperand(int sair_operand);

    bool AllowUseBeforeDef(int index) { return index == 1; }

    llvm::SmallBitVector MustFuseDimensions(int sair_operand);
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

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"OpBuilder &builder, OperationState &result", [{
    Region *region = result.addRegion();
    SairProgramOp::ensureTerminator(*region, builder, result.location);}]>];
  let parser = [{return ParseProgramOp(parser, result);}];
  let printer = [{Print(*this, p);}];
  let verifier = [{return Verify(*this);}];

  let skipDefaultBuilders = 1;
  let builders = [
    // Builds a sair.program operation. Allocates the region its block.
    OpBuilder<[{mlir::OpBuilder &builder,
                mlir::OperationState &result,
                mlir::TypeRange result_types = {} }]>
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
    sair.exit (<ssa-value> <access-pattern>),* <attr-dict>
      (':' <element-type> (, <element-type)*)?
    ```
  }];

  let arguments = (ins
    SairAccessPatternArrayAttr:$access_pattern_array,
    Variadic<SairValue>:$inputs
  );

  let parser = [{return ParseExitOp(parser, result);}];
  let printer = [{Print(*this, p);}];
  let verifier = [{return Verify(*this);}];

  DerivedAttr shape = SairEmptyDomainShapeAttr;

  let builders = [
    OpBuilder<[{mlir::OpBuilder &builder,
                mlir::OperationState &result,
                mlir::ValueRange operands = {} }]>
  ];
}

#endif  // SAIR_SAIR_OPS_TD_
