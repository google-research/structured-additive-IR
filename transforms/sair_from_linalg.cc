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

#include "transforms/sair_from_linalg.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "sair_attributes.h"
#include "sair_ops.h"
#include "sair_types.h"
#include "storage.h"

namespace sair {
namespace {

// A bound for loops implied by Linalg operations used during conversion to
// Sair. In Linalg, there exist implicit loops that iterate over a particular
// dimension of a shaped MLIR value. The extent of this dimension may or may not
// be known statically.
struct LoopBound {
  // Value that the loop iterates over.
  mlir::Value referenced_value;

  // Dimension of a (multi-dimensional) value that the loop iterates over.
  int dimension;
};

// Obtains an upper bound for a loop iterating over one dimension of one of the
// shaped "operands". Interprets the dimensions of all operands as a single list
// and uses "map_position" to index that list. Expects all values in the range
// to be of shaped type and "map_position" to be in range.
LoopBound FindLoopBound(mlir::ValueRange operands, int map_position) {
  int num_seen_dimensions = 0;
  for (const mlir::Value &operand : operands) {
    auto type = operand.getType().cast<MemRefType>();
    int rank = type.getRank();
    int position_in_memref = map_position - num_seen_dimensions;
    if (position_in_memref < rank) {
      return {.referenced_value = operand, .dimension = position_in_memref};
    }
    num_seen_dimensions += rank;
  }

  llvm_unreachable("map_position out of bounds");
}

// Creates a Sair range corresponding to "bound". Uses "rewriter" to produce new
// operations and places them at "loc". The operations vary depending on the
// bound being statically known. For dynamic bounds, emit non-Sair arithemtic
// computation immediately before "sair_program".
mlir::Value CreateSairRange(mlir::Location loc, const LoopBound &bound,
                            SairProgramOp sair_program,
                            mlir::OpBuilder &rewriter) {
  mlir::MLIRContext *context = loc.getContext();
  auto domain_0d = DomainShapeAttr::HyperRectangular(context, /*rank=*/0);
  auto range_type = RangeType::get(domain_0d);
  auto shaped_type = bound.referenced_value.getType().cast<mlir::ShapedType>();
  int dimension = shaped_type.getDimSize(bound.dimension);

  // If the shape is statically known, create a simple static range.
  if (!mlir::ShapedType::isDynamic(dimension)) {
    return rewriter.create<SairStaticRangeOp>(
        loc, range_type, /*size=*/rewriter.getIndexAttr(dimension),
        /*step=*/rewriter.getIndexAttr(1));
  }

  // Otherwise, extract the dynamic dimension of the shaped type, construct a 0d
  // Sair value, and use this value to create a dependent range.
  auto mapping = MappingAttr::GetIdentity(context, /*num_dimensions=*/0);
  auto mapping_array = rewriter.getArrayAttr(mapping);
  auto value_type = ValueType::get(rewriter.getIndexType());

  // Create the IR obtaining the dimension of the memref outside the main Sair
  // program, since it is not allowed inside it. Temporarily switch the rewriter
  // insertion point for this reason.
  mlir::Value bound_dim = [&] {
    mlir::OpBuilder::InsertionGuard raii(rewriter);
    rewriter.setInsertionPoint(sair_program);
    return rewriter.create<mlir::memref::DimOp>(loc, bound.referenced_value,
                                                bound.dimension);
  }();
  mlir::Value bound_value =
      rewriter.create<SairFromScalarOp>(loc, value_type, bound_dim);
  return rewriter.create<SairDynRangeOp>(loc, range_type, mlir::ValueRange(),
                                         mapping_array,
                                         /*begin=*/nullptr,
                                         /*end=*/bound_value,
                                         /*step=*/rewriter.getIndexAttr(1));
}

// Extracts bounds of the loops comprised in the iteration domain from the list
// of "operands" with shaped type. The "subscripts_to_loops" affine map contains
// the mapping between shape dimensions and at least "num_loops" loops. The
// bounds are appended to "loop_bounds".
void CollectLoopBounds(int num_loops, mlir::AffineMap subscripts_to_loops,
                       mlir::ValueRange operands,
                       llvm::SmallVectorImpl<LoopBound> &loop_bounds) {
  loop_bounds.reserve(num_loops);
  for (int i = 0; i < num_loops; ++i) {
    auto expr = subscripts_to_loops.getResult(i).cast<mlir::AffineDimExpr>();
    LoopBound bound = FindLoopBound(operands, expr.getPosition());
    loop_bounds.push_back(bound);
  }
}

// Constructs a loop bounds list that corresponds to loops iterating over all
// dimensions of the given "shaped_value". Appends the bounds to "loop_bounds".
llvm::SmallVector<LoopBound, 4> LoopBoundsOnShapedType(
    mlir::Value shaped_value) {
  auto type = shaped_value.getType().cast<mlir::ShapedType>();
  int rank = type.getRank();

  llvm::SmallVector<LoopBound, 4> loop_bounds;
  loop_bounds.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    loop_bounds.push_back({.referenced_value = shaped_value, .dimension = i});
  }
  return loop_bounds;
}

// Creates Sair ranges to define a statically-shaped hyperrectangular domain
// with given "dimensions". Uses "rewriter" to produce new operations defining
// the range, and places them at location "loc". Populates "ranges" with values
// defined by the newly created operations. Non-Sair operations will be created
// before "sair_program".
llvm::SmallVector<mlir::Value, 4> CreateSairDomain(
    mlir::Location loc, llvm::ArrayRef<LoopBound> dimensions,
    SairProgramOp sair_program, mlir::OpBuilder &rewriter) {
  llvm::SmallVector<mlir::Value, 4> ranges;
  ranges.reserve(dimensions.size());
  for (const LoopBound &bound : dimensions) {
    ranges.push_back(CreateSairRange(loc, bound, sair_program, rewriter));
  }
  return ranges;
}

// Converts Linalg indexing maps into Sair mappings. Populates
// "operand_mappings" with the results, using the same array indices as
// "indexing_maps". Uses "sair_to_linalg_loops" to reshuffle the dimensions so
// that reduction loops always come last, as expected by Sair. This map is
// expected to be a bijective map between Sair loop order and Linalg loop order.
// Additionally, computes the mapping from value subscripts to surrounding loops
// and returns it in "subscripts_to_loops". If there is no subscript
// corresponding to a loop, return failure.
mlir::LogicalResult ConvertOperandMappings(
    mlir::ArrayAttr indexing_maps, mlir::AffineMap sair_to_linalg_loops,
    llvm::SmallVectorImpl<mlir::Attribute> &operand_mappings,
    mlir::AffineMap &subscripts_to_loops) {
  // Affine maps are straightforwardly converted to mappings. Also
  // accumulate the maps extracted from the attribute.
  int num_operands = indexing_maps.size();
  operand_mappings.reserve(num_operands);
  llvm::SmallVector<mlir::AffineMap, 4> loops_to_subscripts;
  loops_to_subscripts.reserve(num_operands);
  for (mlir::Attribute attr : indexing_maps.getValue()) {
    mlir::AffineMap indexing = attr.cast<AffineMapAttr>().getValue();
    indexing = indexing.compose(sair_to_linalg_loops);
    operand_mappings.push_back(MappingAttr::FromAffineMap(indexing));
    loops_to_subscripts.push_back(indexing);
  }

  // Concatenate all maps and try to invert them. The inversion only works for
  // permutation maps.
  mlir::AffineMap loops_to_all_subscripts =
      mlir::concatAffineMaps(loops_to_subscripts);
  if (loops_to_all_subscripts.isPermutation()) {
    return mlir::failure();
  }
  subscripts_to_loops = mlir::inversePermutation(loops_to_all_subscripts);
  return mlir::success();
}

// Converts Linalg indexing maps into Sair mappings suitable for casting
// Sair value back to memref used in Linalg. The iteration domain of this
// casting lives in the space of the memref, and the value lives in the loop
// space, so this needs to invert the indexing maps defined in "loop->memref"
// space.
mlir::LogicalResult ConvertResultMappings(
    llvm::ArrayRef<mlir::Attribute> indexing_maps,
    mlir::AffineMap parallel_to_positions,
    llvm::SmallVectorImpl<mlir::Attribute> &mappings) {
  // Invert indexing maps and transform them into mappings. If some maps
  // are not invertible, return failure.
  mappings.reserve(indexing_maps.size());
  for (mlir::Attribute attr : indexing_maps) {
    mlir::AffineMap indexing = attr.cast<AffineMapAttr>().getValue();
    indexing = indexing.compose(parallel_to_positions);
    mlir::AffineMap inverted = mlir::inversePermutation(indexing);
    if (!inverted) {
      return mlir::failure();
    }
    mappings.push_back(MappingAttr::FromAffineMap(inverted));
  }

  return mlir::success();
}

// Emits operations converting MemRef-typed "operands" to Sair values, using
// "rewriter" to create operations and positioning them at "loc". Stores the
// results in "map_operands". The last "num_outputs" operands are treated as
// in/out operands and their ranges are stored in "result_ranges" for further
// use, e.g. to convert them back to MemRefs. Non-Sair operations will be
// created before "sair_program".
void EmitMemRefToValue(
    mlir::ValueRange operands, int num_outputs, mlir::Location loc,
    SairProgramOp sair_program, StorageAnalysis &storage_analysis,
    mlir::OpBuilder &rewriter, llvm::SmallVectorImpl<mlir::Value> &map_operands,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value, 4>> &result_ranges) {
  mlir::MLIRContext *context = loc.getContext();
  int num_operands = operands.size();
  int num_inputs = num_operands - num_outputs;
  map_operands.reserve(num_operands);
  result_ranges.reserve(num_outputs);

  // For each operand, construct a Sair values with hyper-rectangular domain
  // with static dimensions obtained from MemRef shape.
  for (auto en : llvm::enumerate(operands)) {
    int position = en.index();
    mlir::Value operand = en.value();
    auto type = operand.getType().cast<mlir::ShapedType>();

    llvm::SmallVector<LoopBound, 4> bounds = LoopBoundsOnShapedType(operand);
    llvm::SmallVector<mlir::Value, 4> ranges =
        CreateSairDomain(loc, bounds, sair_program, rewriter);

    auto domain_shape =
        DomainShapeAttr::HyperRectangular(context, type.getRank());
    auto value_type = ValueType::get(domain_shape, type.getElementType());
    auto mappings = rewriter.getArrayAttr(
        {MappingAttr::GetIdentity(context, 0, type.getRank())});
    auto memref_value_type =
        ValueType::get(DomainShapeAttr::get(context), type);

    auto from_scalar =
        rewriter.create<SairFromScalarOp>(loc, memref_value_type, operand);
    Value new_operand = rewriter.create<SairFromMemRefOp>(
        loc, value_type, mlir::ValueRange(), ranges, mappings, from_scalar,
        storage_analysis.GetFreshBufferName());
    // Insert a copy to avoid storage specification mismatch.
    // TODO(b/181850491): introduce a sair.maybe_copy operation instead.
    auto copy_mapping = rewriter.getArrayAttr(
        {MappingAttr::GetIdentity(context, ranges.size())});
    Value copied_operand = rewriter.create<SairCopyOp>(
        loc, value_type, ranges, copy_mapping, new_operand,
        /*loop_nest=*/nullptr, /*storage=*/nullptr);
    map_operands.push_back(copied_operand);

    // For in/out operands, store the ranges.
    if (position >= num_inputs) {
      result_ranges.emplace_back(std::move(ranges));
    }
  }
}

// Emits operations storing the content of "sair_values" to the given memrefs,
// using "rewriter" to create the operations and positioning them at "loc".
// Values are indexed with "mappings" and have the provided "ranges".
// Expects equal number of Sair values, memrefs, mappings and ranges
// to be passed.
void EmitValueToMemRef(mlir::Location loc, SairProgramOp program,
                       mlir::ValueRange sair_values, mlir::ValueRange memrefs,
                       llvm::ArrayRef<mlir::Attribute> mappings,
                       llvm::ArrayRef<llvm::SmallVector<mlir::Value, 4>> ranges,
                       StorageAnalysis &storage_analysis,
                       mlir::OpBuilder &rewriter) {
  assert(sair_values.size() == memrefs.size());
  assert(sair_values.size() == mappings.size());
  assert(sair_values.size() == ranges.size());

  mlir::MLIRContext *context = loc.getContext();
  int num_results = sair_values.size();
  for (int i = 0; i < num_results; ++i) {
    auto mapping_array = ArrayAttr::get(
        context,
        {MappingAttr::GetIdentity(context, 0, ranges[i].size()), mappings[i]});
    auto shape = DomainShapeAttr::HyperRectangular(context, ranges[i].size());
    auto memref_value_type =
        ValueType::get(DomainShapeAttr::get(context), memrefs[i].getType());
    mlir::Value from_scalar;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&program.body().front());
      from_scalar =
          rewriter.create<SairFromScalarOp>(loc, memref_value_type, memrefs[i]);
    }
    rewriter.create<SairToMemRefOp>(
        loc, mlir::ValueRange(), ranges[i], mapping_array, from_scalar,
        sair_values[i], shape, storage_analysis.GetFreshBufferName());
  }
}

// Returns true if "elements" contains unique values within the [0, N-1] range
// where N is the size of "elements". Such an array can be used as permutation
// map.
bool IsPermutation(llvm::ArrayRef<int> elements) {
  llvm::SmallDenseSet<int> seen;
  int size = elements.size();
  for (int e : elements) {
    if (e < 0 || e >= size || seen.count(e)) {
      return false;
    }
    seen.insert(e);
  }
  return true;
}

// Permutes the arguments of "block" as indicated in "new_positions". This
// permutation table is expected to contain unique values within the [0, N-1]
// range where N is the number of arguments to permute. i-th value in this table
// defines the new position of the i-th block argument. The optional "offset"
// indicates the position of the first argument to be permuted the list of block
// arguments. When the offset is non-zero, the zero-based indices in the
// permutation table "new_positions" are interpreted relatively to "offset". The
// block is expected to have a sufficient number of arguments.
void PermuteBlockArguments(llvm::ArrayRef<int> new_positions, int offset,
                           mlir::Block &block) {
  int num_permuted_args = new_positions.size();
  assert(offset + num_permuted_args <= block.getNumArguments());
  assert(IsPermutation(new_positions));

  llvm::SmallVector<mlir::Value, 8> new_arguments(num_permuted_args);
  for (int i = 0; i < num_permuted_args; ++i) {
    mlir::Value old_argument = block.getArgument(offset + i);
    new_arguments[new_positions[i]] = block.insertArgument(
        offset + num_permuted_args + i, old_argument.getType());
  }

  mlir::ValueRange old_arguments =
      block.getArguments().slice(offset, num_permuted_args);
  for (auto pair : llvm::zip(old_arguments, new_arguments)) {
    mlir::Value old_arg = std::get<0>(pair);
    mlir::Value new_arg = std::get<1>(pair);
    old_arg.replaceAllUsesWith(new_arg);
  }

  for (int i = 0; i < num_permuted_args; ++i) {
    block.eraseArgument(offset);
  }
}

// Sets up "permutation" to be a permutation table placing the segment of
// "num_first" elements after a segment of "num_second" elements. The
// permutation table contains, for i-th element, its new position. For example,
// the segment permutation with segments of 2 and 3 will produce the table
//   [3 4 0 1 2]
// which, when applied, will permute the table
//   [A B C D E] into [C D E A B].
void SegmentPermutation(int num_first, int num_second,
                        llvm::SmallVectorImpl<int> &permutation) {
  permutation.reserve(num_first + num_second);
  for (int i = 0; i < num_first; ++i) {
    permutation.push_back(num_second + i);
  }
  for (int i = 0; i < num_second; ++i) {
    permutation.push_back(i);
  }
}

// Moves the body region of "source_op" into "target_region" using "rewriter" to
// keep track of changes. The region is cleared of its existing content.
// TODO(zinenko): use PatternRewriter infrastructure when it supports changes to
// Block arguments.
void MoveBodyBlock(mlir::AffineMap linalg_to_sair_loops,
                   mlir::OpBuilder &rewriter, mlir::Region &target_region,
                   mlir::linalg::LinalgOp source_op) {
  assert(isa<mlir::linalg::GenericOp>(source_op.getOperation()) ||
         isa<mlir::linalg::IndexedGenericOp>(source_op.getOperation()));

  mlir::Region &source_region = source_op.getOperation()->getRegion(0);
  target_region.getBlocks().clear();
  target_region.getBlocks().splice(target_region.begin(),
                                   source_region.getBlocks());
  mlir::Block &body = target_region.front();

  // Move block arguments that correspond to input-only operands to the end
  // of the list to comply with Sair order if "map_reduce" is generated. Since
  // we don't have access to the underlying argument storage, simply recreate
  // the arguments.
  bool has_indices =
      isa<mlir::linalg::IndexedGenericOp>(source_op.getOperation());
  bool has_reductions = source_op.getNumReductionLoops() != 0;
  if (has_reductions) {
    int first_value_arg = has_indices ? source_op.getNumLoops() : 0;
    llvm::SmallVector<int, 8> permutation;
    SegmentPermutation(source_op.getNumInputs(), source_op.getNumOutputs(),
                       permutation);
    PermuteBlockArguments(permutation, first_value_arg, body);

    // Permute indices to put those related to reductions last.
    if (has_indices) {
      llvm::SmallVector<int, 4> index_permutation;
      index_permutation.reserve(linalg_to_sair_loops.getNumResults());
      MappingAttr linalg_to_sair_loops_mapping =
          MappingAttr::FromAffineMap(linalg_to_sair_loops);
      for (MappingExpr expr : linalg_to_sair_loops_mapping) {
        index_permutation.push_back(expr.cast<MappingDimExpr>().dimension());
      }
      PermuteBlockArguments(index_permutation, /*offset=*/0, body);
    }
  }

  // Insert arguments for iteration indices if they are not already present.
  if (!has_indices) {
    int num_loops = source_op.getNumLoops();
    for (int i = 0; i < num_loops; ++i) {
      body.insertArgument(body.args_begin(), rewriter.getIndexType());
    }
  }

  // Replace the linalg.yield terminator with sair.return.
  {
    OpBuilder::InsertionGuard raii(rewriter);
    rewriter.setInsertionPointToEnd(&body);
    rewriter.create<SairReturnOp>(source_op.getLoc(),
                                  body.getTerminator()->getOperands());
  }
  body.back().getPrevNode()->erase();
}

// Populates "result_types" with Sair value types having the same elemental type
// as "types" and a hyper-rectangular domain with the given number of dimensons.
// Uses "rewriter" to construct the types.
void CreateResultTypes(mlir::Builder &rewriter, int num_dimensions,
                       const SmallVectorImpl<MemRefType> &types,
                       llvm::SmallVectorImpl<mlir::Type> &result_types) {
  mlir::MLIRContext *context = rewriter.getContext();
  auto result_domain_shape =
      DomainShapeAttr::HyperRectangular(context, num_dimensions);
  int num_results = types.size();
  result_types.reserve(num_results);
  for (Type type : types) {
    mlir::Type element_type = type.cast<mlir::ShapedType>().getElementType();
    result_types.push_back(ValueType::get(result_domain_shape, element_type));
  }
}

// Creates two affine map representing the loop permutations between Linalg and
// Sair implicit loop nesting structures. The first map, "linalg_to_sair_loops",
// maps inputs corresponding to Linalg reduction loops at the trailing positions
// of the map, preserving their relative order. The second map,
// "parallel_loop_positions", maps consecutive dimensions to positions of
// parallel loops in Linalg notation and assigns "-1" to other kinds of loops.
// Expects "attr" to be an array attribute containing string attributes with one
// of two values corresponding to parallel or reduciton Linalg dimensions.
void ComputePermutationMaps(mlir::ArrayAttr attr,
                            mlir::AffineMap &linalg_to_sair_loops,
                            mlir::AffineMap &parallel_loop_positions) {
  mlir::MLIRContext *context = attr.getContext();
  llvm::SmallVector<mlir::AffineExpr, 8> parallel_dimensions;
  llvm::SmallVector<mlir::AffineExpr, 4> reduction_dimensions;

  llvm::SmallVector<mlir::AffineExpr, 8> inverse_dimensions;
  auto ignore_this_dimension_expr = mlir::getAffineConstantExpr(-1, context);
  int num_visited_parallel_dims = 0;

  inverse_dimensions.reserve(attr.size());
  for (const auto &en : llvm::enumerate(attr)) {
    mlir::Attribute dim = en.value();
    mlir::AffineExpr expr = mlir::getAffineDimExpr(en.index(), context);
    llvm::StringRef type = dim.cast<StringAttr>().getValue();
    if (type == mlir::getParallelIteratorTypeName()) {
      parallel_dimensions.push_back(expr);
      mlir::AffineExpr inverse_expr =
          mlir::getAffineDimExpr(num_visited_parallel_dims++, context);
      inverse_dimensions.push_back(inverse_expr);
    } else if (type == mlir::getReductionIteratorTypeName()) {
      reduction_dimensions.push_back(expr);
      inverse_dimensions.push_back(ignore_this_dimension_expr);
    } else {
      llvm_unreachable("unknown iterator type");
    }
  }

  parallel_loop_positions =
      mlir::AffineMap::get(num_visited_parallel_dims, /*symbolCount=*/0,
                           inverse_dimensions, context);
  llvm::append_range(parallel_dimensions, reduction_dimensions);
  linalg_to_sair_loops = mlir::AffineMap::get(attr.size(), /*symbolCount=*/0,
                                              parallel_dimensions, context);
}

// Creates a Sair "map_reduce" operation from the partial results of Linalg to
// Sair conversion. The operation is created at "loc" using "rewriter". The
// "domain" is a list of ranges for parallel dimensions, immediately followed by
// "num_reduction_loops" reduction dimensions. The list of "linalg_operands"
// contains Linalg Op operands converted to Sair values in the same order as the
// Linalg operation. That is, leading operands are input-only while
// "num_outputs" trailing operands are input/output. The latter are used as
// initial values for Sair reductions.
mlir::Operation *CreateMapReduceOp(
    mlir::Location loc, llvm::ArrayRef<mlir::Type> result_types,
    mlir::ValueRange domain, mlir::ValueRange linalg_operands,
    llvm::ArrayRef<mlir::Attribute> operand_mappings,
    DomainShapeAttr domain_shape, int num_reduction_loops, int num_outputs,
    mlir::OpBuilder &rewriter) {
  // Split domain and operand lists into reduction and parallel parts.
  mlir::ValueRange parallel_domain = domain.drop_back(num_reduction_loops);
  mlir::ValueRange reduction_domain = domain.take_back(num_reduction_loops);
  mlir::ValueRange init_operands = linalg_operands.take_back(num_outputs);
  mlir::ValueRange input_operands = linalg_operands.drop_back(num_outputs);

  // Reorder mappings to match the expected order in Sair "map_reduce".
  llvm::SmallVector<mlir::Attribute, 8> mappings;
  mappings.reserve(operand_mappings.size());
  llvm::append_range(mappings, operand_mappings.drop_front(num_outputs));
  llvm::append_range(mappings, operand_mappings.take_front(num_outputs));
  mlir::ArrayAttr mappings_attr = rewriter.getArrayAttr(mappings);

  return rewriter.create<SairMapReduceOp>(
      loc, result_types, parallel_domain, reduction_domain, mappings_attr,
      init_operands, input_operands, domain_shape,
      /*loop_nest=*/nullptr, /*memory_space=*/nullptr);
}

// Rewrites Linalg generic operation into a semantically equivalent sequence of
// Sair operations. This sequence contains conversions between MemRefs and Sair
// values and a Sair map or map_reduce operation.
mlir::LogicalResult RewriteLinalgToSair(mlir::linalg::LinalgOp op,
                                        mlir::OpBuilder &rewriter) {
  // Only support Linalg on memrefs.
  if (!op.hasBufferSemantics() || op.getNumWindowLoops() != 0) {
    return mlir::failure();
  }

  // Linalg operations with outlined body are not supported.
  mlir::Operation *operation = op.getOperation();
  if (operation->getNumRegions() != 1 || operation->getRegion(0).empty()) {
    return mlir::failure();
  }

  mlir::Location loc = op.getLoc();

  // Compute the mappings between Linalg and Sair implicit loops. Sair has a
  // convention that reduction loops always come last.
  mlir::AffineMap parallel_to_positions;
  mlir::AffineMap linalg_to_sair_loops;
  ComputePermutationMaps(op.iterator_types(), linalg_to_sair_loops,
                         parallel_to_positions);
  mlir::AffineMap sair_to_linalg_loops =
      mlir::inversePermutation(linalg_to_sair_loops);

  // Convert Linalg indexing maps to Sair mappings and keep track of the
  // mapping between value access subscripts and iteration domain dimensions.
  llvm::SmallVector<mlir::Attribute, 4> operand_mappings;
  mlir::AffineMap subscripts_to_loops;
  if (mlir::failed(
          ConvertOperandMappings(op.indexing_maps(), sair_to_linalg_loops,
                                 operand_mappings, subscripts_to_loops))) {
    return mlir::failure();
  }

  // Linalg does not seem to restrict the output indexing to parallel dimensions
  // only, but Sair does. Abort the conversion in case of incompatibility.
  int num_parallel_loops = op.getNumParallelLoops();
  int num_operands = op.getNumShapedOperands();
  for (int i = op.getNumInputs(); i < num_operands; ++i) {
    auto mapping = operand_mappings[i].cast<MappingAttr>();
    if (mlir::failed(VerifyReductionMapping(mapping, num_parallel_loops))) {
      return mlir::failure();
    }
  }

  // Convert Linalg indexing maps to Sair mappings usable in "to_memref".  Some
  // mappings may not be convertible, so return before we start constructing any
  // new IR.
  llvm::SmallVector<mlir::Attribute, 4> result_mappings;
  llvm::ArrayRef<mlir::Attribute> all_indexing_maps =
      op.indexing_maps().getValue();
  int num_outputs = op.getNumOutputs();
  if (mlir::failed(
          ConvertResultMappings(all_indexing_maps.take_back(num_outputs),
                                parallel_to_positions, result_mappings))) {
    return mlir::failure();
  }

  auto sair_program = rewriter.create<SairProgramOp>(loc);
  rewriter.setInsertionPointToStart(&sair_program.body().front());
  StorageAnalysis storage_analysis(sair_program);

  // Convert input and input/output MemRefs used by Linalg to Sair values.
  llvm::SmallVector<mlir::Value, 4> map_operands;
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> result_ranges;
  EmitMemRefToValue(op.getShapedOperands(), op.getNumOutputs(), loc,
                    sair_program, storage_analysis, rewriter, map_operands,
                    result_ranges);

  // Prepare parameters of the Sair map operation.
  int num_loops = op.getNumLoops();
  llvm::SmallVector<LoopBound, 8> loop_bounds;
  CollectLoopBounds(num_loops, subscripts_to_loops, op.getShapedOperands(),
                    loop_bounds);
  llvm::SmallVector<mlir::Value, 4> domain_ranges =
      CreateSairDomain(loc, loop_bounds, sair_program, rewriter);

  llvm::SmallVector<mlir::Type, 4> result_types;
  CreateResultTypes(rewriter, num_parallel_loops, op.getOutputBufferTypes(),
                    result_types);

  // Construct the main map or map_reduce operation.
  mlir::Operation *map_op;
  int num_reduction_dims = op.getNumReductionLoops();
  DomainShapeAttr domain_shape =
      DomainShapeAttr::HyperRectangular(rewriter.getContext(), num_loops);
  if (num_reduction_dims == 0) {
    map_op = rewriter.create<SairMapOp>(
        loc, result_types, domain_ranges,
        rewriter.getArrayAttr(operand_mappings), map_operands, domain_shape,
        /*loop_nest=*/nullptr, /*memory_space=*/nullptr);
  } else {
    map_op = CreateMapReduceOp(
        loc, result_types, domain_ranges, map_operands, operand_mappings,
        domain_shape, num_reduction_dims, op.getNumOutputs(), rewriter);
  }
  MoveBodyBlock(linalg_to_sair_loops, rewriter, map_op->getRegion(0), op);

  // Convert output values to input/output MemRefs used by Linalg.
  EmitValueToMemRef(loc, sair_program, map_op->getResults(),
                    op.getOutputBuffers(), result_mappings, result_ranges,
                    storage_analysis, rewriter);

  // Add the sair.program terminator.
  rewriter.create<SairExitOp>(loc);

  // Delete the source operation after conversion.
  op.erase();
  return mlir::success();
}

#define GEN_PASS_CLASSES
#include "transforms/sair_from_linalg.h.inc"

// A pass converting Linalg (indexed) generic operations to Sair equivalents in
// the given function.
class LinalgToSairConversion
    : public SairFromLinalgPassBase<LinalgToSairConversion> {
 public:
  // Runs the pass on a function.
  void runOnFunction() override;
};

void LinalgToSairConversion::runOnFunction() {
  mlir::MLIRContext *context = &getContext();

  // Replace all suitable Linalg generic operations in a function.
  getFunction().walk([context, this](mlir::linalg::LinalgOp op) {
    mlir::OpBuilder builder(context);
    builder.setInsertionPoint(op);
    if (mlir::failed(sair::RewriteLinalgToSair(op, builder))) {
      mlir::emitError(op.getLoc()) << "Linalg op is not compatible with Sair";
      signalPassFailure();
    }
  });
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateLinalgToSairConversionPass() {
  return std::make_unique<LinalgToSairConversion>();
}
}  // namespace sair
