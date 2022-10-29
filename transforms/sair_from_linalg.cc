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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
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
  auto domain_0d = DomainShapeAttr::get(context);
  auto shaped_type = bound.referenced_value.getType().cast<mlir::ShapedType>();
  int dimension = shaped_type.getDimSize(bound.dimension);

  // If the shape is statically known, create a simple static range.
  if (!mlir::ShapedType::isDynamic(dimension)) {
    auto range_type = StaticRangeType::get(dimension, 1, context);
    return rewriter.create<SairStaticRangeOp>(loc, range_type,
                                              /*instances=*/nullptr);
  }

  // Otherwise, extract the dynamic dimension of the shaped type, construct a 0d
  // Sair value, and use this value to create a dependent range.
  auto range_type = DynRangeType::get(domain_0d);
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
                                         /*step=*/rewriter.getIndexAttr(1),
                                         /*instances=*/nullptr);
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
// the range, and places them at location "loc". Populates "sair_dimensions"
// with values defined by the newly created operations and
// "sair_dimension_shapes" with the dimensions of the sair domain shape
// attribute.  Non-Sair operations will be created before "sair_program".
void CreateSairDomain(mlir::Location loc, llvm::ArrayRef<LoopBound> dimensions,
                      SairProgramOp sair_program,
                      llvm::SmallVector<mlir::Value> &sair_dimensions,
                      llvm::SmallVector<DomainShapeDim> &sair_dimension_shapes,
                      mlir::OpBuilder &rewriter) {
  mlir::MLIRContext *context = sair_program.getContext();

  sair_dimensions.reserve(dimensions.size());
  sair_dimension_shapes.reserve(dimensions.size());
  for (const LoopBound &bound : dimensions) {
    sair_dimensions.push_back(
        CreateSairRange(loc, bound, sair_program, rewriter));
    auto mapping = MappingAttr::get(context, sair_dimension_shapes.size(), {});
    auto type = sair_dimensions.back().getType().cast<DimensionType>();
    sair_dimension_shapes.emplace_back(type, mapping);
  }
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
    llvm::SmallVector<mlir::Value> ranges;
    llvm::SmallVector<DomainShapeDim> shape_dims;
    CreateSairDomain(loc, bounds, sair_program, ranges, shape_dims, rewriter);
    auto domain_shape = DomainShapeAttr::get(context, shape_dims);
    auto value_type = ValueType::get(domain_shape, type.getElementType());
    auto mappings = rewriter.getArrayAttr(
        {MappingAttr::GetIdentity(context, 0, type.getRank())});
    auto memref_value_type =
        ValueType::get(DomainShapeAttr::get(context), type);

    auto from_scalar =
        rewriter.create<SairFromScalarOp>(loc, memref_value_type, operand);
    Value new_operand = rewriter.create<SairFromMemRefOp>(
        loc, value_type, mlir::ValueRange(), ranges, mappings, from_scalar,
        storage_analysis.GetFreshBufferName(), /*instances=*/nullptr,
        /*copies=*/nullptr);
    // Insert a copy to avoid storage specification mismatch.
    // TODO(b/181850491): introduce a sair.maybe_copy operation instead.
    auto copy_mapping = rewriter.getArrayAttr(
        {MappingAttr::GetIdentity(context, ranges.size())});
    Value copied_operand = rewriter.create<SairCopyOp>(
        loc, value_type, ranges, copy_mapping, new_operand,
        /*decisions=*/nullptr, /*copies=*/nullptr);
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
    auto value_type = sair_values[i].getType().cast<ValueType>();
    auto shape = value_type.Shape().AccessedShape(
        mappings[i].cast<MappingAttr>().Inverse());
    auto memref_value_type =
        ValueType::get(DomainShapeAttr::get(context), memrefs[i].getType());
    mlir::Value from_scalar;
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&program.getBody().front());
      from_scalar =
          rewriter.create<SairFromScalarOp>(loc, memref_value_type, memrefs[i]);
    }
    rewriter.create<SairToMemRefOp>(
        loc, mlir::ValueRange(), ranges[i], mapping_array, from_scalar,
        sair_values[i], shape, storage_analysis.GetFreshBufferName(),
        /*instances=*/nullptr,
        /*copies=*/nullptr);
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
    new_arguments[new_positions[i]] =
        block.insertArgument(offset + num_permuted_args + i,
                             old_argument.getType(), old_argument.getLoc());
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
void MoveBodyBlock(mlir::AffineMap linalg_to_sair_loops,
                   mlir::OpBuilder &rewriter, mlir::Region &target_region,
                   mlir::linalg::LinalgOp source_op) {
  assert(isa<mlir::linalg::GenericOp>(source_op.getOperation()));

  mlir::Region &source_region = source_op.getOperation()->getRegion(0);
  target_region.getBlocks().clear();
  target_region.getBlocks().splice(target_region.begin(),
                                   source_region.getBlocks());
  mlir::Block &body = target_region.front();

  // Move block arguments that correspond to input-only operands to the end
  // of the list to comply with Sair order if "map_reduce" is generated. Since
  // we don't have access to the underlying argument storage, simply recreate
  // the arguments.
  bool has_reductions = source_op.getNumReductionLoops() != 0;
  if (has_reductions) {
    llvm::SmallVector<int, 8> permutation;
    SegmentPermutation(source_op.getNumDpsInputs(), source_op.getNumDpsInits(),
                       permutation);
    PermuteBlockArguments(permutation, 0, body);
  }

  // Insert arguments for iteration indices.
  int num_loops = source_op.getNumLoops();
  for (int i = 0; i < num_loops; ++i) {
    body.insertArgument(body.args_begin(), rewriter.getIndexType(),
                        source_op.getLoc());
  }

  // Replace index operations with index values coming from block arguments.
  body.walk([&](mlir::linalg::IndexOp index_op) {
    Value index = body.getArgument(index_op.getDim());
    index_op.replaceAllUsesWith(index);
    index_op.erase();
  });

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
// as "types" and the given shape. Uses "rewriter" to construct the types.
void CreateResultTypes(mlir::Builder &rewriter, DomainShapeAttr shape,
                       const SmallVectorImpl<MemRefType> &types,
                       llvm::SmallVectorImpl<mlir::Type> &result_types) {
  int num_results = types.size();
  result_types.reserve(num_results);
  for (Type type : types) {
    mlir::Type element_type = type.cast<mlir::ShapedType>().getElementType();
    result_types.push_back(ValueType::get(shape, element_type));
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
void ComputePermutationMaps(mlir::MLIRContext *context,
                            llvm::ArrayRef<llvm::StringRef> iterator_types,
                            mlir::AffineMap &linalg_to_sair_loops,
                            mlir::AffineMap &parallel_loop_positions) {
  llvm::SmallVector<mlir::AffineExpr, 8> parallel_dimensions;
  llvm::SmallVector<mlir::AffineExpr, 4> reduction_dimensions;

  llvm::SmallVector<mlir::AffineExpr, 8> inverse_dimensions;
  auto ignore_this_dimension_expr = mlir::getAffineConstantExpr(-1, context);
  int num_visited_parallel_dims = 0;

  inverse_dimensions.reserve(iterator_types.size());
  for (const auto &en : llvm::enumerate(iterator_types)) {
    llvm::StringRef dim = en.value();
    mlir::AffineExpr expr = mlir::getAffineDimExpr(en.index(), context);
    if (mlir::linalg::isParallelIterator(dim)) {
      parallel_dimensions.push_back(expr);
      mlir::AffineExpr inverse_expr =
          mlir::getAffineDimExpr(num_visited_parallel_dims++, context);
      inverse_dimensions.push_back(inverse_expr);
    } else if (mlir::linalg::isReductionIterator(dim)) {
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
  linalg_to_sair_loops = mlir::AffineMap::get(
      iterator_types.size(), /*symbolCount=*/0, parallel_dimensions, context);
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
      /*decisions=*/nullptr, /*copies=*/nullptr);
}

// Rewrites Linalg generic operation into a semantically equivalent sequence of
// Sair operations. This sequence contains conversions between MemRefs and Sair
// values and a Sair map or map_reduce operation.
mlir::LogicalResult RewriteLinalgToSair(mlir::linalg::LinalgOp op,
                                        mlir::OpBuilder &rewriter) {
  mlir::MLIRContext *context = op.getContext();
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
  ComputePermutationMaps(op.getContext(), op.getIteratorTypesArray(),
                         linalg_to_sair_loops, parallel_to_positions);
  mlir::AffineMap sair_to_linalg_loops =
      mlir::inversePermutation(linalg_to_sair_loops);

  // Convert Linalg indexing maps to Sair mappings and keep track of the
  // mapping between value access subscripts and iteration domain dimensions.
  llvm::SmallVector<mlir::Attribute, 4> operand_mappings;
  mlir::AffineMap subscripts_to_loops;
  if (mlir::failed(
          ConvertOperandMappings(op.getIndexingMaps(), sair_to_linalg_loops,
                                 operand_mappings, subscripts_to_loops))) {
    return mlir::failure();
  }

  // Linalg does not seem to restrict the output indexing to parallel dimensions
  // only, but Sair does. Abort the conversion in case of incompatibility.
  int num_parallel_loops = op.getNumParallelLoops();
  int num_operands = op->getNumOperands();
  for (int i = op.getNumDpsInputs(); i < num_operands; ++i) {
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
      op.getIndexingMaps().getValue();
  int num_outputs = op.getNumDpsInits();
  if (mlir::failed(
          ConvertResultMappings(all_indexing_maps.take_back(num_outputs),
                                parallel_to_positions, result_mappings))) {
    return mlir::failure();
  }

  auto sair_program = rewriter.create<SairProgramOp>(loc);
  rewriter.setInsertionPointToStart(&sair_program.getBody().front());
  StorageAnalysis storage_analysis(sair_program);

  // Convert input and input/output MemRefs used by Linalg to Sair values.
  llvm::SmallVector<mlir::Value, 4> map_operands;
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> result_ranges;
  llvm::SmallVector<mlir::Value> operands = op->getOperands();
  EmitMemRefToValue(operands, op.getNumDpsInits(), loc, sair_program,
                    storage_analysis, rewriter, map_operands, result_ranges);

  // Prepare parameters of the Sair map operation.
  int num_loops = op.getNumLoops();
  llvm::SmallVector<LoopBound, 8> loop_bounds;
  CollectLoopBounds(num_loops, subscripts_to_loops, operands, loop_bounds);
  llvm::SmallVector<mlir::Value> domain_ranges;
  llvm::SmallVector<DomainShapeDim> shape_dims;
  CreateSairDomain(loc, loop_bounds, sair_program, domain_ranges, shape_dims,
                   rewriter);

  llvm::SmallVector<mlir::Type, 4> result_types;
  int num_reduction_dims = op.getNumReductionLoops();
  DomainShapeAttr domain_shape = DomainShapeAttr::get(context, shape_dims);
  auto result_shape =
      domain_shape.Prefix(domain_shape.NumDimensions() - num_reduction_dims);
  SmallVector<MemRefType> outputBufferTypes;
  for (OpOperand *outputOperand : op.getDpsInitOperands())
    outputBufferTypes.push_back(
        outputOperand->get().getType().cast<MemRefType>());
  CreateResultTypes(rewriter, result_shape, outputBufferTypes, result_types);

  // Check that all operands shapes match.
  for (auto [value, mapping] : llvm::zip(map_operands, operand_mappings)) {
    DomainShapeAttr shape = value.getType().cast<ValueType>().Shape();
    if (domain_shape.AccessedShape(mapping.cast<MappingAttr>()) != shape) {
      return mlir::failure();
    }
  }

  // Construct the main map or map_reduce operation.
  mlir::Operation *map_op;
  if (num_reduction_dims == 0) {
    map_op = rewriter.create<SairMapOp>(loc, result_types, domain_ranges,
                                        rewriter.getArrayAttr(operand_mappings),
                                        map_operands, domain_shape,
                                        /*decisions=*/nullptr,
                                        /*copies=*/nullptr);
  } else {
    map_op = CreateMapReduceOp(
        loc, result_types, domain_ranges, map_operands, operand_mappings,
        domain_shape, num_reduction_dims, op.getNumDpsInits(), rewriter);
  }
  MoveBodyBlock(linalg_to_sair_loops, rewriter, map_op->getRegion(0), op);

  // Convert output values to input/output MemRefs used by Linalg.
  llvm::SmallVector<mlir::Value> output_buffers = op.getDpsInitOperands();
  EmitValueToMemRef(loc, sair_program, map_op->getResults(), output_buffers,
                    result_mappings, result_ranges, storage_analysis, rewriter);

  // Add the sair.program terminator.
  rewriter.create<SairExitOp>(loc);

  // Delete the source operation after conversion.
  op.erase();
  return mlir::success();
}

#define GEN_PASS_DEF_SAIRFROMLINALGPASS
#include "transforms/sair_from_linalg.h.inc"

// A pass converting Linalg (indexed) generic operations to Sair equivalents in
// the given function.
class LinalgToSairConversion
    : public impl::SairFromLinalgPassBase<LinalgToSairConversion> {
 public:
  // Runs the pass on a function.
  void runOnOperation() override;
};

void LinalgToSairConversion::runOnOperation() {
  mlir::MLIRContext *context = &getContext();

  // Replace all suitable Linalg generic operations in a function.
  getOperation().walk([context, this](mlir::linalg::LinalgOp op) {
    mlir::OpBuilder builder(context);
    builder.setInsertionPoint(op);
    if (mlir::failed(sair::RewriteLinalgToSair(op, builder))) {
      mlir::emitError(op.getLoc()) << "Linalg op is not compatible with Sair";
      signalPassFailure();
    }
  });
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLinalgToSairConversionPass() {
  return std::make_unique<LinalgToSairConversion>();
}
}  // namespace sair
