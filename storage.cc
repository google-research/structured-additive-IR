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

#include "storage.h"

#include "llvm/ADT/SmallString.h"
#include "loop_nest.h"

namespace sair {

// Sort value view so that all predecesors of a value view are visited before
// the view is visited, except in presence of cycles. This is achieved by doing
// a reverse post-order traversal.
static llvm::SmallVector<ValueViewOp, 4> SortValueViews(SairProgramOp program) {
  llvm::SmallVector<ValueViewOp, 4> sorted;
  llvm::DenseSet<mlir::Operation *> visited;

  program.walk([&](ValueViewOp op) {
    llvm::SmallVector<ValueViewOp, 4> queue;
    queue.push_back(op);

    while (!queue.empty()) {
      ValueViewOp current = queue.back();
      for (mlir::Value result : current->getResults()) {
        for (mlir::Operation *user : result.getUsers()) {
          auto user_view = dyn_cast<ValueViewOp>(user);
          if (user_view == nullptr || visited.count(user) > 0) continue;
          queue.push_back(user_view);
          visited.insert(user);
        }
      }

      // Add to sorted and pop after all children are visited.
      if (queue.back() != current) continue;
      queue.pop_back();
      sorted.push_back(current);
    }
  });

  std::reverse(sorted.begin(), sorted.end());
  return sorted;
}

Buffer::Buffer(mlir::Type element_type, int rank, ComputeOp op, int result)
    : loc_(op.getLoc()), element_type_(element_type) {
  assert(element_type != nullptr);
  assert(rank >= 0);

  llvm::ArrayRef<mlir::Attribute> loop_nest = op.LoopNestLoops();
  loop_nest_.reserve(loop_nest.size());
  for (mlir::Attribute attr : loop_nest) {
    auto loop = attr.cast<LoopAttr>();
    loop_nest_.push_back(loop.name());
  }

  auto none_expr = MappingNoneExpr::get(element_type.getContext());
  layout_.resize(rank, none_expr);
  writes_.emplace_back(op, result);
}

void Buffer::TrimLoopNest(int new_size,
                          const LoopFusionAnalysis &fusion_analysis) {
  assert(new_size <= loop_nest_.size());
  loop_nest_.resize(new_size);
  if (domain_.empty()) return;

  mlir::MLIRContext *context = element_type_.getContext();
  // Compute dimensions used by layout.
  llvm::SmallBitVector used_dimensions(domain_.size());
  for (MappingExpr layout_expr : layout_) {
    layout_expr.SetDependenciesInMask(used_dimensions);
  }

  // Trim domain from unused dimensions.
  llvm::SmallVector<ValueAccess> old_domain;
  std::swap(old_domain, domain_);
  llvm::append_range(domain_, fusion_analysis.GetLoopNest(loop_nest_).domain);
  llvm::SmallVector<MappingExpr> renaming(old_domain.size(),
                                          MappingNoneExpr::get(context));

  for (int dim : used_dimensions.set_bits()) {
    // Already added to the new domain.
    if (renaming[dim].isa<MappingDimExpr>()) continue;
    renaming[dim] = MappingDimExpr::get(domain_.size(), context);
    domain_.push_back({
        .value = old_domain[dim].value,
        .mapping = old_domain[dim].mapping.ResizeUseDomain(new_size),
    });
  }

  for (MappingExpr &layout_expr : layout_) {
    layout_expr = layout_expr.SubstituteDims(renaming);
    assert(layout_expr.IsFullySpecified());
  }
}

void Buffer::UnifyLayoutDim(int layout_dim, MappingExpr expr) {
  layout_[layout_dim] = layout_[layout_dim].Unify(expr);
  assert(layout_[layout_dim] != nullptr);
}

void Buffer::AddWrite(ComputeOp op, int result) {
  writes_.emplace_back(op, result);
}

StorageAnalysis::StorageAnalysis(mlir::Operation *operation)
    : StorageAnalysis(operation->getContext()) {
  mlir::LogicalResult result = Init(cast<SairProgramOp>(operation));
  assert(mlir::succeeded(result));
  (void)result;
}

std::optional<StorageAnalysis> StorageAnalysis::Create(SairProgramOp program) {
  StorageAnalysis analysis(program.getContext());
  if (mlir::failed(analysis.Init(program))) {
    return std::nullopt;
  }
  return analysis;
}

// Verifies that the storage attribute of the operation is well-formed:
// - that storage attributes are arrays of buffer or unit attributes,
// - that the number of entries in the storage array matches the number of,
//   results of the operation,
// - that indexes are not stored in memory,
// - that memory spaces referenced by the attribute exist,
// - that multi-dimensional buffers are not stored in registers,
// - that loops referenced by the attribute exist and
// - that the buffer has a name if and only if the memory space is addressable.
static mlir::LogicalResult VerifyStorageAttrWellFormed(ComputeOp op) {
  auto *sair_dialect = op.getContext()->getLoadedDialect<SairDialect>();

  llvm::Optional<mlir::ArrayAttr> storage_attr = op.storage();
  if (!op.storage().hasValue()) return mlir::success();
  llvm::ArrayRef<mlir::Attribute> storage = storage_attr.getValue().getValue();

  if (storage.size() != op->getNumResults()) {
    return op.emitError() << "wrong number of storage entries";
  }

  llvm::DenseSet<mlir::Attribute> loop_names;
  if (op.loop_nest().hasValue()) {
    for (mlir::Attribute attr : op.LoopNestLoops()) {
      LoopAttr loop = attr.cast<LoopAttr>();
      loop_names.insert(loop.name());
    }
  }

  for (auto [attr, value] : llvm::zip(storage, op->getResults())) {
    if (attr.isa<UnitAttr>()) continue;
    BufferAttr buffer = attr.dyn_cast<BufferAttr>();
    if (buffer == nullptr) {
      return op.emitError() << "storage attribute must be an array of buffers "
                               "or unit attributes";
    }

    if (buffer.space() != sair_dialect->register_attr() &&
        buffer.space() != sair_dialect->memory_attr()) {
      return op.emitError() << "invalid memory space " << buffer.space();
    }

    ValueType type = value.getType().cast<ValueType>();
    if (buffer.space() == sair_dialect->memory_attr() &&
        type.ElementType().isa<mlir::IndexType>()) {
      return op.emitError() << "index variables cannot be allocated in memory";
    }

    if ((buffer.space() == sair_dialect->memory_attr()) ^
        buffer.name() != nullptr) {
      return op.emitError() << "buffers must have a name if and only if they "
                               "are stored in memory";
    }

    if (buffer.space() == sair_dialect->register_attr() &&
        !buffer.layout().mapping().empty()) {
      return op.emitError() << "only 0D buffers can be stored in registers";
    }

    for (mlir::StringAttr loop_name : buffer.layout().names()) {
      if (!loop_names.contains(loop_name)) {
        return op.emitError() << "unknown loop name " << loop_name;
      }
    }
  }

  return mlir::success();
}

// Returns a mapping from the loop nest of `op` to the loops indexed by
// `buffer_attr`.
static MappingAttr LoopsToIndexedLoopsMapping(ComputeOp op,
                                              BufferAttr buffer_attr,
                                              const Buffer &buffer) {
  mlir::MLIRContext *context = op.getContext();
  auto none_expr = MappingNoneExpr::get(context);
  auto loop_nest = op.LoopNestLoops();

  llvm::SmallVector<MappingExpr> loops_to_indexed_loops_exprs(buffer.rank(),
                                                              none_expr);
  for (auto p : llvm::enumerate(buffer_attr.layout().names())) {
    auto it = llvm::find_if(loop_nest, [&](mlir::Attribute attr) {
      auto loop = attr.cast<LoopAttr>();
      return loop.name() == p.value();
    });
    assert(it != loop_nest.end());
    int pos = std::distance(loop_nest.begin(), it);
    loops_to_indexed_loops_exprs[p.index()] = MappingDimExpr::get(pos, context);
  }

  return MappingAttr::get(context, loop_nest.size(),
                          loops_to_indexed_loops_exprs);
}

// Declares buffer `attr` in `buffer_map`. If the
// buffer is already present, ensure that rank and element type are coherent and
// trims the buffer loop nest to the common prefix with `op` loop nest.
static mlir::LogicalResult DeclareBuffer(
    ComputeOp op, int result, BufferAttr attr,
    const LoopFusionAnalysis &fusion_analysis,
    llvm::DenseMap<mlir::Attribute, Buffer> &buffer_map) {
  if (attr == nullptr || attr.name() == nullptr) return mlir::success();
  mlir::Type element_type =
      op->getResult(result).getType().cast<ValueType>().ElementType();

  int rank = attr.layout().mapping().size();
  auto it = buffer_map.try_emplace(attr.name(), element_type, rank, op, result);
  Buffer &buffer = it.first->second;

  if (!it.second) {
    buffer.AddWrite(op, result);
  }

  // Check that element types match.
  if (buffer.element_type() != element_type) {
    mlir::InFlightDiagnostic diag =
        op.emitError()
        << "buffer " << attr.name()
        << " has different element type than in previous occurence";
    diag.attachNote(buffer.getLoc()) << "previous occurence here";
    return mlir::failure();
  }

  // Ensure that the number of dimension is coherent.
  if (buffer.rank() != rank) {
    mlir::InFlightDiagnostic diag = op.emitError()
                                    << "buffer " << attr.name()
                                    << " rank differs from previous occurence";
    diag.attachNote(buffer.getLoc()) << "previous occurence here";
    return mlir::failure();
  }

  // Trims the buffer loop nest
  auto loop_nest = op.LoopNestLoops();
  int num_deps = std::min(loop_nest.size(), buffer.loop_nest().size());
  for (; num_deps > 0; --num_deps) {
    LoopAttr loop = loop_nest[num_deps - 1].cast<LoopAttr>();
    if (loop.name() == buffer.loop_nest()[num_deps - 1]) break;
  }

  llvm::DenseSet<mlir::Attribute> loop_names;
  for (mlir::StringAttr name : attr.layout().names()) {
    loop_names.insert(name);
  }

  for (int i = 0; i < num_deps; ++i) {
    // Trim indexed loops from the loop nest.
    if (loop_names.count(buffer.loop_nest()[i]) > 0) {
      num_deps = i;
      break;
    }
  }

  buffer.TrimLoopNest(num_deps, fusion_analysis);
  return mlir::success();
}

// Unifies the shape of `buffer` with the shape specified by attribute
// `buffer_attr` of `op`. Raises an error if shapes cannot be unified.
static mlir::LogicalResult UnifyBufferShape(
    ComputeOp op, BufferAttr buffer_attr, const LoopNest &buffer_loop_nest,
    MappingAttr loops_to_indexed_loops,
    const LoopFusionAnalysis &fusion_analysis, Buffer &buffer) {
  mlir::MLIRContext *context = buffer_attr.getContext();
  auto none_expr = MappingNoneExpr::get(context);
  LoopNest loop_nest = fusion_analysis.GetLoopNest(op.LoopNestLoops());

  // Get a mapping from domain to buffer layout.
  MappingAttr domain_to_layout =
      loop_nest.domain_to_loops.Compose(loops_to_indexed_loops)
          .Compose(buffer_attr.layout().mapping());

  // Compute unification constraints. Dimensions used by the buffer loop nest
  // must be exactly the same for both uses.
  llvm::SmallVector<MappingExpr, 4> constraints(loop_nest.domain.size(),
                                                none_expr);
  for (int i = 0, e = buffer_loop_nest.domain.size(); i < e; ++i) {
    constraints[i] = MappingDimExpr::get(i, context);
  }

  for (auto [old_expr, new_expr] :
       llvm::zip(buffer.layout(), domain_to_layout)) {
    if (mlir::failed(new_expr.UnificationConstraints(old_expr, constraints))) {
      return op.emitError()
             << "buffer " << buffer_attr.name()
             << " layout is incompatible with previous occurences";
    }
  }

  // Resolve constraints.
  std::string buffer_name_internal;
  llvm::raw_string_ostream buffer_name(buffer_name_internal);
  buffer_name << "buffer " << buffer_attr.name();

  llvm::SmallBitVector indexed_dims = domain_to_layout.DependencyMask();
  for (int dimension : indexed_dims.set_bits()) {
    ValueAccess dim_access = loop_nest.domain[dimension];
    dim_access.mapping =
        dim_access.mapping.ResizeUseDomain(buffer.loop_nest().size());
    if (!dim_access.mapping.IsFullySpecified()) {
      return op.emitError()
             << "buffer " << buffer_attr.name()
             << " layout depends on loops it cannot be nested in";
    }
    if (mlir::failed(ResolveUnificationConstraint(
            op.getLoc(), buffer_name.str(), dim_access, constraints[dimension],
            buffer.domain()))) {
      return mlir::failure();
    }
  }

  // Unify dimensions.
  for (int i = 0; i < buffer.rank(); ++i) {
    buffer.UnifyLayoutDim(
        i, domain_to_layout.Dimension(i).SubstituteDims(constraints));
  }

  return mlir::success();
}

// Ensures that we can insert a malloc operation for the buffer. Increases
// `min_num_loops` to make sure that a malloc operation can be inserted if
// needed.
static mlir::LogicalResult CheckMallocInsertionPoint(
    mlir::StringAttr buffer_name, const Buffer &buffer,
    const llvm::SmallBitVector &used_dimensions,
    const IterationSpaceAnalysis &iteration_spaces, int &min_num_loops) {
  // Find the first compute op writting to the buffer.
  ComputeOp first_write = buffer.writes().front().first;
  for (auto p : buffer.writes()) {
    if (p.first->isBeforeInBlock(first_write)) {
      first_write = p.first;
    }
  }

  llvm::ArrayRef<mlir::Attribute> write_loop_nest = first_write.LoopNestLoops();

  for (int dim : used_dimensions.set_bits()) {
    auto dimension_op =
        cast<SairOp>(buffer.domain()[dim].value.getDefiningOp());
    if (first_write->isBeforeInBlock(dimension_op)) {
      mlir::InFlightDiagnostic diag =
          first_write.emitError()
          << "buffer " << buffer_name
          << " is used before one of its dimensions is defined";
      diag.attachNote(dimension_op.getLoc()) << "dimension defined here";
      return mlir::failure();
    }

    for (ValueOperand operand : dimension_op.ValueOperands()) {
      llvm::ArrayRef<mlir::Attribute> operand_iteration_space =
          iteration_spaces.IterationSpace(operand.value());
      int new_min =
          std::min(write_loop_nest.size(), operand_iteration_space.size());
      for (; new_min > 0; --new_min) {
        auto write_loop = write_loop_nest[new_min - 1].cast<LoopAttr>();
        auto operand_loop =
            operand_iteration_space[new_min - 1].cast<LoopAttr>();
        if (write_loop.name() == operand_loop.name()) break;
      }

      // TODO(b/170195606): this check is not enough if other operations are
      // present between the dimension definition and its arguments.
      if (new_min > buffer.loop_nest().size()) {
        mlir::InFlightDiagnostic diag =
            first_write.emitError()
            << "buffer " << buffer_name
            << " depends on a dimension that is defined after the buffer "
               "is allocated";
        diag.attachNote(dimension_op.getLoc()) << "dimension defined here";
        return mlir::failure();
      }

      min_num_loops = std::max(min_num_loops, new_min);
    }
  }
  return mlir::success();
}

// Check that buffer's layout is well-formed and only depends on loops in
// `loop_nest`. Increases `min_num_loops` to the minimal number of loops needed
// in `loop_nest` for the layout to be valid.
static mlir::LogicalResult CheckLayoutMapping(
    const LoopNest &loop_nest, mlir::StringAttr buffer_name,
    const Buffer &buffer, const IterationSpaceAnalysis &iteration_spaces,
    int &min_num_loops) {
  mlir::MLIRContext *context = loop_nest.domain_to_loops.getContext();
  int domain_size = buffer.domain().size();
  int loop_nest_size = buffer.loop_nest().size();

  // Get a mapping that maps both loop-nest and layout indices. This corresponds
  // to the different instances of the buffer.
  llvm::SmallVector<MappingExpr> exprs;
  exprs.reserve(loop_nest_size + buffer.layout().size());
  llvm::append_range(exprs, loop_nest.domain_to_loops);
  llvm::append_range(exprs, buffer.layout());
  auto mapping = MappingAttr::get(context, domain_size, exprs);

  // Update `min_num_loops` based on domain dimensions layout depends on.
  llvm::SmallBitVector used_dimensions(domain_size);
  for (MappingExpr layout_expr : buffer.layout()) {
    layout_expr.SetDependenciesInMask(used_dimensions);
  }
  for (int dim : used_dimensions.set_bits()) {
    int new_min = buffer.domain()[dim].mapping.MinDomainSize();
    min_num_loops = std::max(new_min, min_num_loops);
  }

  // Update `min_num_loop` to account for dependencies accross layout and
  // loop-nest dimensions.
  auto hr_shape = DomainShapeAttr::HyperRectangular(context, domain_size);
  MappingAttr inverse = mapping.Inverse().ResizeUseDomain(loop_nest_size);
  for (MappingExpr layout_expr : buffer.layout()) {
    DomainShapeDim shape_dim =
        layout_expr.AccessedShape(hr_shape.Dimensions(), inverse);
    if (!shape_dim.dependency_mapping().IsFullySpecified()) {
      return mlir::emitError(buffer.getLoc())
             << "buffer " << buffer_name
             << " layout depends on loops it cannot be nested in";
    }
    int new_min = shape_dim.dependency_mapping().MinDomainSize();
    min_num_loops = std::max(new_min, min_num_loops);
  }

  return CheckMallocInsertionPoint(buffer_name, buffer, used_dimensions,
                                   iteration_spaces, min_num_loops);
}

mlir::LogicalResult StorageAnalysis::Init(SairProgramOp program) {
  // TODO(b/181938550): use cached analysis.
  LoopFusionAnalysis fusion_analysis(program);
  IterationSpaceAnalysis iteration_spaces(program);

  // Declare buffer uses and check element type/rank.
  mlir::WalkResult result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      BufferAttr buffer_attr = op.Storage(i);
      if (mlir::failed(
              DeclareBuffer(op, i, buffer_attr, fusion_analysis, buffers_))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  });
  if (result.wasInterrupted()) return mlir::failure();

  for (auto &[name_attr, buffer] : buffers_) {
    mlir::StringAttr name = name_attr.cast<mlir::StringAttr>();

    // Prefix buffer domain with its loop nest domain.
    const LoopNest &loop_nest = fusion_analysis.GetLoopNest(buffer.loop_nest());
    buffer.domain().reserve(loop_nest.domain.size());
    int num_loops = buffer.loop_nest().size();
    for (const ValueAccess &access : loop_nest.domain) {
      buffer.domain().push_back(
          {access.value, access.mapping.ResizeUseDomain(num_loops)});
    }

    // Unify buffer layout.
    for (auto [op, result] : buffer.writes()) {
      BufferAttr buffer_attr = op.Storage(result);
      // Unify layouts.
      MappingAttr loops_to_indexed_loops =
          LoopsToIndexedLoopsMapping(op, buffer_attr, buffer);
      if (mlir::failed(UnifyBufferShape(op, buffer_attr, loop_nest,
                                        loops_to_indexed_loops, fusion_analysis,
                                        buffer))) {
        return mlir::failure();
      }
    }

    // Check that layout mapping is correct and compute the minimal loop nest
    // each buffer needs to be nested in.
    int min_num_loops = 0;
    if (mlir::failed(CheckLayoutMapping(loop_nest, name, buffer,
                                        iteration_spaces, min_num_loops))) {
      return mlir::failure();
    }

    // Minimize layout loop-nest.
    buffer.TrimLoopNest(min_num_loops, fusion_analysis);
  }

  // Compute value storages.
  program.walk([&](ComputeOp op) {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      ValueStorage &value_storage = value_storages_[op->getResult(i)];
      BufferAttr buffer = op.Storage(i);
      if (buffer == nullptr) continue;
      value_storage.space = buffer.space();
      value_storage.buffer_name = buffer.name();
    }
  });

  for (ValueViewOp op : SortValueViews(program)) {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      auto result_opt = op.InferStorage(i, value_storages_);
      if (!result_opt.has_value()) {
        return op.emitError() << "operands have different storage";
      }
      value_storages_[op->getResult(i)] = result_opt.value();
    }
  }

  return mlir::success();
}

mlir::StringAttr StorageAnalysis::GetFreshBufferName() {
  llvm::SmallString<10> name("buffer_");
  int original_size = name.size();
  mlir::StringAttr attr;
  do {
    name.resize(original_size);
    name += std::to_string(next_buffer_id_++);
    attr = mlir::StringAttr::get(context_, name);
  } while (buffers_.count(attr) > 0);
  return attr;
}

mlir::LogicalResult VerifyStorages(SairProgramOp program) {
  // Check storage attributes are well-formed.
  mlir::WalkResult result = program.walk([](ComputeOp op) -> mlir::WalkResult {
    return VerifyStorageAttrWellFormed(op);
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Ensure storage attributes are compatibles with each other.
  auto analysis_result = StorageAnalysis::Create(program);
  if (!analysis_result.has_value()) return mlir::failure();
  StorageAnalysis analysis = std::move(analysis_result).value();

  // Ensure that map-reduce arguments are compatible.
  result = program.walk([&](SairMapReduceOp op) -> mlir::WalkResult {
    for (ValueOperand input : op.Inits()) {
      ValueStorage input_storage =
          analysis.GetStorage(input.value()).Map(input.Mapping());
      const ValueStorage &result_storage =
          analysis.GetStorage(op.getResult(input.position()));
      if (input_storage != result_storage) {
        return op.emitError() << "initializer and result storages must match";
      }
    }
    return mlir::success();
  });

  // TODO(b/174127497): dependency analysis
  return mlir::failure(result.wasInterrupted());
}

BufferAttr GetRegister0DBuffer(mlir::MLIRContext *context) {
  auto *sair_dialect = context->getLoadedDialect<SairDialect>();
  return BufferAttr::get(/*space=*/sair_dialect->register_attr(),
                         /*name=*/nullptr,
                         /*layout=*/NamedMappingAttr::GetIdentity(context, {}),
                         context);
}

}  // namespace sair