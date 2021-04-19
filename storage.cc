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

Buffer::Buffer(mlir::Type element_type, ComputeOp op, int result,
               const LoopFusionAnalysis &fusion_analysis)
    : loc_(op.getLoc()), element_type_(element_type), import_op_(nullptr) {
  assert(element_type != nullptr);

  llvm::ArrayRef<mlir::Attribute> loop_nest = op.LoopNestLoops();
  loop_nest_.reserve(loop_nest.size());
  for (mlir::Attribute attr : loop_nest) {
    auto loop = attr.cast<LoopAttr>();
    loop_nest_.push_back(loop.name());
  }
  loop_nest_mapping_ = fusion_analysis.GetLoopNest(loop_nest_).domain_to_loops;
  writes_.emplace_back(op, result);
}

Buffer::Buffer(FromToMemRefOp import_op,
               const IterationSpaceAnalysis &iteration_spaces,
               const LoopFusionAnalysis &fusion_analysis)
    : loc_(import_op.getLoc()),
      element_type_(import_op.MemRefType().getElementType()),
      import_op_(import_op) {
  const IterationSpace iter_space =
      iteration_spaces.Get(cast<SairOp>(import_op.getOperation()));
  llvm::append_range(loop_nest_, iter_space.loop_names());
  loop_nest_mapping_ = fusion_analysis.GetLoopNest(loop_nest_).domain_to_loops;
}

std::optional<int> Buffer::rank() const {
  return layout_.has_value() ? std::make_optional(layout_->size())
                             : std::nullopt;
}

void Buffer::TrimLoopNest(int new_size) {
  assert(new_size <= loop_nest_.size());
  loop_nest_.resize(new_size);
  loop_nest_mapping_ = loop_nest_mapping_.Resize(new_size);
  if (domain_.empty()) return;

  mlir::MLIRContext *context = element_type_.getContext();
  // Compute dimensions used by layout.
  llvm::SmallBitVector used_dimensions(domain_.size());
  used_dimensions |= loop_nest_mapping_.DependencyMask();
  if (layout_.has_value()) {
    used_dimensions |= layout_.value().DependencyMask();
  }

  // Trim domain from unused dimensions.
  llvm::SmallVector<ValueAccess> old_domain;
  std::swap(old_domain, domain_);
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

  if (layout_.has_value()) {
    auto renaming_mapping = MappingAttr::get(context, domain_.size(), renaming);
    layout_ = renaming_mapping.Compose(layout_.value());
    assert(layout_->IsSurjective());
  }
}

void Buffer::UnifyLayout(MappingAttr layout) {
  if (!layout_.has_value()) {
    layout_ = layout;
  } else {
    layout_ =
        layout_.value().ResizeUseDomain(domain_.size()).UnifyNoneExprs(layout);
  }
}

void Buffer::AddWrite(ComputeOp op, int result) {
  writes_.emplace_back(op, result);
}

void Buffer::AddRead(ComputeOp op, int operand) {
  reads_.emplace_back(op, operand);
}

void Buffer::AddValue(mlir::Value value) { values_.push_back(value); }

MappingAttr Buffer::PrefixedLayout() const {
  assert(layout_.has_value());

  mlir::MLIRContext *context = loop_nest_mapping_.getContext();
  llvm::SmallVector<MappingExpr> exprs;
  exprs.reserve(loop_nest_.size() + layout_->size());
  llvm::append_range(exprs, loop_nest_mapping_);
  llvm::append_range(exprs, layout_.value());
  return MappingAttr::get(context, domain_.size(), exprs);
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

  llvm::DenseSet<mlir::Attribute> buffer_names;
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
        (type.ElementType().isa<mlir::IndexType>() ||
         type.ElementType().isa<mlir::MemRefType>())) {
      return op.emitError()
             << "index and memref variables cannot be allocated in memory";
    }

    if ((buffer.space() == sair_dialect->memory_attr()) ^
        buffer.name() != nullptr) {
      return op.emitError() << "buffers must have a name if and only if they "
                               "are stored in memory";
    }

    if (buffer.name() != nullptr &&
        !buffer_names.insert(buffer.name()).second) {
      return op.emitError()
             << "operation cannot store two results in the same buffer";
    }

    if (buffer.layout() == nullptr) continue;

    if (buffer.layout().mapping().HasUnknownExprs()) {
      return op.emitError() << "layouts cannot contain `?` expressions";
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

// Returns a mapping from the iteration space of `op` to the loops indexed by
// `buffer_attr`.
static MappingAttr IterSpaceToIndexedLoopsMapping(
    ComputeOp op, BufferAttr buffer,
    const IterationSpaceAnalysis &iteration_spaces) {
  mlir::MLIRContext *context = op.getContext();
  auto none_expr = MappingNoneExpr::get(context);
  const IterationSpace &iter_space = iteration_spaces.Get(op.getOperation());

  llvm::SmallVector<MappingExpr> loops_to_indexed_loops_exprs(
      buffer.layout().mapping().UseDomainSize(), none_expr);
  for (auto p : llvm::enumerate(buffer.layout().names())) {
    auto it = llvm::find(iter_space.loop_names(), p.value());
    assert(it != iter_space.loop_names().end());
    int pos = std::distance(iter_space.loop_names().begin(), it);
    loops_to_indexed_loops_exprs[p.index()] = MappingDimExpr::get(pos, context);
  }

  return MappingAttr::get(context, iter_space.mapping().size(),
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

  auto it = buffer_map.try_emplace(attr.name(), element_type, op, result,
                                   fusion_analysis);
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

  // Trims the buffer loop nest
  auto loop_nest = op.LoopNestLoops();
  int num_deps = std::min(loop_nest.size(), buffer.loop_nest().size());
  for (; num_deps > 0; --num_deps) {
    LoopAttr loop = loop_nest[num_deps - 1].cast<LoopAttr>();
    if (loop.name() == buffer.loop_nest()[num_deps - 1]) break;
  }

  if (attr.layout() != nullptr) {
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
  }

  buffer.TrimLoopNest(num_deps);
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
  LoopNest loop_nest = fusion_analysis.GetLoopNest(op);

  // Get a mapping from domain to buffer layout.
  MappingAttr domain_to_layout =
      loop_nest.domain_to_loops.Compose(loops_to_indexed_loops)
          .Compose(buffer_attr.layout().mapping())
          .Canonicalize();

  // Compute unification constraints. Dimensions used by the buffer loop nest
  // must be exactly the same for both uses.
  llvm::SmallVector<MappingExpr, 4> constraints(loop_nest.domain.size(),
                                                none_expr);
  for (int i = 0, e = buffer_loop_nest.domain.size(); i < e; ++i) {
    constraints[i] = MappingDimExpr::get(i, context);
  }

  if (buffer.layout().has_value()) {
    for (auto [old_expr, new_expr] :
         llvm::zip(buffer.layout().value(), domain_to_layout)) {
      if (mlir::failed(
              UnificationConstraints(new_expr, old_expr, constraints))) {
        return op.emitError()
               << "buffer " << buffer_attr.name()
               << " layout is incompatible with previous occurences";
      }
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
    if (dim_access.mapping.HasNoneExprs()) {
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
  auto buffer_domain_to_domain =
      MappingAttr::get(context, buffer.domain().size(), constraints);
  buffer.UnifyLayout(buffer_domain_to_domain.Compose(domain_to_layout));

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

  llvm::ArrayRef<mlir::StringAttr> write_loops =
      iteration_spaces.Get(cast<SairOp>(first_write.getOperation()))
          .loop_names();
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
      auto defining_op = cast<SairOp>(operand.value().getDefiningOp());
      llvm::ArrayRef<mlir::StringAttr> operand_loops =
          iteration_spaces.Get(defining_op).loop_names();
      int new_min = std::min(write_loops.size(), operand_loops.size());
      for (; new_min > 0; --new_min) {
        if (operand_loops[new_min - 1] == write_loops[new_min - 1]) break;
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
    mlir::StringAttr buffer_name, const Buffer &buffer,
    const IterationSpaceAnalysis &iteration_spaces, int &min_num_loops) {
  if (!buffer.layout().has_value()) return mlir::success();

  mlir::MLIRContext *context = buffer_name.getContext();
  int domain_size = buffer.domain().size();
  int loop_nest_size = buffer.loop_nest().size();

  MappingAttr mapping = buffer.PrefixedLayout();
  if (mapping.HasNoneExprs()) {
    return mlir::emitError(buffer.getLoc())
           << "buffer " << buffer_name << " layout is not fully specified";
  }

  // Update `min_num_loops` based on domain dimensions layout depends on.
  llvm::SmallBitVector used_dimensions = buffer.layout()->DependencyMask();
  for (int dim : used_dimensions.set_bits()) {
    int new_min = buffer.domain()[dim].mapping.MinDomainSize();
    min_num_loops = std::max(new_min, min_num_loops);
  }

  // Update `min_num_loop` to account for dependencies accross layout and
  // loop-nest dimensions.
  auto hr_shape = DomainShapeAttr::HyperRectangular(context, domain_size);
  MappingAttr inverse = mapping.Inverse().ResizeUseDomain(loop_nest_size);
  for (MappingExpr layout_expr : buffer.layout()->Dimensions()) {
    DomainShapeDim shape_dim =
        layout_expr.AccessedShape(hr_shape.Dimensions(), inverse);
    if (shape_dim.dependency_mapping().HasNoneExprs()) {
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

// Declare buffers used by `program` in `buffers`. If a buffer has multiple
// uses, chek that element type and rank are compatible.
static mlir::LogicalResult DeclareBuffers(
    SairProgramOp program, const IterationSpaceAnalysis &iteration_spaces,
    const LoopFusionAnalysis &fusion_analysis,
    llvm::DenseMap<mlir::Attribute, Buffer> &buffers) {
  // Declare external buffers imported using from/to memref.
  mlir::WalkResult result =
      program.walk([&](FromToMemRefOp op) -> mlir::WalkResult {
        auto name_attr =
            mlir::StringAttr::get(op.getContext(), op.buffer_name());
        bool was_inserted =
            buffers
                .try_emplace(name_attr, op, iteration_spaces, fusion_analysis)
                .second;
        if (!was_inserted) {
          return op.emitError() << "buffer name is already used";
        }
        return mlir::success();
      });
  if (result.wasInterrupted()) return mlir::failure();

  // Declare internal buffers.
  result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      BufferAttr buffer_attr = op.Storage(i);
      if (mlir::failed(
              DeclareBuffer(op, i, buffer_attr, fusion_analysis, buffers))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  });
  return mlir::failure(result.wasInterrupted());
}

// Compute buffers shape by updating `buffers` in-place.
static mlir::LogicalResult ComputeBuffersShape(
    mlir::MLIRContext *context, const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    llvm::DenseMap<mlir::Attribute, Buffer> &buffers) {
  for (auto &[name_attr, buffer] : buffers) {
    mlir::StringAttr name = name_attr.cast<mlir::StringAttr>();

    // Prefix buffer domain with its loop nest domain.
    const LoopNest &loop_nest = fusion_analysis.GetLoopNest(buffer.loop_nest());
    buffer.domain().reserve(loop_nest.domain.size());
    int num_loops = buffer.loop_nest().size();
    for (const ValueAccess &access : loop_nest.domain) {
      buffer.domain().push_back(
          {access.value, access.mapping.ResizeUseDomain(num_loops)});
    }

    // Define external buffers shape.
    if (buffer.is_external()) {
      FromToMemRefOp op = buffer.import_op();
      auto sair_op = cast<SairOp>(op.getOperation());
      int parallel_domain_size = op.parallel_domain().size();
      MappingAttr loops_to_domain_mapping =
          iteration_spaces.Get(sair_op).MappingToLoops().Inverse();
      int rank = op.memref_domain().size();
      for (int i = 0; i < rank; ++i) {
        const DomainShapeDim &shape_dim =
            sair_op.shape().Dimensions()[parallel_domain_size + i];
        MappingAttr shape_dim_mapping = shape_dim.dependency_mapping();
        buffer.domain().push_back({
            .value = op.memref_domain()[i],
            .mapping = loops_to_domain_mapping.Compose(
                shape_dim_mapping.ResizeUseDomain(parallel_domain_size)),
        });
      }
      buffer.UnifyLayout(MappingAttr::GetIdentity(context, rank));
    }

    // Unify buffer layout.
    for (auto [op, result] : buffer.writes()) {
      BufferAttr buffer_attr = op.Storage(result);
      if (buffer_attr.layout() == nullptr) continue;

      // Ensure that the number of dimension is coherent.
      if (buffer.rank().has_value() &&
          buffer.rank() != buffer_attr.layout().mapping().size()) {
        mlir::InFlightDiagnostic diag =
            op.emitError() << "buffer " << name
                           << " rank differs from previous occurence";
        diag.attachNote(buffer.getLoc()) << "previous occurence here";
        return mlir::failure();
      }

      // Unify layouts.
      MappingAttr loops_to_indexed_loops =
          IterSpaceToIndexedLoopsMapping(op, buffer_attr, iteration_spaces);
      if (mlir::failed(UnifyBufferShape(op, buffer_attr, loop_nest,
                                        loops_to_indexed_loops, fusion_analysis,
                                        buffer))) {
        return mlir::failure();
      }
    }

    // If the buffer is external, we already know that its layout is correct and
    // we cannot trim the list of loops its definition must be nested in.
    if (buffer.is_external()) continue;

    // Check that layout mapping is correct and compute the minimal loop nest
    // buffers needs to be nested in.
    int min_num_loops = 0;
    if (mlir::failed(CheckLayoutMapping(name, buffer, iteration_spaces,
                                        min_num_loops))) {
      return mlir::failure();
    }

    // Minimize layout loop-nest.
    buffer.TrimLoopNest(min_num_loops);
  }

  return mlir::success();
}

// Computes how values are stored and stores the result into `value_storages`.
mlir::LogicalResult StorageAnalysis::ComputeValueStorages(
    SairProgramOp program, const IterationSpaceAnalysis &iteration_spaces) {
  mlir::MLIRContext *context = program.getContext();
  auto *sair_dialect = context->getLoadedDialect<SairDialect>();
  mlir::StringAttr memory_space = sair_dialect->memory_attr();

  // Initialize storage information from compute operations.
  auto result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      BufferAttr buffer = op.Storage(i);
      if (buffer == nullptr) continue;
      MappingAttr layout;
      if (buffer.layout() != nullptr) {
        layout = IterSpaceToIndexedLoopsMapping(op, buffer, iteration_spaces)
                     .Compose(buffer.layout().mapping());
      }
      ValueStorage storage(buffer.space(), buffer.name(), layout);
      if (mlir::failed(
              SetStorage(op->getResult(i), storage, iteration_spaces))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Initialize from from_memref operations.
  result = program.walk([&](SairFromMemRefOp op) -> mlir::WalkResult {
    const IterationSpace &iter_space = iteration_spaces.Get(op);
    MappingAttr layout = iter_space.mapping().Inverse().Compose(op.Layout());
    ValueStorage storage(memory_space, op.buffer_nameAttr(), layout);
    return SetStorage(op.result(), storage, iteration_spaces);
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Initialize from from_scalar operations.
  result = program.walk([&](SairFromScalarOp op) -> mlir::WalkResult {
    auto layout = MappingAttr::get(context, 0, {});
    ValueStorage storage(sair_dialect->register_attr(), nullptr, layout);
    return SetStorage(op.result(), storage, iteration_spaces);
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Initialize from to_memref operations.
  result = program.walk([&](SairToMemRefOp op) -> mlir::WalkResult {
    const IterationSpace &iter_space = iteration_spaces.Get(op);
    MappingAttr layout = iter_space.mapping().Inverse().Compose(op.Layout());
    ValueStorage operand_storage(memory_space, op.buffer_nameAttr(), layout);
    mlir::Operation *defining_op = op.value().getDefiningOp();
    ValueStorage storage = operand_storage.Map(
        op, defining_op, op.Value().Mapping().Inverse(), iteration_spaces);
    return SetStorage(op.value(), storage, iteration_spaces);
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Ensure all sair values have an entry.
  program.walk([&](SairOp op) {
    for (mlir::Value result : op->getResults()) {
      value_storages_.FindAndConstruct(result);
    }
  });

  return mlir::success();
}

mlir::LogicalResult StorageAnalysis::Init(SairProgramOp program) {
  // TODO(b/181938550): use cached analysis.
  LoopFusionAnalysis fusion_analysis(program);
  IterationSpaceAnalysis iteration_spaces(program);
  mlir::MLIRContext *context = program.getContext();

  if (mlir::failed(DeclareBuffers(program, iteration_spaces, fusion_analysis,
                                  buffers_))) {
    return mlir::failure();
  }

  if (mlir::failed(ComputeBuffersShape(context, fusion_analysis,
                                       iteration_spaces, buffers_))) {
    return mlir::failure();
  }

  if (mlir::failed(ComputeValueStorages(program, iteration_spaces))) {
    return mlir::failure();
  }

  // Register buffer reads.
  program.walk([&](ComputeOp op) {
    for (ValueOperand operand : cast<SairOp>(*op).ValueOperands()) {
      const ValueStorage &storage = value_storages_[operand.value()];
      if (storage.buffer_name() == nullptr) continue;
      buffers_.find(storage.buffer_name())
          ->second.AddRead(op, operand.position());
    }
  });

  // Ensure that writes to external buffers occure after the buffer is defined.
  for (auto &[name, buffer] : buffers_) {
    if (!buffer.is_external()) continue;
    mlir::Operation *defining_op =
        buffer.import_op().MemRef().value().getDefiningOp();
    // We only need to check writes as reads always occure after writes.
    for (auto write : buffer.writes()) {
      if (write.first->isBeforeInBlock(defining_op)) {
        mlir::InFlightDiagnostic diag = write.first.emitError()
                                        << "buffer " << name
                                        << " used before it is defined";
        diag.attachNote(defining_op->getLoc()) << "buffer defined here";
        return mlir::failure();
      }
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

mlir::LogicalResult StorageAnalysis::SetStorage(
    mlir::Value value, ValueStorage storage,
    const IterationSpaceAnalysis &iteration_spaces) {
  llvm::SmallVector<mlir::Value> work_list;

  // Merge storage information for a value with existing information. Fails and
  // emits an error in case of conflicts.
  auto merge_storage = [&](mlir::Value value,
                           ValueStorage new_storage) -> mlir::LogicalResult {
    ValueStorage &storage = value_storages_[value];
    if (new_storage == storage) return mlir::success();

    if (storage.buffer_name() == nullptr &&
        new_storage.buffer_name() != nullptr) {
      buffers_.find(new_storage.buffer_name())->second.AddValue(value);
    }

    if (mlir::failed(storage.MergeSpace(new_storage.space()))) {
      return value.getDefiningOp()->emitError()
             << "conflicting memory spaces: expected " << new_storage.space()
             << ", got " << storage.space();
    }
    if (mlir::failed(storage.MergeBufferName(new_storage.buffer_name()))) {
      return value.getDefiningOp()->emitError()
             << "conflicting buffer names: expected "
             << new_storage.buffer_name() << ", got " << storage.buffer_name();
    }
    if (mlir::failed(storage.MergeLayout(new_storage.layout()))) {
      return value.getDefiningOp()->emitError()
             << "conflicting layouts: expected " << new_storage.layout()
             << ", got " << storage.layout();
    }

    work_list.push_back(value);
    return mlir::success();
  };

  if (mlir::failed(merge_storage(value, storage))) return mlir::failure();

  // Propagate storage information.
  while (!work_list.empty()) {
    mlir::Value value = work_list.pop_back_val();
    ValueStorage storage = value_storages_[value];

    // Forward propagation.
    for (mlir::OpOperand &mlir_operand : value.getUses()) {
      mlir::Operation *user = mlir_operand.getOwner();
      ValueOperand operand(&mlir_operand);
      int result;
      if (isa<SairProjAnyOp, SairProjLastOp, SairFbyOp>(user)) {
        result = 0;
      } else if (auto map_reduce = dyn_cast<SairMapReduceOp>(user)) {
        if (operand.position() >= map_reduce.Inits().size()) continue;
        result = operand.position();
      } else {
        continue;
      }
      ValueStorage new_storage = storage.Map(operand, iteration_spaces);
      if (mlir::failed(merge_storage(user->getResult(result), new_storage))) {
        return mlir::failure();
      }
    }

    // Backward propagation.
    mlir::Operation *defining_op = value.getDefiningOp();

    // Handle map-reduce separately.
    if (auto map_reduce = dyn_cast<SairMapReduceOp>(defining_op)) {
      int pos = value.cast<OpResult>().getResultNumber();
      ValueOperand operand = map_reduce.Inits()[pos];
      ValueStorage new_storage =
          storage.Map(defining_op, operand.value().getDefiningOp(),
                      operand.Mapping().Inverse(), iteration_spaces);
      if (mlir::failed(merge_storage(operand.value(), new_storage))) {
        return mlir::failure();
      }
      continue;
    }

    if (!isa<SairProjAnyOp, SairProjLastOp, SairFbyOp>(defining_op)) continue;
    for (ValueOperand operand : cast<SairOp>(defining_op).ValueOperands()) {
      ValueStorage new_storage =
          storage.Map(defining_op, operand.value().getDefiningOp(),
                      operand.Mapping().Inverse(), iteration_spaces);
      if (mlir::failed(merge_storage(operand.value(), new_storage))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

// Ensures that communication between the producer and the user of operand only
// occurs within the same loop iteration or along dimensions that are
// materialized in memory.
static mlir::LogicalResult VerifyCommunicationVolume(
    mlir::Location loc, const IterationSpace &use_iter_space,
    const ValueAccess &operand, const IterationSpaceAnalysis &iteration_spaces,
    const StorageAnalysis &storage_analysis) {
  const IterationSpace &def_iter_space =
      iteration_spaces.Get(operand.value.getDefiningOp());
  // Only check if loop nest are specified.
  if (!use_iter_space.fully_specified() || !def_iter_space.fully_specified()) {
    return mlir::success();
  }

  const ValueStorage &storage = storage_analysis.GetStorage(operand.value);
  // Success if storage is not yet specified.
  if (storage.layout() == nullptr) return mlir::success();

  MappingAttr communication_volume = CommunicationVolume(
      operand.mapping.size(), def_iter_space, use_iter_space);
  MappingAttr layout_to_operand =
      def_iter_space.mapping().Compose(storage.layout()).Inverse();
  MappingAttr layout_to_communication_volume =
      layout_to_operand.Compose(communication_volume).Canonicalize();

  // Check that the layout covers the sub-domain of the operand that is not
  // covered by common dimensions.
  if (layout_to_communication_volume.HasNoneExprs()) {
    mlir::InFlightDiagnostic diag =
        mlir::emitError(loc)
        << "operand storage must cover all operand dimensions "
           "that are not covered by loops common to both operand and user";
    diag.attachNote(operand.value.getDefiningOp()->getLoc())
        << "operand defined here";
    return mlir::failure();
  }

  return mlir::success();
}

// Ensures that communication between producers and users only occurs within the
// same loop iteration or along dimensions that are materialized in memory.
static mlir::LogicalResult VerifyCommunicationVolume(
    SairProgramOp program, const IterationSpaceAnalysis &iteration_spaces,
    const StorageAnalysis &storage_analysis) {
  // Ensure that values storage have enough dimensions.
  auto result = program.walk([&](SairOp op) -> mlir::WalkResult {
    const IterationSpace iter_space = iteration_spaces.Get(op);
    // Check dependencies for value operands.
    for (ValueOperand operand : op.ValueOperands()) {
      if (mlir::failed(
              VerifyCommunicationVolume(op.getLoc(), iter_space, operand.Get(),
                                        iteration_spaces, storage_analysis))) {
        return mlir::failure();
      }
    }
    // Check dependencies for domain dimensions.
    int domain_size = op.domain().size();
    for (int i = 0; i < domain_size; ++i) {
      auto dim_op = cast<SairOp>(op.domain()[i].getDefiningOp());
      const DomainShapeDim &shape_dim = op.shape().Dimension(i);
      MappingAttr dim_mapping =
          shape_dim.dependency_mapping().ResizeUseDomain(domain_size);
      for (ValueOperand operand : dim_op.ValueOperands()) {
        ValueAccess access = operand.Get();
        access.mapping = dim_mapping.Compose(access.mapping);
        if (mlir::failed(VerifyCommunicationVolume(
                op.getLoc(), iter_space, operand.Get(), iteration_spaces,
                storage_analysis))) {
          return mlir::failure();
        }
      }
    }
    return mlir::success();
  });
  return mlir::failure(result.wasInterrupted());
}

mlir::LogicalResult VerifyStorages(
    SairProgramOp program, const IterationSpaceAnalysis &iteration_spaces) {
  // Check storage attributes are well-formed.
  mlir::WalkResult result = program.walk([](ComputeOp op) -> mlir::WalkResult {
    return VerifyStorageAttrWellFormed(op);
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Ensure storage attributes are compatibles with each other.
  auto analysis_result = StorageAnalysis::Create(program);
  if (!analysis_result.has_value()) return mlir::failure();
  StorageAnalysis analysis = std::move(analysis_result).value();

  // Ensure that operation updating a buffers in place use the same layout for
  // both inputs and outputs.
  result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    for (mlir::Value result : op->getResults()) {
      const ValueStorage &result_storage = analysis.GetStorage(result);
      if (result_storage.buffer_name() == nullptr) continue;
      auto sair_op = cast<SairOp>(op.getOperation());
      for (ValueOperand operand : sair_op.ValueOperands()) {
        const ValueStorage &operand_storage =
            analysis.GetStorage(operand.value());
        if (operand_storage.buffer_name() != result_storage.buffer_name()) {
          continue;
        }
        ValueStorage mapped_storage =
            operand_storage.Map(operand, iteration_spaces);
        if (mapped_storage.layout() != result_storage.layout()) {
          return op.emitError()
                 << "in-place update of buffer " << result_storage.buffer_name()
                 << " must use the same layout in input and output ("
                 << mapped_storage.layout() << " vs " << result_storage.layout()
                 << ")";
        }
      }
    }
    return mlir::success();
  });
  if (result.wasInterrupted()) return mlir::failure();

  // TODO(b/174127497): make sure that value is not ovewritten by another write.
  return VerifyCommunicationVolume(program, iteration_spaces, analysis);
}

BufferAttr GetRegister0DBuffer(mlir::MLIRContext *context) {
  auto *sair_dialect = context->getLoadedDialect<SairDialect>();
  return BufferAttr::get(/*space=*/sair_dialect->register_attr(),
                         /*name=*/nullptr,
                         /*layout=*/NamedMappingAttr::GetIdentity(context, {}),
                         context);
}

bool operator==(const ValueStorage &lhs, const ValueStorage &rhs) {
  return lhs.space() == rhs.space() && lhs.buffer_name() == rhs.buffer_name() &&
         lhs.layout() == rhs.layout();
}

bool operator!=(const ValueStorage &lhs, const ValueStorage &rhs) {
  return !(lhs == rhs);
}

mlir::LogicalResult ValueStorage::MergeSpace(mlir::StringAttr new_space) {
  if (new_space == nullptr) return mlir::success();
  if (space_ == nullptr) space_ = new_space;
  return mlir::success(space_ == new_space);
}

mlir::LogicalResult ValueStorage::MergeBufferName(mlir::StringAttr new_name) {
  if (new_name == nullptr) return mlir::success();
  if (buffer_name_ == nullptr) buffer_name_ = new_name;
  return mlir::success(buffer_name_ == new_name);
}

mlir::LogicalResult ValueStorage::MergeLayout(MappingAttr new_layout) {
  if (new_layout == nullptr) return mlir::success();
  if (layout_ == nullptr) layout_ = new_layout;
  return mlir::success(layout_ == new_layout);
}

ValueStorage ValueStorage::Map(
    const ValueOperand &operand,
    const IterationSpaceAnalysis &iteration_spaces) const {
  return Map(operand.value().getDefiningOp(), operand.getOwner(),
             operand.Mapping(), iteration_spaces);
}

ValueStorage ValueStorage::Map(
    SairOp from, SairOp to, MappingAttr mapping,
    const IterationSpaceAnalysis &iteration_spaces) const {
  MappingAttr layout;
  if (layout_ != nullptr) {
    // We need to resize mapping to match operations domain size as values may
    // have a smaller rank than the operations that creates them.
    MappingAttr domain_mapping = mapping.Resize(from.domain().size())
                                     .ResizeUseDomain(to.domain().size());
    MappingAttr iter_space_mapping =
        iteration_spaces.TranslateMapping(to, from, domain_mapping);
    assert(iter_space_mapping != nullptr);
    layout = iter_space_mapping.Compose(layout_).Canonicalize();
  }
  return ValueStorage(space_, buffer_name_, layout);
}

MappingAttr CommunicationVolume(int value_rank,
                                const IterationSpace &def_iter_space,
                                const IterationSpace &use_iter_space) {
  int num_common_loops = def_iter_space.NumCommonLoops(use_iter_space);

  // Mapping from the domain of the operand to common loops.
  MappingAttr domain_to_common_loops = def_iter_space.mapping()
                                           .ResizeUseDomain(value_rank)
                                           .Resize(num_common_loops);
  // Extend `domain_to_common_loops` to cover the full operand domain then drop
  // common loops. This gives a mapping that only covers the sub-domain of the
  // operand that is not covered by common loops.
  return domain_to_common_loops.Inverse().MakeSurjective().Inverse().DropFront(
      num_common_loops);
}

}  // namespace sair
