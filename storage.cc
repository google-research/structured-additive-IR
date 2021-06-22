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
#include "sequence.h"

namespace sair {

// Returns the layout of from_memref or to_memref operation value.
static MappingAttr FromToMemRefLayout(FromToMemRefOp op,
                                      const IterationSpace &iter_space) {
  int rank = op.memref_domain().size();
  int parallel_domain_size = op.parallel_domain().size();
  auto domain_to_layout = MappingAttr::GetIdentity(op.getContext(), rank)
                              .ShiftRight(parallel_domain_size);
  return iter_space.mapping().Inverse().Compose(domain_to_layout);
}

Buffer::Buffer(mlir::Location loc, mlir::StringAttr name,
               mlir::Type element_type, const LoopNest &loop_nest)
    : MappedDomain(loc, "buffer", name, loop_nest),
      element_type_(element_type) {
  assert(element_type != nullptr);
}

Buffer::Buffer(FromToMemRefOp import_op, mlir::StringAttr name,
               const LoopNest &loop_nest)
    : Buffer(import_op.getLoc(), name, import_op.MemRefType().getElementType(),
             loop_nest) {
  import_op_ = import_op;
}

void Buffer::AddValue(mlir::Value value) {
  values_.push_back(value);
  auto defining_op = dyn_cast<ComputeOp>(value.getDefiningOp());
  if (defining_op != nullptr) {
    int position = value.cast<OpResult>().getResultNumber();
    writes_.emplace_back(defining_op, position);
  }
  for (mlir::OpOperand &use : value.getUses()) {
    auto user = dyn_cast<ComputeOp>(use.getOwner());
    if (user == nullptr) continue;
    ValueOperand sair_operand(&use);
    reads_.emplace_back(user, sair_operand.position());
  }
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

// Returns the layout of `buffer` as a mapping from the iteration space of
// `op` to buffer dimensions.
static MappingAttr GetBufferLayout(
    ComputeOp op, BufferAttr buffer,
    const IterationSpaceAnalysis &iteration_spaces) {
  if (buffer.layout() == nullptr) return nullptr;

  mlir::MLIRContext *context = op.getContext();
  auto none_expr = MappingNoneExpr::get(context);
  const IterationSpace &iter_space = iteration_spaces.Get(op.getOperation());
  MappingAttr mapping = buffer.layout().mapping();

  llvm::SmallVector<MappingExpr> loops_to_indexed_loops_exprs(
      mapping.UseDomainSize(), none_expr);
  for (auto p : llvm::enumerate(buffer.layout().names())) {
    auto it = llvm::find(iter_space.loop_names(), p.value());
    assert(it != iter_space.loop_names().end());
    int pos = std::distance(iter_space.loop_names().begin(), it);
    loops_to_indexed_loops_exprs[p.index()] = MappingDimExpr::get(pos, context);
  }

  auto loops_to_indexed_loops = MappingAttr::get(
      context, iter_space.mapping().size(), loops_to_indexed_loops_exprs);
  return loops_to_indexed_loops.Compose(mapping);
}

// Unifies the shape of `buffer` with the shape specified by attribute
// `buffer_attr` of `op`. Raises an error if shapes cannot be unified.
static mlir::LogicalResult UnifyBufferShape(
    mlir::StringAttr buffer_name, SairOp op, MappingAttr layout,
    const IterationSpace &op_iter_space,
    const LoopFusionAnalysis &loop_analysis, Buffer &buffer) {
  mlir::MLIRContext *context = op.getContext();
  int iter_space_size = op_iter_space.mapping().size();
  LoopNest op_loop_nest = loop_analysis.GetLoopNest(op_iter_space.loop_names());
  MappingAttr loop_nest_mapping =
      op_loop_nest.DomainToLoops().Resize(buffer.loop_nest().size());

  // The operation loop nest might not cover all operation dimensions. We thus
  // define a new domain that maps to loop nest dimensions when possible and
  // directly to operation dimensions otherwise.
  llvm::SmallVector<ValueAccess> domain;
  llvm::append_range(domain, op_loop_nest.domain());

  auto none = MappingNoneExpr::get(context);
  llvm::SmallVector<MappingExpr> constraints(op.domain().size(), none);
  AssertSuccess(UnificationConstraints(
      op_iter_space.mapping(), loop_nest_mapping.Resize(iter_space_size),
      constraints));
  for (int i = 0, e = op.domain().size(); i < e; ++i) {
    if (!constraints[i].isa<MappingNoneExpr>()) continue;
    auto renaming = MappingAttr::get(context, domain.size(), constraints);
    auto mapping = op.shape().Dimension(i).dependency_mapping();
    constraints[i] = MappingDimExpr::get(domain.size(), context);
    domain.push_back({op.domain()[i], renaming.Compose(mapping)});
  }

  auto renaming = MappingAttr::get(context, domain.size(), constraints);
  auto domain_to_loops = loop_nest_mapping.ResizeUseDomain(domain.size());
  auto domain_to_iter_space =
      renaming.Compose(op_iter_space.mapping())
          .Unify(domain_to_loops.Resize(iter_space_size));
  auto domain_to_layout = domain_to_iter_space.Compose(layout).Canonicalize();
  return buffer.UnifyMapping(op.getLoc(), domain_to_loops, domain_to_layout,
                             domain);
}

// Trims `buffer` loop nest so that it can be accessed from the given iteration
// space, with the given layout. Layout is ignored if null.
static void TrimBufferLoopNestForAccess(
    const IterationSpace &iter_space, MappingAttr layout,
    const LoopFusionAnalysis &fusion_analysis, Buffer &buffer) {
  // Trims the buffer loop nest so that only common loops that are not indexed
  // by the layout remain.
  int max_loop_nest = iter_space.NumCommonLoops(buffer.loop_nest());
  if (layout != nullptr) {
    llvm::SmallBitVector indexed_loops = layout.DependencyMask();
    int first_indexed_loop = indexed_loops.find_first();
    if (first_indexed_loop >= 0 && first_indexed_loop < max_loop_nest) {
      max_loop_nest = first_indexed_loop;
    }
  }

  LoopNest new_loop_nest = fusion_analysis.GetLoopNest(
      iter_space.loop_names().take_front(max_loop_nest));
  buffer.SetLoopNest(new_loop_nest);
}

// Declares buffer `attr` in `buffer_map`. If the
// buffer is already present, ensure that rank and element type are coherent and
// trims the buffer loop nest to the common prefix with `op` loop nest.
static mlir::LogicalResult DeclareBuffer(
    ComputeOp op, int result, BufferAttr attr,
    const LoopFusionAnalysis &loop_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    llvm::DenseMap<mlir::Attribute, Buffer> &buffer_map,
    llvm::DenseSet<mlir::Attribute> &buffers_with_rank_set) {
  if (attr == nullptr || attr.name() == nullptr) return mlir::success();
  mlir::Type element_type =
      op->getResult(result).getType().cast<ValueType>().ElementType();
  auto sair_op = cast<SairOp>(op.getOperation());
  const IterationSpace &iter_space = iteration_spaces.Get(sair_op);
  const LoopNest &loop_nest =
      loop_analysis.GetLoopNest(iter_space.loop_names());
  auto it = buffer_map.try_emplace(attr.name(), op.getLoc(), attr.name(),
                                   element_type, loop_nest);
  Buffer &buffer = it.first->second;

  // Check that element types match.
  if (buffer.element_type() != element_type) {
    mlir::InFlightDiagnostic diag =
        op.emitError()
        << "buffer " << attr.name()
        << " has different element type than in previous occurence";
    diag.attachNote(buffer.location()) << "previous occurence here";
    return mlir::failure();
  }

  MappingAttr layout = GetBufferLayout(op, attr, iteration_spaces);
  TrimBufferLoopNestForAccess(iter_space, layout, loop_analysis, buffer);
  if (layout == nullptr) return mlir::success();

  // Ensure that the number of dimension is coherent.
  if (!buffers_with_rank_set.insert(attr.name()).second) {
    if (buffer.rank() != layout.size()) {
      mlir::InFlightDiagnostic diag =
          op.emitError() << "buffer " << attr.name()
                         << " rank differs from previous occurence";
      diag.attachNote(buffer.location()) << "previous occurence here";
      return mlir::failure();
    }
  } else {
    buffer.AddNonePrefixToMapping(layout.size());
  }

  return UnifyBufferShape(attr.name(), sair_op, layout, iter_space,
                          loop_analysis, buffer);
}

// Declare buffers used by `program` in `buffers`. If a buffer has multiple
// uses, chek that element type and rank are compatible.
static mlir::LogicalResult DeclareBuffers(
    SairProgramOp program, const IterationSpaceAnalysis &iteration_spaces,
    const LoopFusionAnalysis &fusion_analysis,
    llvm::DenseMap<mlir::Attribute, Buffer> &buffers) {
  llvm::DenseSet<mlir::Attribute> buffers_with_rank_set;

  // Declare external buffers imported using from/to memref.
  mlir::WalkResult result =
      program.walk([&](FromToMemRefOp op) -> mlir::WalkResult {
        auto sair_op = cast<SairOp>(op.getOperation());
        auto name = mlir::StringAttr::get(op.getContext(), op.buffer_name());
        const IterationSpace &iter_space = iteration_spaces.Get(sair_op);
        const LoopNest &loop_nest =
            fusion_analysis.GetLoopNest(iter_space.loop_names());
        auto [buffer_it, was_inserted] =
            buffers.try_emplace(name, op, name, loop_nest);
        if (!was_inserted)
          return op.emitError() << "buffer name is already used";
        MappingAttr layout = FromToMemRefLayout(op, iter_space);
        buffers_with_rank_set.insert(name);
        buffer_it->second.AddNonePrefixToMapping(layout.size());
        return UnifyBufferShape(name, sair_op, layout, iter_space,
                                fusion_analysis, buffer_it->second);
      });
  if (result.wasInterrupted()) return mlir::failure();

  // Declare internal buffers.
  result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      BufferAttr buffer_attr = op.Storage(i);
      if (mlir::failed(DeclareBuffer(op, i, buffer_attr, fusion_analysis,
                                     iteration_spaces, buffers,
                                     buffers_with_rank_set))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Ensure all buffers layout is fully specified.
  for (auto [name, buffer] : buffers) {
    if (buffer.mapping().HasNoneExprs()) {
      return buffer.EmitError() << "layout is not fully specified";
    }
  }

  return mlir::failure(result.wasInterrupted());
}

// Computes how values are stored and stores the result into `value_storages`.
mlir::LogicalResult StorageAnalysis::ComputeValueStorages(
    SairProgramOp program, const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces) {
  mlir::MLIRContext *context = program.getContext();
  auto *sair_dialect = context->getLoadedDialect<SairDialect>();
  mlir::StringAttr memory_space = sair_dialect->memory_attr();

  // Initialize storage information from compute operations.
  auto result = program.walk([&](ComputeOp op) -> mlir::WalkResult {
    for (int i = 0, e = op->getNumResults(); i < e; ++i) {
      BufferAttr buffer = op.Storage(i);
      if (buffer == nullptr) continue;
      MappingAttr layout = GetBufferLayout(op, buffer, iteration_spaces);
      ValueStorage storage(buffer.space(), buffer.name(), layout);
      if (mlir::failed(SetStorage(op->getResult(i), storage, fusion_analysis,
                                  iteration_spaces))) {
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
    return SetStorage(op.result(), storage, fusion_analysis, iteration_spaces);
  });
  if (result.wasInterrupted()) return mlir::failure();

  // Initialize from from_scalar operations.
  result = program.walk([&](SairFromScalarOp op) -> mlir::WalkResult {
    auto layout = MappingAttr::get(context, 0, {});
    ValueStorage storage(sair_dialect->register_attr(), nullptr, layout);
    return SetStorage(op.result(), storage, fusion_analysis, iteration_spaces);
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
    return SetStorage(op.value(), storage, fusion_analysis, iteration_spaces);
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
  SequenceAnalysis sequence_analysis(program);
  LoopFusionAnalysis fusion_analysis(program, &sequence_analysis);
  IterationSpaceAnalysis iteration_spaces(program);

  if (mlir::failed(DeclareBuffers(program, iteration_spaces, fusion_analysis,
                                  buffers_))) {
    return mlir::failure();
  }

  if (mlir::failed(
          ComputeValueStorages(program, fusion_analysis, iteration_spaces))) {
    return mlir::failure();
  }

  if (mlir::failed(VerifyAndMinimizeBufferLoopNests(
          fusion_analysis, iteration_spaces, sequence_analysis))) {
    return mlir::failure();
  }

  // Ensure that writes to external buffers occure after the buffer is defined.
  for (auto &[name, buffer] : buffers_) {
    if (!buffer.is_external()) continue;
    auto defining_op =
        buffer.import_op().MemRef().value().getDefiningOp<SairOp>();
    // We only need to check writes as reads always occure after writes.
    for (auto write : buffer.writes()) {
      if (sequence_analysis.IsBefore(write.first, defining_op)) {
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

void StorageAnalysis::MergeStorage(
    mlir::Value value, const ValueStorage &new_storage,
    const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces) {
  AssertSuccess(
      SetStorage(value, new_storage, fusion_analysis, iteration_spaces));
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

void StorageAnalysis::AddDimensionsToBuffer(
    mlir::StringAttr buffer_name, SairOp op,
    const IterationSpace &op_iter_space,
    const LoopFusionAnalysis &fusion_analysis, MappingAttr new_layout) {
  Buffer &buffer = buffers_.find(buffer_name)->second;
  assert(new_layout != nullptr);
  assert(new_layout.size() >= buffer.mapping().size());
  assert(!buffer.is_external());

  // Extend buffer domain.
  TrimBufferLoopNestForAccess(op_iter_space, new_layout, fusion_analysis,
                              buffer);
  int old_size = buffer.rank();
  buffer.AddNonePrefixToMapping(new_layout.size() - old_size);
  AssertSuccess(UnifyBufferShape(buffer_name, op, new_layout, op_iter_space,
                                 fusion_analysis, buffer));

  // Add a dimension to values layout.
  for (mlir::Value value : buffer.values()) {
    ValueStorage &storage = value_storages_.find(value)->second;
    storage.AddUnknownPrefixToLayout(new_layout.size() - old_size);
  }
}

// Update the storage information for value. Updates buffers to register new
// buffer uses.
static mlir::LogicalResult UpdateStorage(
    mlir::Value value, const ValueStorage &new_storage,
    const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces, ValueStorage &storage,
    llvm::DenseMap<mlir::Attribute, Buffer> &buffers) {
  if (storage.buffer_name() == nullptr &&
      new_storage.buffer_name() != nullptr) {
    Buffer &buffer = buffers.find(new_storage.buffer_name())->second;
    buffer.AddValue(value);
    // Trim buffer loop nest to ensure it can be used from value def and uses
    // iteration spaces.
    SairOp defining_op = value.getDefiningOp();
    assert(defining_op != nullptr);
    TrimBufferLoopNestForAccess(iteration_spaces.Get(defining_op), nullptr,
                                fusion_analysis, buffer);
    for (mlir::Operation *user : value.getUsers()) {
      TrimBufferLoopNestForAccess(iteration_spaces.Get(user), nullptr,
                                  fusion_analysis, buffer);
    }
  }

  if (mlir::failed(storage.MergeSpace(new_storage.space()))) {
    return value.getDefiningOp()->emitError()
           << "conflicting memory spaces: expected " << new_storage.space()
           << ", got " << storage.space();
  }
  if (mlir::failed(storage.MergeBufferName(new_storage.buffer_name()))) {
    return value.getDefiningOp()->emitError()
           << "conflicting buffer names: expected " << new_storage.buffer_name()
           << ", got " << storage.buffer_name();
  }
  MappingAttr canonical_layout;
  if (new_storage.layout() != nullptr) {
    canonical_layout = new_storage.layout().Canonicalize();
  }
  if (mlir::failed(storage.MergeLayout(canonical_layout))) {
    return value.getDefiningOp()->emitError()
           << "conflicting layouts: expected " << canonical_layout << ", got "
           << storage.layout();
  }

  return mlir::success();
}

mlir::LogicalResult StorageAnalysis::SetStorage(
    mlir::Value value, ValueStorage storage,
    const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces) {
  llvm::SmallVector<mlir::Value> work_list;

  // Merge storage information for a value with existing information. Fails and
  // emits an error in case of conflicts.
  auto update_storage = [&](mlir::Value value,
                            ValueStorage new_storage) -> mlir::LogicalResult {
    ValueStorage &storage = value_storages_[value];
    if (new_storage == storage) return mlir::success();

    work_list.push_back(value);
    return UpdateStorage(value, new_storage, fusion_analysis, iteration_spaces,
                         storage, buffers_);
  };

  if (mlir::failed(update_storage(value, storage))) return mlir::failure();

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
      if (mlir::failed(update_storage(user->getResult(result), new_storage))) {
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
      if (mlir::failed(update_storage(operand.value(), new_storage))) {
        return mlir::failure();
      }
      continue;
    }

    if (!isa<SairProjAnyOp, SairProjLastOp, SairFbyOp>(defining_op)) continue;
    for (ValueOperand operand : cast<SairOp>(defining_op).ValueOperands()) {
      ValueStorage new_storage =
          storage.Map(defining_op, operand.value().getDefiningOp(),
                      operand.Mapping().Inverse(), iteration_spaces);
      if (mlir::failed(update_storage(operand.value(), new_storage))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

// Ensures that we can insert a malloc operation for the buffer. Increases
// `min_num_loops` to make sure that a malloc operation can be inserted if
// needed.
static mlir::LogicalResult CheckMallocInsertionPoint(
    mlir::StringAttr buffer_name, const Buffer &buffer,
    const llvm::SmallBitVector &used_dimensions,
    const IterationSpaceAnalysis &iteration_spaces,
    const SequenceAnalysis &sequence_analysis, int &min_num_loops) {
  // Find the first compute op writting to the buffer.
  ComputeOp first_write = buffer.writes().front().first;
  for (auto p : buffer.writes()) {
    if (sequence_analysis.IsBefore(p.first, first_write)) {
      first_write = p.first;
    }
  }

  llvm::ArrayRef<mlir::StringAttr> write_loops =
      iteration_spaces.Get(cast<SairOp>(first_write.getOperation()))
          .loop_names();
  for (int dim : used_dimensions.set_bits()) {
    auto dimension_op =
        cast<SairOp>(buffer.domain()[dim].value.getDefiningOp());
    if (sequence_analysis.IsBefore(first_write, dimension_op)) {
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

mlir::LogicalResult StorageAnalysis::VerifyAndMinimizeBufferLoopNests(
    const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    const SequenceAnalysis &sequence_analysis) {
  for (auto &[name_attr, buffer] : buffers_) {
    mlir::StringAttr name = name_attr.cast<mlir::StringAttr>();
    MappingAttr mapping = buffer.NestedMapping();
    DomainShapeAttr domain_shape = buffer.DomainShape();

    if (mlir::failed(VerifyMappingShape(buffer, mapping, domain_shape))) {
      return mlir::failure();
    }

    DomainShapeAttr shape = domain_shape.AccessedShape(mapping);
    int rank = shape.NumDimensions();
    int min_num_loops = 0;
    for (int i = buffer.loop_nest().size(); i < rank; ++i) {
      int max_dependency =
          shape.Dimension(i).dependency_mapping().MinDomainSize();
      min_num_loops = std::max(min_num_loops, max_dependency);
    }

    if (min_num_loops > buffer.loop_nest().size()) {
      return buffer.EmitError()
             << "layout depends on loops it cannot be nested in";
    }

    // We cannot minimize external buffers loop nests.
    if (buffer.is_external()) continue;

    llvm::SmallBitVector used_dimensions = buffer.mapping().DependencyMask();
    if (mlir::failed(CheckMallocInsertionPoint(
            name, buffer, used_dimensions, iteration_spaces, sequence_analysis,
            min_num_loops))) {
      return mlir::failure();
    }

    // Minimize layout loop-nest.
    LoopNest new_loop_nest = fusion_analysis.GetLoopNest(
        buffer.loop_nest().take_front(min_num_loops));
    buffer.SetLoopNest(new_loop_nest);
  }

  return mlir::success();
}

void StorageAnalysis::CreateBuffer(
    mlir::Value value, llvm::ArrayRef<mlir::StringAttr> loop_names,
    const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces) {
  mlir::StringAttr buffer_name = GetFreshBufferName();
  mlir::Type element_type = value.getType().cast<ValueType>().ElementType();
  LoopNest loop_nest = fusion_analysis.GetLoopNest(loop_names);
  buffers_.try_emplace(buffer_name, value.getLoc(), buffer_name, element_type,
                       loop_nest);

  mlir::MLIRContext *context = value.getContext();
  auto *sair_dialect = context->getLoadedDialect<SairDialect>();

  ValueStorage storage = GetStorage(value);
  AssertSuccess(storage.MergeBufferName(buffer_name));
  AssertSuccess(storage.MergeSpace(sair_dialect->memory_attr()));
  MergeStorage(value, storage, fusion_analysis, iteration_spaces);
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

// Verifies that `buffer` is not written to by operations other that
// `allowed_write` between `from` and `to`. `allowed_write` may be null in the
// case were no write is allowed between `from` and `to`.
//
// If `from` is in loop nest [A, B] and `to` is in loop nest [A, C] where A, B
// and C are lists of loops with loops of B and C distinct, we consider that
// there is a write between `from` and `to` if any of the following condition is
// statisfied:
// * If there is a write operation between `from` and `to` operations.
// * If there is a write operation before `from` that is nested in at least one
//   loop of B and whose layout differs from `layout`. This corresponds to the
//   case where the value is produced in a loop nest and is overwritten in the
//   same loop nest.
// * If there is a write operation after `to` that is nested in at least one
//   loop of C. This corresponds to the case where the value is overwritten in
//   the loop nest where it is used.
static mlir::LogicalResult VerifyNoWriteBetween(
    mlir::StringAttr buffer_name, const Buffer &buffer,
    const ProgramPoint &from, const ProgramPoint &to, MappingAttr layout,
    ComputeOp allowed_write, const IterationSpaceAnalysis &iteration_spaces,
    const StorageAnalysis &storage_analysis,
    const SequenceAnalysis &sequence_analysis) {
  int num_common_loops = from.NumCommonLoops(to);
  for (auto [write_op, write_pos] : buffer.writes()) {
    const IterationSpace &iter_space =
        iteration_spaces.Get(cast<SairOp>(write_op.getOperation()));
    if (write_op == allowed_write) continue;

    // Check if the write occurs before `from`.
    if (sequence_analysis.IsAfter(from, write_op)) {
      int write_common_loops = iter_space.NumCommonLoops(from.loop_nest());
      if (write_common_loops <= num_common_loops) continue;
      const ValueStorage &value_storage =
          storage_analysis.GetStorage(write_op->getResult(write_pos));
      // We consider that there is no overwrite if the write if before `from`
      // and layouts are the same.
      if (layout == nullptr || value_storage.layout() == nullptr ||
          value_storage.layout().ResizeUseDomain(write_common_loops) ==
              layout.ResizeUseDomain(write_common_loops)) {
        continue;
      }
    } else if (sequence_analysis.IsBefore(to, write_op)) {
      int write_common_loops = iter_space.NumCommonLoops(to.loop_nest());
      if (write_common_loops <= num_common_loops) continue;
    }

    mlir::InFlightDiagnostic diag =
        write_op->emitError()
        << "operation overwrites a value stored in buffer " << buffer_name
        << " before it is used";
    if (from.operation() == nullptr) {
      diag.attachNote(write_op->getParentOp()->getLoc())
          << "value stored before entering sair program";
    } else {
      diag.attachNote(from.operation()->getLoc()) << "value stored here";
    }

    if (to.operation() == nullptr) {
      diag.attachNote(write_op->getParentOp()->getLoc())
          << "value used after exiting sair program";
    } else {
      diag.attachNote(to.operation()->getLoc()) << "value used here";
    }
    return mlir::failure();
  }
  return mlir::success();
}

// Verifies that `value` storage is not overwritten by an operation between the
// operation that stores the value in `buffer` and `use`.
static mlir::LogicalResult VerifyValueNotOverwritten(
    mlir::StringAttr buffer_name, const Buffer &buffer, mlir::Value value,
    ProgramPoint use, const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    const StorageAnalysis &storage_analysis,
    const SequenceAnalysis &sequence_analysis) {
  // Mark visited fby operations to avoid infinite loops.
  llvm::DenseSet<mlir::Operation *> visited_fby;
  // Allow the use to overwritte the buffer in order to support in-place
  // updates.
  ComputeOp allowed_write = use.operation();

  // Walk producers of `value` to find program points where it is stored in its
  // buffer. Maintain a work list of producers to process. For each, {value,
  // use} in the work-list, we must verify that there is no write to `buffer`
  // between `value` and `use`.
  llvm::SmallVector<std::pair<mlir::Value, ProgramPoint>> work_list;
  work_list.push_back({value, use});
  while (!work_list.empty()) {
    auto [value, use_point] = work_list.pop_back_val();
    mlir::Operation *defining_op = value.getDefiningOp();
    const IterationSpace &iter_space = iteration_spaces.Get(defining_op);

    if (auto producer = dyn_cast<ComputeOp>(defining_op)) {
      ProgramPoint def_point(producer, Direction::kAfter,
                             iter_space.loop_names());
      const ValueStorage &storage = storage_analysis.GetStorage(value);
      if (mlir::failed(VerifyNoWriteBetween(
              buffer_name, buffer, def_point, use_point, storage.layout(),
              allowed_write, iteration_spaces, storage_analysis,
              sequence_analysis))) {
        return mlir::failure();
      }
    } else if (auto proj = dyn_cast<SairProjLastOp>(defining_op)) {
      work_list.emplace_back(proj.value(), use_point);
    } else if (auto proj = dyn_cast<SairProjAnyOp>(defining_op)) {
      work_list.emplace_back(proj.value(), use_point);
    } else if (auto from_memref = dyn_cast<SairFromMemRefOp>(defining_op)) {
      MappingAttr layout = FromToMemRefLayout(from_memref, iter_space);
      ProgramPoint before_program(
          cast<SairProgramOp>(defining_op->getParentOp()), Direction::kBefore);
      if (mlir::failed(VerifyNoWriteBetween(buffer_name, buffer, before_program,
                                            use_point, layout, allowed_write,
                                            iteration_spaces, storage_analysis,
                                            sequence_analysis))) {
        return mlir::failure();
      }
    } else if (auto fby = dyn_cast<SairFbyOp>(defining_op)) {
      // Find outermost loop that iterate along fby dimensions.
      MappingAttr mapping_to_loops = iter_space.MappingToLoops();
      auto it = llvm::find_if(mapping_to_loops, [&](MappingExpr expr) {
        return expr.MinDomainSize() >= fby.parallel_domain().size();
      });
      int first_carry_loop = std::distance(mapping_to_loops.begin(), it);

      // Ensure that there is no write between init and the use. We trim
      // use_loops to remove dependency-carrying dimensions as we are only going
      // to use init at the first iteration.
      ProgramPoint init_use_point = use_point;
      if (init_use_point.loop_nest().size() > first_carry_loop &&
          init_use_point.loop_nest()[first_carry_loop] ==
              iter_space.loop_names()[first_carry_loop]) {
        init_use_point.TrimLoopNest(first_carry_loop);
      }
      work_list.emplace_back(fby.init(), init_use_point);

      // Ensure that there is no write between the value produced at the last
      // iteration of the loop nest and the end of the loop nest.
      if (!visited_fby.insert(defining_op).second) continue;

      // Case where there are no dependency-carrying dimension.
      if (first_carry_loop == iter_space.loop_names().size()) continue;

      // Ensure that there is no write between the produce of fby value and the
      // end of dependency-carrying dimensions.
      mlir::StringAttr carry_loop_name =
          iter_space.loop_names()[first_carry_loop];
      const LoopFusionClass &carry_loop_class =
          fusion_analysis.GetClass(carry_loop_name);
      work_list.emplace_back(fby.value(), carry_loop_class.EndPoint());
    } else {
      llvm_unreachable("unexpected operation");
    }
  }
  return mlir::success();
}

// Verifies that values are not overwritten by another operation before they are
// used.
mlir::LogicalResult VerifyValuesNotOverwritten(
    const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    const StorageAnalysis &storage_analysis,
    const SequenceAnalysis &sequence_analysis) {
  // Ensure that no operation is writting in buffers between the moment
  // where a value is written and the moment where a value is read.
  for (const auto &[name_attr, buffer] : storage_analysis.buffers()) {
    auto buffer_name = name_attr.cast<mlir::StringAttr>();
    for (auto [op, operand_pos] : buffer.reads()) {
      auto sair_op = cast<SairOp>(op.getOperation());
      const IterationSpace &iter_space = iteration_spaces.Get(sair_op);
      mlir::Value operand = sair_op.ValueOperands()[operand_pos].value();
      ProgramPoint use_point(op, Direction::kBefore, iter_space.loop_names());
      if (mlir::failed(VerifyValueNotOverwritten(
              buffer_name, buffer, operand, use_point, fusion_analysis,
              iteration_spaces, storage_analysis, sequence_analysis))) {
        return mlir::failure();
      }
    }

    // If the buffer is used in a to_memref operation, ensure that the output is
    // not overwritten.
    if (!buffer.is_external()) continue;
    auto to_memref =
        dyn_cast<SairToMemRefOp>(buffer.import_op().getOperation());
    if (to_memref == nullptr) continue;
    ProgramPoint after_program(cast<SairProgramOp>(to_memref->getParentOp()),
                               Direction::kAfter);
    if (mlir::failed(VerifyValueNotOverwritten(
            buffer_name, buffer, to_memref.value(), after_program,
            fusion_analysis, iteration_spaces, storage_analysis,
            sequence_analysis))) {
      return mlir::failure();
    }
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
    SairProgramOp program, const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    const SequenceAnalysis &sequence_analysis) {
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

  if (mlir::failed(
          VerifyCommunicationVolume(program, iteration_spaces, analysis))) {
    return mlir::failure();
  }
  return VerifyValuesNotOverwritten(fusion_analysis, iteration_spaces, analysis,
                                    sequence_analysis);
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
  if (layout_ == nullptr) {
    layout_ = new_layout;
    return mlir::success();
  }

  new_layout = new_layout.UnifyUnknownExprs(layout_);
  if (new_layout == nullptr) return mlir::failure();
  layout_ = new_layout;
  return mlir::success();
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

void ValueStorage::AddUnknownPrefixToLayout(int num_new_dims) {
  assert(layout_ != nullptr);
  assert(num_new_dims >= 0);
  mlir::MLIRContext *context = layout_.getContext();
  llvm::SmallVector<MappingExpr> prefix(num_new_dims,
                                        MappingUnknownExpr::get(context));
  layout_ = layout_.AddPrefix(prefix);
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
