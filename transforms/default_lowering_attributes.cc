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

#include "transforms/default_lowering_attributes.h"

#include <iterator>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sequence.h"
#include "storage.h"

namespace sair {
namespace {

// Include passes base class declaration generated by MLIR. The #define in front
// selects the part of the file to include (pass base class declaration or pass
// registration). See
// https://mlir.llvm.org/docs/PassManagement/#declarative-pass-specification for
// more information.
#define GEN_PASS_DEF_DEFAULTEXPANSIONPASS
#define GEN_PASS_DEF_DEFAULTINSTANCEPASS
#define GEN_PASS_DEF_DEFAULTLOOPNESTPASS
#define GEN_PASS_DEF_DEFAULTSEQUENCEPASS
#define GEN_PASS_DEF_DEFAULTSTORAGEPASS
#include "transforms/default_lowering_attributes.h.inc"

// Creates a blank instance for ComputeOp with no instances.
class DefaultInstance : public impl::DefaultInstancePassBase<DefaultInstance> {
 public:
  void runOnOperation() {
    getOperation().walk([](SairOp op) {
      if (op.NumInstances() > 0) return;
      op.AddInstance(DecisionsAttr::get(nullptr, nullptr, nullptr, nullptr,
                                        nullptr, nullptr, op.getContext()));
    });
  }
};

// Writes the storage information infered by the storage analysis pass to
// Compute operations.
mlir::LogicalResult CommitStorage(
    ComputeOpInstance &op, const IterationSpaceAnalysis &iteration_spaces,
    const StorageAnalysis &storage_analysis) {
  mlir::MLIRContext *context = op.context();
  const IterationSpace &iter_space = iteration_spaces.Get(op);

  for (int i = 0, e = op.num_results(); i < e; ++i) {
    const ValueStorage &storage = storage_analysis.GetStorage(op.Result(i));

    NamedMappingAttr layout;
    if (storage.layout() != nullptr) {
      llvm::SmallBitVector indexed_loops = storage.layout().DependencyMask();
      auto none = MappingNoneExpr::get(context);
      llvm::SmallVector<MappingExpr> renaming(iter_space.mapping().size(),
                                              none);
      llvm::SmallVector<mlir::StringAttr> loop_names;
      for (int loop : indexed_loops.set_bits()) {
        renaming[loop] = MappingDimExpr::get(loop_names.size(), context);
        loop_names.push_back(iter_space.loop_names()[loop]);
      }

      layout = NamedMappingAttr::get(loop_names, renaming, context)
                   .Compose(storage.layout());
    }
    auto attr = BufferAttr::get(storage.space(), storage.buffer_name(), layout,
                                context);
    op.SetStorage(i, attr);
  }
  return mlir::success();
}

// Indicates if an operand can use the value from registers. The operand value
// must be defined.
bool FitsInRegisters(const OperandInstance &operand,
                     const IterationSpaceAnalysis &iteration_spaces) {
  OpInstance defining_op = operand.GetValue()->defining_op();
  MappingAttr mapping = iteration_spaces.TranslateMapping(
      operand.owner(), defining_op,
      operand.Mapping().Resize(defining_op.domain_size()));
  int common_loops = iteration_spaces.Get(operand.owner())
                         .NumCommonLoops(iteration_spaces.Get(defining_op));
  // Test if the operand is only accessed along common loops.
  return mapping.MinDomainSize() <= common_loops;
}

// Initialize storage for value with default values if needed. Memory space is
// initialized with `register` and layout is initialized with `?`
// expressions.
void InitializeStorage(ResultInstance value,
                       const LoopFusionAnalysis &fusion_analysis,
                       const IterationSpaceAnalysis &iteration_spaces,
                       StorageAnalysis &storage_analysis) {
  OpInstance defining_op = value.defining_op();
  mlir::MLIRContext *context = defining_op.context();
  SairDialect *sair_dialect = defining_op.GetSairDialect();
  ValueStorage storage = storage_analysis.GetStorage(value);

  // Set memory space to register.
  if (storage.space() == nullptr) {
    AssertSuccess(storage.MergeSpace(sair_dialect->register_attr()));
  }

  // Initialize layout.
  if (storage.layout() == nullptr) {
    int num_dimensions = 0;
    if (storage.buffer_name() != nullptr) {
      const Buffer &buffer = storage_analysis.GetBuffer(storage.buffer_name());
      num_dimensions = buffer.rank();
    }
    const IterationSpace &iter_space = iteration_spaces.Get(defining_op);

    auto unknown_expr = MappingUnknownExpr::get(context);
    llvm::SmallVector<MappingExpr> exprs(num_dimensions, unknown_expr);
    auto layout = MappingAttr::get(context, iter_space.mapping().size(), exprs);
    AssertSuccess(storage.MergeLayout(layout));
  }
  storage_analysis.MergeStorage(value, storage, fusion_analysis,
                                iteration_spaces);
}

// Adds new dimensions to the operand value layout so that the operand has
// access to the data it needs.
mlir::LogicalResult ExtendLayout(const OperandInstance &operand,
                                 const IterationSpaceAnalysis &iteration_spaces,
                                 const LoopFusionAnalysis &fusion_analysis,
                                 StorageAnalysis &storage_analysis) {
  mlir::MLIRContext *context = operand.owner().context();
  auto value = operand.GetValue();
  if (!value.has_value()) return mlir::success();
  const ValueStorage &storage = storage_analysis.GetStorage(*value);
  OpInstance defining_op = value->defining_op();
  const IterationSpace &def_iter_space = iteration_spaces.Get(defining_op);
  const IterationSpace &use_iter_space = iteration_spaces.Get(operand.owner());

  // Check what dimensions of communication volume are covered by the layout.
  int operand_rank = operand.Mapping().size();
  MappingAttr communication_volume =
      CommunicationVolume(operand_rank, def_iter_space, use_iter_space);

  MappingAttr layout_to_operand =
      def_iter_space.mapping().Compose(storage.layout()).Inverse();
  MappingAttr layout_to_communication_volume =
      layout_to_operand.Compose(communication_volume);

  if (layout_to_communication_volume.IsSurjective()) return mlir::success();

  assert(storage.buffer_name() != nullptr &&
         "-default-storage-attribute pass should have added buffer names "
         "before reaching this point.");
  const Buffer &buffer = storage_analysis.GetBuffer(storage.buffer_name());
  if (buffer.is_external()) {
    return defining_op.EmitError()
           << "specifying value layout would require to increase the rank of "
              "an external buffer";
  }

  // Extend layout to cover comunication volume and permute dimensions so that
  // new dimensions are in front of the domain.
  MappingAttr extended_layout = layout_to_communication_volume.MakeSurjective();
  int num_new_dims = extended_layout.UseDomainSize() - buffer.rank();
  auto new_dims_identity = MappingAttr::GetIdentity(context, num_new_dims);
  auto permutation = MappingAttr::GetIdentity(context, buffer.rank())
                         .ShiftRight(num_new_dims)
                         .AddSuffix(new_dims_identity.Dimensions());
  extended_layout = permutation.Compose(extended_layout);

  // Unify extended_layout with the old layout as some mapping expressions of
  // the old mapping will not appear in the extended one if they do not map to
  // dimensions of communication_volume.
  auto none = MappingNoneExpr::get(context);
  llvm::SmallVector<MappingExpr> none_exprs(num_new_dims, none);
  MappingAttr extended_old_layout = storage.layout().AddPrefix(none_exprs);
  MappingAttr new_layout = def_iter_space.mapping()
                               .Inverse()
                               .Compose(communication_volume)
                               .Compose(extended_layout.Inverse())
                               .Unify(extended_old_layout);
  storage_analysis.AddDimensionsToBuffer(storage.buffer_name(), defining_op,
                                         def_iter_space, fusion_analysis,
                                         new_layout);

  // Set the value layout.
  ValueStorage new_storage = storage;
  AssertSuccess(new_storage.MergeLayout(new_layout));
  storage_analysis.MergeStorage(*value, new_storage, fusion_analysis,
                                iteration_spaces);
  return mlir::success();
}

// Converts unknown expressions from value layout to `none` expressions.
void MakeLayoutFullySpecified(const ResultInstance &value,
                              const LoopFusionAnalysis &fusion_analysis,
                              const IterationSpaceAnalysis &iteration_spaces,
                              StorageAnalysis &storage_analysis) {
  ValueStorage storage = storage_analysis.GetStorage(value);
  AssertSuccess(storage.MergeLayout(storage.layout().MakeFullySpecified()));
  storage_analysis.MergeStorage(value, storage, fusion_analysis,
                                iteration_spaces);
}

// Assings a buffer name to the operand if it cannot fit in registers.
static mlir::LogicalResult CreateBufferIfNeeded(
    const OperandInstance &operand, const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    StorageAnalysis &storage_analysis) {
  auto value = operand.GetValue();
  if (!value.has_value()) return mlir::success();
  const ValueStorage &storage = storage_analysis.GetStorage(*value);
  if (storage.space() != nullptr) return mlir::success();
  if (FitsInRegisters(operand, iteration_spaces)) return mlir::success();
  mlir::Type element_type =
      llvm::cast<ValueType>(value->GetType()).ElementType();
  if (llvm::isa<mlir::IndexType>(element_type)) {
    return value->defining_op().EmitError()
           << "cannot generate default storage for multi-dimensional index "
              "values";
  }

  const IterationSpace iter_space = iteration_spaces.Get(operand.owner());
  storage_analysis.CreateBuffer(*value, iter_space.loop_names(),
                                fusion_analysis, iteration_spaces);
  return mlir::success();
}

// Assigns the default storage to sair values. This uses registers when possible
// and materializes the minimum amount of dimensions in RAM otherwise. Fails if
// the sub-domain of dimensions to materialize is a dependent domain.
class DefaultStorage : public impl::DefaultStoragePassBase<DefaultStorage> {
 public:
  void runOnOperation() override {
    auto result = getOperation().walk([&](SairProgramOp program) {
      return program.TryWalkComputeOpInstances(
          [](const ComputeOpInstance &op) -> mlir::WalkResult {
            DecisionsAttr decisions = op.GetDecisions();
            if (decisions.loop_nest() == nullptr) {
              return op.EmitError() << "expected a loop-nest attribute";
            }
            return mlir::success();
          });
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    getOperation().walk([&](SairProgramOp program) -> mlir::WalkResult {
      mlir::LogicalResult result = RunOnProgram(program);
      if (mlir::failed(result)) signalPassFailure();
      return result;
    });
  }

 private:
  mlir::LogicalResult RunOnProgram(SairProgramOp program) {
    auto &iteration_spaces = getChildAnalysis<IterationSpaceAnalysis>(program);
    auto &fusion_analysis = getChildAnalysis<LoopFusionAnalysis>(program);
    auto &storage_analysis = getChildAnalysis<StorageAnalysis>(program);
    auto &sequence_analysis = getChildAnalysis<SequenceAnalysis>(program);

    // Assign memory space and buffer names to values that won't fit in
    // register.
    auto result = program.TryWalkOpInstances(
        [&](const OpInstance &op) -> mlir::WalkResult {
          for (OperandInstance operand : op.Operands()) {
            if (mlir::failed(CreateBufferIfNeeded(operand, fusion_analysis,
                                                  iteration_spaces,
                                                  storage_analysis))) {
              return mlir::failure();
            }
          }
          return mlir::success();
        });
    if (result.wasInterrupted()) return mlir::failure();

    // Assign all remaining values to register and intialize layout fields.
    program.WalkOpInstances([&](const OpInstance &op) {
      for (ResultInstance value : op.Results()) {
        if (!llvm::isa<ValueType>(value.GetType())) continue;
        InitializeStorage(value, fusion_analysis, iteration_spaces,
                          storage_analysis);
      }
    });

    // Add layout dimensions when necessary.
    result = program.TryWalkOpInstances(
        [&](const OpInstance &op) -> mlir::WalkResult {
          for (OperandInstance operand : op.Operands()) {
            if (mlir::failed(ExtendLayout(operand, iteration_spaces,
                                          fusion_analysis, storage_analysis))) {
              return mlir::failure();
            }
          }
          return mlir::success();
        });
    if (result.wasInterrupted()) return mlir::failure();

    // Convert unknown expressions to none expressions. Unknown expressions
    // occure when adding dimensions to buffers. When the buffer is used in
    // multiple places, only the place where the dimension is added will have
    // the layout set for the new dimensions and other places will be unknown.
    program.WalkOpInstances([&](const OpInstance &op) {
      for (ResultInstance value : op.Results()) {
        if (!llvm::isa<ValueType>(value.GetType())) continue;
        MakeLayoutFullySpecified(value, fusion_analysis, iteration_spaces,
                                 storage_analysis);
      }
    });

    if (mlir::failed(storage_analysis.VerifyAndMinimizeBufferLoopNests(
            fusion_analysis, iteration_spaces, sequence_analysis)) ||
        mlir::failed(
            VerifyValuesNotOverwritten(fusion_analysis, iteration_spaces,
                                       storage_analysis, sequence_analysis))) {
      return program.emitError()
             << "unable to generate storage attributes, see other "
                "errors for more information";
    }

    // Commit storage decisions.
    result = program.TryWalkComputeOpInstances([&](ComputeOpInstance &op)
                                                   -> mlir::WalkResult {
      if (mlir::failed(CommitStorage(op, iteration_spaces, storage_analysis))) {
        return mlir::failure();
      }
      return mlir::success();
    });
    return mlir::failure(result.wasInterrupted());
  }
};

// Generates the default `loop_nest` attribute for an operation with the given
// number of dimensions. The loop nest will start with the given prefix.
mlir::ArrayAttr GetDefaultLoopNest(int num_dimensions,
                                   llvm::ArrayRef<mlir::Attribute> prefix,
                                   LoopFusionAnalysis &fusion_analysis) {
  mlir::MLIRContext *context = fusion_analysis.getContext();
  llvm::SmallVector<MappingExpr, 4> iter_exprs;
  for (mlir::Attribute attr : prefix) {
    LoopAttr loop = llvm::cast<LoopAttr>(attr);
    iter_exprs.push_back(loop.iter());
  }

  // Inverse iter expressions and complete the resulting mapping by
  // allocating new loops. Then inverse again to obtain loop iterators.
  MappingAttr partial_inverse =
      MappingAttr::get(context, num_dimensions, iter_exprs).Inverse();
  MappingAttr full_inverse = partial_inverse.MakeSurjective();
  MappingAttr new_iter_exprs = full_inverse.Inverse();

  llvm::SmallVector<mlir::Attribute, 8> loop_nest(prefix.begin(), prefix.end());
  for (MappingExpr expr :
       new_iter_exprs.Dimensions().drop_front(prefix.size())) {
    mlir::StringAttr name = fusion_analysis.GetFreshLoopName();
    loop_nest.push_back(LoopAttr::get(name, expr, /*unroll=*/{}, context));
  }

  return mlir::ArrayAttr::get(context, loop_nest);
}

// Sets the `loop_nest` attribute to its default value. The default loop nest
// iterates over each dimension of the domain, in order, without
// rematerialization or strip-mining.
class DefaultLoopNest : public impl::DefaultLoopNestPassBase<DefaultLoopNest> {
 public:
  void runOnOperation() override {
    getOperation().walk([&](SairProgramOp program) {
      auto &fusion_analysis = getChildAnalysis<LoopFusionAnalysis>(program);
      program.WalkComputeOpInstances([&](ComputeOpInstance &op) {
        DecisionsAttr decisions = op.GetDecisions();
        if (decisions.loop_nest() != nullptr) return;
        int num_dimensions = op.domain_size();
        op.SetLoopNest(GetDefaultLoopNest(num_dimensions, {}, fusion_analysis));
      });
    });
  }
};

// Modifies the "sequence" attribute of all compute ops in the given program to
// be the canonical sequence value inferred from use-def dependencies of Sair
// values and available sequence attributes. The relative order is preserved but
// not the absolute sequence numbers. The traversal order is deterministic but
// otherwise unspecified for operations that do not have "sequence" attribute
// and belong to different connected components of the use-def dependency graph.
void UpdateSequence(SairProgramOp program) {
  SequenceAnalysis(program).AssignInferred();
}

class DefaultSequencePass
    : public impl::DefaultSequencePassBase<DefaultSequencePass> {
 public:
  void runOnOperation() override {
    getOperation().walk(
        [](SairProgramOp program_op) { UpdateSequence(program_op); });
  }
};

// Sets the expansion field of op to a default scalar
// expansion pattern implementing the operation.
static mlir::LogicalResult SetDefaultExpansion(ComputeOpInstance &op) {
  DecisionsAttr decisions = op.GetDecisions();
  if (decisions.expansion() != nullptr) return mlir::success();

  llvm::StringRef pattern_name;
  if (op.is_copy()) {
    pattern_name = kCopyExpansionPattern;
  } else {
    mlir::Operation *operation = op.GetDuplicatedOp();
    if (isa<SairMapReduceOp>(operation)) {
      return op.EmitError()
             << "map_reduce is not supported by sair-assign-default-expansion";
    }

    pattern_name =
        llvm::TypeSwitch<mlir::Operation *, llvm::StringRef>(operation)
            .Case<SairCopyOp>([](auto) { return kCopyExpansionPattern; })
            .Case<SairMapOp>([](auto) { return kMapExpansionPattern; })
            .Case<SairAllocOp>([](auto) { return kAllocExpansionPattern; })
            .Case<SairFreeOp>([](auto) { return kFreeExpansionPattern; })
            .Case<SairLoadFromMemRefOp>(
                [](auto) { return kLoadExpansionPattern; })
            .Case<SairStoreToMemRefOp>(
                [](auto) { return kStoreExpansionPattern; });
  }

  mlir::MLIRContext *context = decisions.getContext();
  op.SetDecisions(DecisionsAttr::get(
      decisions.sequence(), decisions.loop_nest(), decisions.storage(),
      mlir::StringAttr::get(context, pattern_name), decisions.copy_of(),
      decisions.operands(), context));
  return mlir::success();
}

// Sets the `expansion` attribute of compute operations to a default scalar
// expansion pattern implementing the operation.
class DefaultExpansion
    : public impl::DefaultExpansionPassBase<DefaultExpansion> {
 public:
  void runOnOperation() override {
    auto result = getOperation().walk([&](SairProgramOp program) {
      return program.TryWalkComputeOpInstances(
          [&](ComputeOpInstance &op) -> mlir::WalkResult {
            return SetDefaultExpansion(op);
          });
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateDefaultInstancePass() {
  return std::make_unique<DefaultInstance>();
}

std::unique_ptr<mlir::Pass> CreateDefaultLoopNestPass() {
  return std::make_unique<DefaultLoopNest>();
}

std::unique_ptr<mlir::Pass> CreateDefaultSequencePass() {
  return std::make_unique<DefaultSequencePass>();
}

std::unique_ptr<mlir::Pass> CreateDefaultStoragePass() {
  return std::make_unique<DefaultStorage>();
}

std::unique_ptr<mlir::Pass> CreateDefaultExpansionPass() {
  return std::make_unique<DefaultExpansion>();
}

void CreateDefaultLoweringAttributesPipeline(mlir::OpPassManager *pm) {
  pm->addPass(CreateDefaultInstancePass());
  pm->addPass(CreateDefaultSequencePass());
  pm->addPass(CreateDefaultLoopNestPass());
  pm->addPass(CreateDefaultStoragePass());
  pm->addPass(CreateDefaultExpansionPass());
}

}  // namespace sair
