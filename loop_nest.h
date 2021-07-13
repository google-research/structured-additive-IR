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

#ifndef SAIR_LOOP_NEST_H_
#define SAIR_LOOP_NEST_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mapped_domain.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sequence.h"

namespace sair {

// Indicates how an operation and the data it produces is distributed accross
// loop nest iterations. As opposed to loop nests, iteration spaces are defined
// even for operations that are not ComputeOps.
//
// Maps the domain of an operation to a domain where the first `num_loops` are
// the loops the operation belongs to. Because loops may not cover the full
// domain, the mapping may have more dimensions than the number of loops.
//
// Consider for example following Sair code.
// ```
// %2 = sair.copy[d0:%0] %1 {
//   loop_nest = [{name = "A", iter = #sair.iter<d0>}]
// } : !sair.value<d0:range, memref<f32>>
// %3 = sair.from_memref[d0:%0, d1:%0] %2 memref
//  : #sair.shape<d0:range x d1:range>, memref<f32>
// ```
// %3 is a 2D operation nested in loop A. Its iteration space will be a 2D
// domain where the first dimension corresponds to loop A.
class IterationSpace {
 public:
  // Infers the iteration space of the operation given loop names and a mapping
  // from the operation domain to loops.
  IterationSpace(llvm::SmallVector<mlir::StringAttr> loop_names,
                 MappingAttr domain_to_loops, bool fully_specified);

  // Names of the loops.
  llvm::ArrayRef<mlir::StringAttr> loop_names() const { return loop_names_; }

  // Number of loops in the iteration space.
  int num_loops() const { return loop_names_.size(); }

  // Mapping from the operation domain to the iteration space.
  MappingAttr mapping() const { return mapping_; }

  // Indicates if the loop nest is fully specified or not.
  bool fully_specified() const { return fully_specified_; }

  // Mapping from operation domain to loops.
  MappingAttr MappingToLoops() const { return mapping_.Resize(num_loops()); }

  // Returns the number of common loops between this iteration space and
  // another.
  int NumCommonLoops(const IterationSpace &other) const;
  int NumCommonLoops(llvm::ArrayRef<mlir::StringAttr> other) const;

 private:
  llvm::SmallVector<mlir::StringAttr> loop_names_;
  MappingAttr mapping_;
  bool fully_specified_;
};

// Compute iteration spaces for each operation and value.
class IterationSpaceAnalysis {
 public:
  explicit IterationSpaceAnalysis(SairProgramOp program_op);
  explicit IterationSpaceAnalysis(mlir::Operation *operation)
      : IterationSpaceAnalysis(dyn_cast<SairProgramOp>(operation)) {}

  // Computes or retrieves the loops `op` is nested in. Returns the empty
  // iteration space if the loop nest is left unspecified.
  const IterationSpace &Get(SairOp op) const;

  // Translates a mapping from the domain of `from` to the domain of `to` into a
  // mapping from the iteration space of `from` to the iteration space of `to`.
  // Maps common loops with the identity function.
  //
  // The try version returns nullptr if common loops cannot be mapped with
  // identity while the non-try version fails.
  MappingAttr TranslateMapping(SairOp from, SairOp to, MappingAttr map) const;
  MappingAttr TryTranslateMapping(SairOp from, SairOp to,
                                  MappingAttr map) const;

 private:
  // Computes the iteration space for the given operation.
  const IterationSpace &ComputeIterationSpace(mlir::Operation *operation);

  llvm::DenseMap<mlir::Operation *, IterationSpace> iteration_space_;
};

// A class of fused loops.
class LoopFusionClass : public MappedDomain {
 public:
  // Builds an empty loop fusion class for the inner-most loop of loop_nest.
  LoopFusionClass(mlir::StringAttr name, ComputeOp op,
                  const LoopNest &loop_nest);

  // Registers an operation nested in the loop.
  void AddUse(ComputeOp op, const SequenceAnalysis &sequence_analysis);

  // Program point at which the loop ends.
  ProgramPoint EndPoint() const;

  // Reduces the number of dependencies.
  void TrimDependencies(int num_dependencies);

  // Returns the unroll factor of the loop, zero if no unrolling is specified.
  unsigned unroll_factor() const { return unroll_factor_; }

  // Returns the attribute containing the unroll factor suitable for
  // constructing a loop nest attribute.
  mlir::IntegerAttr GetUnrollAttr(mlir::MLIRContext &context) const;

 private:
  // Last loop of the loop nest this loop depends on.
  int num_dependencies_;
  llvm::SmallVector<ValueAccess> domain_;

  ComputeOp last_op_;

  // Unroll factor of the (current) loop.
  unsigned unroll_factor_;
};

// A loop nest of fused loops.
class LoopNest {
 public:
  // Creates an empty loop nest.
  LoopNest(mlir::MLIRContext *context) : context_(context) {}

  // Creates a loop nest given the fusion class of the inner-most loop.
  LoopNest(const LoopFusionClass *fusion_class)
      : context_(fusion_class->mapping().getContext()),
        fusion_class_(fusion_class) {}

  // Number of loops in the loop nest.
  int size() const;

  // Indicates if the loop nest contains no loop.
  bool empty() const { return fusion_class_ == nullptr; }

  // Domain used to define loop ranges.
  llvm::ArrayRef<ValueAccess> domain() const;

  // Mapping from domain to loops.
  MappingAttr DomainToLoops() const;

  // Name of the loops in the loop nest.
  llvm::SmallVector<mlir::StringAttr> LoopNames() const;

  // Shape of the loop nest.
  DomainShapeAttr Shape() const;

  // Shape of the nest, normalized so that dependencies between dimensions are
  // identity mappings.
  DomainShapeAttr NormalizedShape() const;

 private:
  mlir::MLIRContext *context_;
  const LoopFusionClass *fusion_class_ = nullptr;
};

// Computes loop fusion classes in a sair program.
class LoopFusionAnalysis {
 public:
  // Builds an analysis populated with all loops appearing in `program_op`. Uses
  // `sequence_analysis` to reason about relative position of operations.
  explicit LoopFusionAnalysis(
      mlir::Operation *operation,
      const SequenceAnalysis *sequence_analysis = nullptr);

  // Creates a LoopFusionAnalysis populated with the loops appearing in
  // `program_op`. Returns `nullopt` if the analysis fails.
  static std::optional<LoopFusionAnalysis> Create(
      SairProgramOp program_op, const SequenceAnalysis &sequence_analysis);

  // Retrieves the fusion class with the given name.
  const LoopFusionClass &GetClass(mlir::StringAttr name) const {
    return fusion_classes_.find(name)->second;
  }

  // Retrives the unified loop nest corresponding to loops.
  LoopNest GetLoopNest(llvm::ArrayRef<mlir::StringAttr> loop_names) const;

  // Generates a fresh loop name. May be called multiple times without
  // invalidating the analysis.
  mlir::StringAttr GetFreshLoopName();

  // Returns the analysis context.
  mlir::MLIRContext *getContext() const { return context_; }

 private:
  LoopFusionAnalysis(mlir::MLIRContext *context) : context_(context) {}

  // Populates the analysis with the operations appearing in `program_op`. Uses
  // `sequence_analysis` to reason about relative position of operations.
  mlir::LogicalResult Init(SairProgramOp program_op,
                           const SequenceAnalysis &sequence_analysis);

  // Registers loop at position `loop_pos` of `op` as a new fusion class or
  // merges it in an existing fusion class.
  mlir::LogicalResult RegisterLoop(ComputeOp op, int loop_pos,
                                   const SequenceAnalysis &sequence_analysis);

  int next_loop_id_ = 0;
  mlir::MLIRContext *context_;
  llvm::DenseMap<mlir::Attribute, LoopFusionClass> fusion_classes_;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<MappingExpr, 4>>
      op_domain_mappings_;
};

// Verifies loop nest attributes of operations nested in the sair.program
// operation. Assumes that Sair operands are defined in the same program.
mlir::LogicalResult VerifyLoopNests(
    SairProgramOp program, const LoopFusionAnalysis &fusion_analysis,
    const IterationSpaceAnalysis &iteration_spaces,
    const SequenceAnalysis &sequence_analysis);

// Verifies that the loop_nest attribute is correct with regard to the shape of
// the operation it is attached to.
mlir::LogicalResult VerifyLoopNestWellFormed(
    mlir::Location loc, DomainShapeAttr shape,
    llvm::ArrayRef<mlir::Attribute> loop_nest);

}  // namespace sair

#endif  // SAIR_LOOP_NEST_H_
