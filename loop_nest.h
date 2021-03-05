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

#ifndef THIRD_PARTY_SAIR_LOOP_NEST_H_
#define THIRD_PARTY_SAIR_LOOP_NEST_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"

namespace sair {

// Verifies loop nest attributes of operations nested in the sair.program
// operation. Assumes that Sair operands are defined in the same program.
mlir::LogicalResult VerifyLoopNests(SairProgramOp program);

// Analysis of how data is distributed on loop nests iterations. It indicates,
// for each value and operation, the loops they are nested in and the slice of
// the domain each loop iteration owns.
//
// The analysis returns the loops each operation or value is nested in, and how
// the iterating along the loop iterates on the domain. A single loop iteration
// may own more that a single point of the domain. For example, consider the
// following Sair code.
// ```
// %4 = sair.copy[d0:%0, d1:%1, d2: %2] %3 {
//   loop_nest = [
//     {name = "A", iter = #sair.iter<d0>},
//     {name = "B", iter = #sair.iter<d2>},
//     {name = "C", iter = #sair.iter<d2>},
// } : !sair.value<d0:range x d1:range x d2:range>
// %5 = sair.proj_last[d0:%0, d1:%2] of[d2:%1] %4(d0, d2, d1)
//   : #sair.shape<d0:range, d1:range>, f32
// ```
// Here,%5 is nested in loop "A", but not in loops "B" and "C", as it is
// projected along loop "B" and "C" is nested in "B". Each point of the loop
// nest ["A", "B", "C"] will own a point of %4, and each point of "A" will own a
// slice of %5.
//
class IterationSpaceAnalysis {
 public:
  explicit IterationSpaceAnalysis(SairProgramOp program_op);
  explicit IterationSpaceAnalysis(mlir::Operation *operation)
      : IterationSpaceAnalysis(dyn_cast<SairProgramOp>(operation)) {}

  // Computes or retrieves the loops `op` is nested in. Returns the empty
  // iteration space if the loop nest is left unspecified.
  llvm::ArrayRef<mlir::Attribute> IterationSpace(SairOp op) const;

  // Computes or retrieves the loops the value is nested in. The value must be
  // defined by a sair operation. Returns the empty iteration space if the loop
  // nest is left unspecified.
  llvm::ArrayRef<mlir::Attribute> IterationSpace(mlir::Value value) const;

 private:
  // Computes the iteration space for the given operation.
  mlir::ArrayAttr ComputeIterationSpace(mlir::Operation *operation);

  llvm::DenseMap<mlir::Operation *, mlir::ArrayAttr> iteration_space_;
};

// A class of fused loops.
struct LoopFusionClass {
  // Loops this class depends on.
  llvm::SmallVector<mlir::StringAttr> dependencies;

  // Domain in which the loop size is defined. This is a list of dimensions,
  // with an access pattern from dependencies indicies to the domain of each
  // dimension.
  //
  // Domains of outer fusion classes must be a prefix of this one.
  llvm::SmallVector<ValueAccess> domain;

  // Mapping from domain indices to the loop indices.
  MappingExpr iter_expr;

  // An occurence on the fusion class, for error reporting purposes.
  ComputeOp occurence;
};

// A loop nest of fused loops.
// TODO: use in normalize_loops.cc
struct LoopNest {
  // Domain used to define loop ranges.
  llvm::ArrayRef<ValueAccess> domain;
  // Mapping from `domain` to loops.
  MappingAttr domain_to_loops;
  // Shape of the resulting loop nest.
  DomainShapeAttr loops_shape;
};

// Computes loop fusion classes in a sair program.
class LoopFusionAnalysis {
 public:
  // Builds an analysis populated with all loops appearing in `program_op`.
  explicit LoopFusionAnalysis(mlir::Operation *operation);

  // Creates a LoopFusionAnalysis populated with the loops appearing in
  // `program_op`. Returns `nullopt` if the analysis fails.
  static std::optional<LoopFusionAnalysis> Create(SairProgramOp program_op);

  // Retrieves the fusion class with the given name.
  const LoopFusionClass &GetClass(mlir::StringAttr name) const {
    return fusion_classes_.find(name)->second;
  }

  // Retrives the unified loop nest corresponding to loops.
  LoopNest GetLoopNest(llvm::ArrayRef<mlir::Attribute> loops) const;
  LoopNest GetLoopNest(llvm::ArrayRef<mlir::StringAttr> loop_names) const;

 private:
  LoopFusionAnalysis(mlir::MLIRContext *context) : context_(context) {}

  // Populates the analysis with the operations appearing in `program_op`.
  mlir::LogicalResult Init(SairProgramOp program_op);

  // Registers a loop of an operation. All occurences of outer loops must be
  // registered first.
  mlir::LogicalResult RegisterLoop(ComputeOp op, LoopAttr loop,
                                   llvm::ArrayRef<mlir::Attribute> outer_loops);

  mlir::MLIRContext *context_;
  llvm::DenseMap<mlir::Attribute, LoopFusionClass> fusion_classes_;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<MappingExpr, 4>>
      op_domain_mappings_;
};

}  // namespace sair

#endif  // THIRD_PARTY_SAIR_LOOP_NEST_H_
