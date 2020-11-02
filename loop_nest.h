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

// Verifies loop nest attributes of operations nested in the
// sair.program operation.
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

  // Computes or retrieves the loops `op` is nested in.
  mlir::ArrayAttr IterationSpace(SairOp op) const;

  // Computes or retrieves the loops the value is nested in. The value must
  // be defined by a sair operation.
  mlir::ArrayAttr IterationSpace(mlir::Value value) const;

 private:
  // Computes the iteration space for the given operation.
  mlir::ArrayAttr ComputeIterationSpace(mlir::Operation *operation);

  llvm::DenseMap<mlir::Operation *, mlir::ArrayAttr> iteration_space_;
};

}  // namespace sair

#endif  // THIRD_PARTY_SAIR_LOOP_NEST_H_
