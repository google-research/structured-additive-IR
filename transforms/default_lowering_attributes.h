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

#ifndef SAIR_DEFAULT_LOWERING_ATTRIBUTES_H_
#define SAIR_DEFAULT_LOWERING_ATTRIBUTES_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "sair_ops.h"

namespace sair {

// Returns a pass that sets the memory space of Sair values to its default
// value. Leaves existing memory space attributes intact.
std::unique_ptr<mlir::Pass> CreateDefaultMemorySpacePass();

// Returns a pass that sets the `loop_nest` attribute of Sair operations to its
// default value. Leaves the attribute untouched if already present.
std::unique_ptr<mlir::Pass> CreateDefaultLoopNestPass();

// Generates the default `loop_nest` attribute for an operation with the given
// number of dimensions. The loop nest will start with the given prefix.
mlir::ArrayAttr GetDefaultLoopNest(SairProgramOp program, int num_dimensions,
                                   llvm::ArrayRef<mlir::Attribute> prefix = {});

}  // namespace sair

#endif  // SAIR_DEFAULT_LOWERING_ATTRIBUTES_H_
