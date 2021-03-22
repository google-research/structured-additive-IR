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

#ifndef SAIR_SAIR_TRANSFORMS_H_
#define SAIR_SAIR_TRANSFORMS_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace sair {

// Returns a pass that lowers sair.map_reduce operations into sair.map,
// sair.proj_last and sair.fby operations.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerMapReducePass();

// Returns a pass that converts sair operations into sair.map operations.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerToMapPass();

// Returns a pass that lowers Sair and Standard dialect operations to the LLVM
// dialect.
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass();

// Returns a pass that replaces the first trivial Sair Op in the function with
// the contents of its body.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateInlineTrivialOpsPass();

// Replaces Sair values by buffers as specified by the `storage` attribute.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
CreateMaterializeBuffersPass();

// Replaces iteration dimensions by loops in sair.map and sair.map_reduce
// operations.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateIntroduceLoopsPass();

// Returns a pass that rewrites the domain of operations so that each loop
// corresponds to a dimension.
std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateNormalizeLoopsPass();

// Populates the pass manager to convert Sair operations to the Loops dialect.
void CreateSairToLoopConversionPipeline(mlir::OpPassManager *pm);

// Populates the pass manages to convert Sair operations to LLVM.
void CreateSairToLLVMConversionPipeline(mlir::OpPassManager *pm);

}  // namespace sair

#endif  // SAIR_SAIR_TRANSFORMS_H_
