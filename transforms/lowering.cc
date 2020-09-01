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

#include "transforms/lowering.h"

#include <algorithm>
#include <cstddef>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "sair_traits.h"
#include "sair_types.h"
#include "transforms/default_lowering_attributes.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {
namespace {

}  // namespace

void CreateSairToLoopConversionPipeline(mlir::OpPassManager *pm) {
  pm->addPass(CreateInsertCopiesPass());
  pm->addPass(CreateCopyToMapPass());
  pm->addPass(CreateLowerToMemRefPass());
  pm->addPass(CreateMaterializeMemRefsPass());
  pm->addPass(CreateIntroduceLoopsPass());
  pm->addPass(mlir::createCSEPass());  // Remove unused dimensions
  pm->addPass(CreateInlineTrivialOpsPass());
}

void CreateSairToLLVMConversionPipeline(mlir::OpPassManager *pm) {
  CreateSairToLoopConversionPipeline(pm);
  pm->addPass(mlir::createLowerAffinePass());
  pm->addPass(mlir::createLowerToCFGPass());
  pm->addPass(mlir::createLowerToLLVMPass());
}

}  // namespace sair
