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

#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "transforms/default_lowering_attributes.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {

void CreateSairToLoopConversionPipeline(mlir::OpPassManager *pm) {
  pm->addPass(CreateInsertCopiesPass());
  pm->addPass(CreateLowerToMapPass());
  pm->addPass(CreateLowerToMemRefPass());
  pm->addPass(CreateMaterializeMemRefsPass());
  pm->addPass(CreateIntroduceLoopsPass());
  pm->addPass(CreateInlineTrivialOpsPass());
}

void CreateSairToLLVMConversionPipeline(mlir::OpPassManager *pm) {
  CreateSairToLoopConversionPipeline(pm);
  pm->addPass(mlir::createLowerAffinePass());
  pm->addPass(mlir::createLowerToCFGPass());
  pm->addPass(mlir::createLowerToLLVMPass());
}

}  // namespace sair
