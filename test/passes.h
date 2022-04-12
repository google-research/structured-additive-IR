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

#ifndef SAIR_TEST_PASSES_H_
#define SAIR_TEST_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace sair {

// Returns a pass that tests mapping expressions methods.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTestMappingExprsPass();

// Returns a pass that tests DomainShapeAttr methods.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTestDomainShapePass();

}  // namespace sair

#endif  // SAIR_TEST_PASSES_H_
