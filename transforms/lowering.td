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

include "mlir/Pass/PassBase.td"

// List of dialects on which all lowering passes depend.
def Deps {
  list<string> dialects = ["::mlir::scf::SCFDialect",
                           "::mlir::StandardOpsDialect",
                           "::sair::SairDialect"];
}

def LowerMapReducePass : Pass<"sair-lower-map-reduce", "mlir::FuncOp"> {
  let summary = "Lowers map_reduce into map + fby operations";
  let constructor = [{ ::sair::CreateLowerMapReducePass(); }];
  let dependentDialects = Deps.dialects;
}

def LowerToMapPass : Pass<"sair-lower-to-map", "mlir::FuncOp"> {
  let summary = "Lowers sair operations into sair.map operations";
  let constructor = [{ ::sair::CreateLowerToMapPass(); }];
  let dependentDialects = !listconcat(
      Deps.dialects, ["::mlir::AffineDialect", "::mlir::memref::MemRefDialect"]);
}

def LowerToLLVM : Pass<"sair-lower-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lowers Sair and Standard dialect operations to LLVM dialect";
  let constructor = [{ ::sair::CreateLowerToLLVMPass(); }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def InlineTrivialSairOpsPass : Pass<"inline-trivial-sair-ops", "mlir::FuncOp"> {
  let summary = "Inlines Sair ops operating on and producing 0D values";
  let constructor = [{ ::sair::CreateInlineTrivialOpsPass(); }];
  let dependentDialects = Deps.dialects;
}

def MaterializeBuffersPass : Pass<"sair-materialize-buffers", "mlir::FuncOp"> {
  let summary = "Replace Sair values by buffers";
  let constructor = [{ ::sair::CreateMaterializeBuffersPass(); }];
  let dependentDialects = !listconcat(Deps.dialects, ["::mlir::AffineDialect"]);
}

def MaterializeInstancesPass : Pass<"sair-materialize-instances", "mlir::FuncOp"> {
  let summary = "Create ops for instances and copies defined in attributes";
  let constructor = [{ ::sair::CreateMaterializeInstancesPass(); }];
}

def IntroduceLoopsPass : Pass<"sair-introduce-loops", "mlir::FuncOp"> {
  let summary = "Replaces Sair iteration dimensions by loops";
  let constructor = [{ ::sair::CreateIntroduceLoopsPass(); }];
  let dependentDialects = Deps.dialects;
}

def NormalizeLoopsPass : Pass<"sair-normalize-loops", "mlir::FuncOp"> {
  let summary =
    "Rewrites operation domains so that each loop corresponds to a dimension";
  let constructor = [{ ::sair::CreateNormalizeLoopsPass(); }];
  let dependentDialects = !listconcat(Deps.dialects, ["::mlir::AffineDialect"]);
}

def LowerProjAnyPass : Pass<"sair-lower-proj-any", "mlir::FuncOp"> {
  let summary = "Eliminates or rewrite proj_any into proj_last operations";
  let constructor = [{ ::sair::CreateLowerProjAnyPass(); }];
  let dependentDialects = Deps.dialects;
}
