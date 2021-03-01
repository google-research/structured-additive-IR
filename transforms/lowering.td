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

def LowerMapReducePass : FunctionPass<"sair-lower-map-reduce"> {
  let summary = "Lowers map_reduce into map + fby operations";
  let constructor = [{ ::sair::CreateLowerMapReducePass(); }];
  let dependentDialects = Deps.dialects;
}

def LowerToMapPass : FunctionPass<"sair-lower-to-map"> {
  let summary = "Lowers sair operations into sair.map operations";
  let constructor = [{ ::sair::CreateLowerToMapPass(); }];
  let dependentDialects = !listconcat(Deps.dialects, ["::mlir::AffineDialect"]);
}

def LowerToMemRefPass : FunctionPass<"sair-lower-tomemref"> {
  let summary = "Folds sair.to_memref operations in their predecessors";
  let constructor = [{ ::sair::CreateLowerToMemRefPass(); }];
  let dependentDialects = Deps.dialects;
}

def LowerToLLVM : Pass<"sair-lower-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Lowers Sair and Standard dialect operations to LLVM dialect";
  let constructor = [{ ::sair::CreateLowerToLLVMPass(); }];
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def InlineTrivialSairOpsPass : FunctionPass<"inline-trivial-sair-ops"> {
  let summary = "Inlines Sair ops operating on and producing 0D values";
  let constructor = [{ ::sair::CreateInlineTrivialOpsPass(); }];
  let dependentDialects = Deps.dialects;
}

def InsertCopiesPass : FunctionPass<"sair-insert-copies"> {
  let summary =
    "Insert copies before operations when needed to operate in place";
  let constructor = [{ ::sair::CreateInsertCopiesPass(); }];
  let dependentDialects = Deps.dialects;
}

def MaterializeMemRefsPass : FunctionPass<"sair-materialize-memrefs"> {
  let summary = "Materializes Sair values as memrefs";
  let constructor = [{ ::sair::CreateMaterializeMemRefsPass(); }];
  let dependentDialects = Deps.dialects;
}

def IntroduceLoopsPass : FunctionPass<"sair-introduce-loops"> {
  let summary = "Replaces Sair iteration dimensions by loops";
  let constructor = [{ ::sair::CreateIntroduceLoopsPass(); }];
  let dependentDialects = Deps.dialects;
}

def RematerializePass : FunctionPass<"sair-rematerialize"> {
  let summary = "Replaces rematerialization annotations with actual dimensions";
  let constructor = [{ ::sair::CreateRematerializePass(); }];
  // This does not create ops from dialects that are not present in the input.
}

def NormalizeLoopsPass : FunctionPass<"sair-normalize-loops"> {
  let summary =
    "Rewrites operation domains so that each loop corresponds to a dimension";
  let constructor = [{ ::sair::CreateNormalizeLoopsPass(); }];
  let dependentDialects = !listconcat(Deps.dialects, ["::mlir::AffineDialect"]);
}
