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

def DefaultMemorySpacePass : FunctionPass<"sair-assign-default-memory-space"> {
  let summary = "Assigns the default memory space to Sair values";
  let constructor = [{ ::sair::CreateDefaultMemorySpacePass(); }];
}

def DefaultLoopNestPass : FunctionPass<"sair-assign-default-loop-nest"> {
  let summary = "Assigns the default loop nest to Sair operations";
  let constructor = [{ ::sair::CreateDefaultLoopNestPass(); }];
}
