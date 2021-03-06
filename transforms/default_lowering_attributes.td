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

def DefaultLoopNestPass : FunctionPass<"sair-assign-default-loop-nest"> {
  let summary = "Assigns the default loop nest to Sair operations";
  let constructor = [{ ::sair::CreateDefaultLoopNestPass(); }];
}

def DefaultSequencePass : FunctionPass<"sair-assign-default-sequence"> {
  let summary = "Assigns the default sequence to Sair compute operations";
  let constructor = [{ ::sair::CreateDefaultSequencePass(); }];
}

def DefaultStoragePass : FunctionPass<"sair-assign-default-storage"> {
  let summary = "Assigns the default storage to Sair values.";

  let description = [{
    Assings 0D values to registers and other values to memory. Leaves existing
    storage attributes untouched. Operations must have a loop nest attribute.
  }];

  let constructor = [{ ::sair::CreateDefaultStoragePass(); }];
}

def DefaultExpansionPass : FunctionPass<"sair-assign-default-expansion"> {
  let summary = "Assigns default expansion patterns to Sair operations";

  let description = [{
    Assigns the default scalar expansion pattern to implement each Sair
    operation.
  }];

  let constructor = [{ ::sair::CreateDefaultExpansionPass(); }];
}
