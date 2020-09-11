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

#ifndef SAIR_SAIR_REGISTRATION_H_
#define SAIR_SAIR_REGISTRATION_H_

namespace mlir {
class DialectRegistry;
}  // namespace mlir

namespace sair {
// Registers Sair dialect with MLIR.
void RegisterSairDialect(mlir::DialectRegistry &registry);

// Registers all Sair passes with MLIR.
void RegisterSairPasses();
}  // namespace sair

#endif  // SAIR_SAIR_REGISTRATION_H_
