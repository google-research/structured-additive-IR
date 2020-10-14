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

#ifndef SAIR_SAIR_TRAITS_H_
#define SAIR_SAIR_TRAITS_H_

#include <algorithm>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_types.h"

// MLIR requires trait to be defined in the mlir::OpTrait namespace.
namespace mlir {
namespace OpTrait {
namespace sair {

// Trait for Sair operations that return a single Sair value or iteration
// dimension.
template <typename ConcreteType>
class OneResult : public OpTrait::TraitBase<ConcreteType, OneResult> {
 public:
  // Verifies that the operation has a single result, of type !sair.value. This
  // is a hook for the Mlir trait system.
  static mlir::LogicalResult verifyTrait(Operation *op) {
    if (op->getNumResults() != 1) {
      return op->emitError() << "requires one result";
    }
    if (!op->getResult(0).getType().isa<::sair::SairShapedType>()) {
      return op->emitError() << "requires a !sair.value return type";
    }
    return success();
  }

  // Returns the shape of the operation.
  ::sair::DomainShapeAttr shape() {
    return this->getOperation()
        ->getResult(0)
        .getType()
        .template cast<::sair::SairShapedType>()
        .Shape();
  }
};

}  // namespace sair
}  // namespace OpTrait
}  // namespace mlir

#endif  // SAIR_SAIR_TRAITS_H_
