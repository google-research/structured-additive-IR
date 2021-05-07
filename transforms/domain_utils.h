// Copyright 2021 Google LLC
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

#ifndef THIRD_PARTY_SAIR_TRANSFORMS_DOMAIN_UTILS_H_
#define THIRD_PARTY_SAIR_TRANSFORMS_DOMAIN_UTILS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "sair_op_interfaces.h"

namespace sair {

// Creates a domain with the given shape using placeholder dimensions.
llvm::SmallVector<mlir::Value> CreatePlaceholderDomain(
    mlir::Location loc, DomainShapeAttr shape, mlir::OpBuilder &builder);

// Parameters of a Sair range. Begin and end of the range can be either values
// or constants.
struct RangeParameters {
  // First index of the range.
  mlir::OpFoldResult begin;
  // End of the range.
  mlir::OpFoldResult end;
  // Step of the range.
  int step;
};

// Returns the parameters (start index, end index and step) of the ranges
// obtained by applying mapping to `source_domain`. Populates `current_body`
// with operations to compute parameters and `current_op_arguments` with
// Sair value argument to pass to the body. Assumes that `current_body` is the
// body of a sair.map operation and `current_to_source` a mapping from the
// domain of the map operation to `source_domain`.
llvm::SmallVector<RangeParameters> GetRangeParameters(
    mlir::Location loc, MappingAttr mapping,
    llvm::ArrayRef<ValueAccess> source_domain, MappingAttr current_to_source,
    mlir::SmallVector<ValueAccess> &current_op_arguments,
    mlir::Block &current_body, mlir::OpBuilder &builder);

}  // namespace sair

#endif  // THIRD_PARTY_SAIR_TRANSFORMS_DOMAIN_UTILS_H_
