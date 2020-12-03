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

#ifndef SAIR_SAIR_OPS_H_
#define SAIR_SAIR_OPS_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/UseDefLists.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"
#include "sair_traits.h"
#include "sair_types.h"

namespace sair {

// Returns the operand range covering two consecutive operand ranges `first` and
// `second`.
OperandRange ChainOperandRanges(OperandRange first, OperandRange second);

}  // namespace sair

#define GET_OP_CLASSES
#include "sair_ops.h.inc"

namespace sair {
// Parses the declaration of an iteration domain. Appends the dimensions that
// compose the domain to 'dimensions' and their names to 'dim_names' with the
// name that the domain binds to each dimension. The syntax for iteration
// domains is the following.
//
//   domain ::= ('[' (name ':' ssa-value),+ ']')?
//
// Note that the empty domain is represented by the empty string. This function
// assumes that the input represents the empty domain if it doesn't start with
// '['.
ParseResult ParseDomain(
    mlir::OpAsmParser &parser,
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> &dimensions);
// Resolves the operands that consitute the dimensions of an iteration domain
// and registers them in 'result'.
ParseResult ResolveDomain(mlir::OpAsmParser &parser,
                          DomainShapeAttr expected_shape,
                          llvm::ArrayRef<mlir::OpAsmParser::OperandType> domain,
                          mlir::OperationState &result);
// Parses an access to a Sair value, composed of an SSA-value and a mapping.
ParseResult ParseValueAccess(int num_dimensions, mlir::OpAsmParser &parser,
                             mlir::OpAsmParser::OperandType &value,
                             MappingAttr &mapping);

// Prints access to a Sair value, composed of the name of the SSA value and of
// the mapping along the dimensions of the operation. Dimensions are
// assigned names based on their position in the domain, following the mapping
// d<position>.
void PrintValueAccess(ValueOperand value, OpAsmPrinter &printer);
// Prints an iteration domain. A domain is a list of dimensions between square
// brackets, with a name assigned to each dimensions. Dimensions are named after
// their positions: the i-th position is named 'd(i+offset)'.
void PrintDomain(mlir::Operation::operand_range dimensions,
                 OpAsmPrinter &printer, int offset = 0);

// Predicates that tests if two values have the same element type. Values can be
// Sair values or mlir shaped values.
bool IsSameElementType(mlir::Value lhs, mlir::Value rhs);
// Tests if a Sair value and an MLIR shaped value have the same rank.
bool IsSameRank(mlir::Value sair_value, mlir::Value mlir_value);
// Verifies that the domain matches the domain shape in the operation.
mlir::LogicalResult VerifyDomain(SairOp op);
// Verifies that the mapping does not reference dimensions beyond
// "num_parallel_dimensions", which are interpreted as reduction dimensions.
mlir::LogicalResult VerifyReductionMapping(MappingAttr mapping,
                                           int num_parallel_dimensions);

}  // namespace sair

#endif  // SAIR_SAIR_OPS_H_
