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

#include "sair_traits.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"

namespace sair {

AccessPatternAttr ValueOperand::AccessPattern() {
  return cast<SairOp>(operand_->getOwner()).access_pattern_array()
      .getValue()[index_]
      .template cast<::sair::AccessPatternAttr>();
}

void ValueOperand::SetAccessPattern(AccessPatternAttr access_pattern) {
  SairOp op = cast<SairOp>(operand_->getOwner());
  op.SetAccessPattern(index_, access_pattern);
}

ValueOperandRange::ValueOperandRange()
    : RangeBaseT(std::make_pair(nullptr, 0), 0) {}

ValueOperandRange::ValueOperandRange(
    llvm::MutableArrayRef<mlir::OpOperand> operands)
    : RangeBaseT(std::make_pair(operands.data(), 0), operands.size()) {}

ValueOperandRange::PtrPair ValueOperandRange::offset_base(PtrPair base_ptr,
                                                          ptrdiff_t offset) {
  base_ptr.first += offset;
  base_ptr.second += offset;
  return base_ptr;
}

ValueOperand ValueOperandRange::dereference_iterator(PtrPair base_ptr,
                                                     ptrdiff_t offset) {
  return ValueOperand(base_ptr.first + offset, base_ptr.second + offset);
}

llvm::SmallBitVector ValueOperand::DimsDependingOnOperand() const {
  return cast<SairOp>(operand_->getOwner()).DimsDependingOnOperand(index_);
}

bool ValueOperand::AllowUseBeforeDef() const {
  return cast<SairOp>(operand_->getOwner()).AllowUseBeforeDef(index_);
}

// Sair operations are only allowed inside a SairProgramOp.
mlir::LogicalResult VerifySairOpParent(mlir::Operation *operation) {
  if (isa<SairProgramOp>(operation->getParentOp())) {
    return mlir::success();
  }

  return operation->emitOpError()
         << "expected to be immediately contained in a '"
         << SairProgramOp::getOperationName() << "'";
}

}  // namespace sair
