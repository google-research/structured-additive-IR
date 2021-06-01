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

#include "util.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"

namespace sair {

void InsertionPoint::Set(mlir::OpBuilder &builder) const {
  if (direction == Direction::kAfter) {
    builder.setInsertionPointAfter(operation);
  } else {
    builder.setInsertionPoint(operation);
  }
}

void ForwardAttributes(mlir::Operation *old_op, mlir::Operation *new_op,
                       llvm::ArrayRef<llvm::StringRef> ignore) {
  for (auto [name, attr] : old_op->getAttrs()) {
    if (new_op->hasAttr(name) || llvm::find(ignore, name) != ignore.end())
      continue;
    new_op->setAttr(name, attr);
  }
}

void SetInArrayAttr(mlir::Operation *operation, llvm::StringRef attr_name,
                    int array_size, int element, mlir::Attribute value) {
  mlir::MLIRContext *context = operation->getContext();
  llvm::SmallVector<mlir::Attribute, 4> values;

  auto old_attr = operation->getAttr(attr_name);
  if (old_attr == nullptr) {
    values.resize(array_size, mlir::UnitAttr::get(context));
  } else {
    auto old_array_attr = old_attr.cast<mlir::ArrayAttr>();
    assert(old_array_attr.size() == array_size);
    llvm::append_range(values, old_array_attr.getValue());
  }

  values[element] = value;
  operation->setAttr(attr_name, mlir::ArrayAttr::get(context, values));
}

mlir::Value Materialize(mlir::Location loc, mlir::OpFoldResult value,
                        mlir::OpBuilder &builder) {
  if (value.is<mlir::Value>()) return value.get<mlir::Value>();
  return builder.create<mlir::ConstantOp>(loc, value.get<mlir::Attribute>());
}

}  // namespace sair
