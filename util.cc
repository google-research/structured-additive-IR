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

InsertionPoint FindInsertionPoint(
    SairOp start, llvm::ArrayRef<mlir::Attribute> current_loop_nest,
    int num_loops, Direction direction) {
  mlir::Operation *current_op = start.getOperation();
  auto target_loop_nest = mlir::ArrayAttr::get(
      start.getContext(), current_loop_nest.take_front(num_loops));
  mlir::Operation *point = current_op;

  // Look for a point where only the first `num_loops` of the current loop nest
  // are open.
  while (current_loop_nest.size() > num_loops) {
    // Look for the next compute op.
    current_op = direction == Direction::kAfter ? current_op->getNextNode()
                                                : current_op->getPrevNode();
    if (current_op == nullptr) break;
    ComputeOp compute_op = dyn_cast<ComputeOp>(current_op);
    if (compute_op == nullptr) continue;

    // Trim current_loop_nest of dimensions that are not opened in current_op.
    if (!compute_op.loop_nest().hasValue()) break;
    llvm::ArrayRef<mlir::Attribute> new_loop_nest = compute_op.LoopNestLoops();
    int size = std::min(current_loop_nest.size(), new_loop_nest.size());
    for (; size > num_loops; --size) {
      if (current_loop_nest[size - 1].cast<LoopAttr>().name() ==
          new_loop_nest[size - 1].cast<LoopAttr>().name()) {
        break;
      }
    }
    current_loop_nest = current_loop_nest.take_front(std::max(size, num_loops));
    if (size > num_loops) {
      point = current_op;
    }
  }

  return {point, direction, target_loop_nest};
}

void ForwardAttributes(mlir::Operation *old_op, mlir::Operation *new_op,
                       llvm::ArrayRef<llvm::StringRef> ignore) {
  for (auto [name, attr] : old_op->getAttrs()) {
    if (new_op->hasAttr(name) || llvm::find(ignore, name) != ignore.end())
      continue;
    new_op->setAttr(name, attr);
  }
}

mlir::LogicalResult ResolveUnificationConstraint(
    mlir::Location loc, llvm::StringRef origin, const ValueAccess &dimension,
    MappingExpr &constraint,
    llvm::SmallVectorImpl<ValueAccess> &target_domain) {
  mlir::MLIRContext *context = constraint.getContext();

  // Ignore placeholders.
  mlir::Operation *defining_op = dimension.value.getDefiningOp();
  if (isa<SairPlaceholderOp>(defining_op)) return mlir::success();

  if (constraint.isa<MappingNoneExpr, MappingUnknownExpr>()) {
    constraint = MappingDimExpr::get(target_domain.size(), context);
    target_domain.push_back(dimension);
  } else if (auto dim_expr = constraint.dyn_cast<MappingDimExpr>()) {
    if (dimension != target_domain[dim_expr.dimension()]) {
      return mlir::emitError(loc) << "use of dimension in " << origin
                                  << " does not match previous occurrences";
    }
  } else {
    return mlir::emitError(loc)
           << "cannot unify " << origin << " with previous occurences";
  }

  return mlir::success();
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

llvm::SmallVector<mlir::Value> CreatePlaceholderDomain(
    mlir::Location loc, DomainShapeAttr shape, mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value> domain;
  domain.reserve(shape.NumDimensions());
  for (const DomainShapeDim &shape_dim : shape.Dimensions()) {
    llvm::SmallVector<mlir::Value> range_domain =
        CreatePlaceholderDomain(loc, shape_dim.type().Shape(), builder);
    domain.push_back(
        builder.create<SairPlaceholderOp>(loc, shape_dim.type(), range_domain));
  }
  return domain;
}

}  // namespace sair
