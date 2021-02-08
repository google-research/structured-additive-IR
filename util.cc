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
      current_loop_nest.take_front(num_loops), start.getContext());
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
    ComputeOp op, int dimension, llvm::StringRef origin,
    MappingAttr target_deps_to_op, MappingExpr &constraint,
    llvm::SmallVectorImpl<ValueAccess> &target_domain) {
  mlir::MLIRContext *context = op.getContext();
  SairOp sair_op = cast<SairOp>(op.getOperation());
  int domain_size = sair_op.domain().size();

  // Get a mapping from target dependencies to `dimension` domain.
  MappingAttr old_dependency_mapping =
      sair_op.shape().Dimension(dimension).dependency_mapping().ResizeUseDomain(
          domain_size);
  MappingAttr new_dependency_mapping =
      target_deps_to_op.Compose(old_dependency_mapping).Canonicalize();

  if (!new_dependency_mapping.IsFullySpecified()) {
    return op->emitError() << "dimension d" << dimension << " in " << origin
                           << " is used before its dependencies";
  }

  ValueAccess dimension_access = {sair_op.domain()[dimension],
                                  new_dependency_mapping};

  if (constraint.isa<MappingNoneExpr>()) {
    constraint = MappingDimExpr::get(target_domain.size(), context);
    target_domain.push_back(dimension_access);
  } else if (auto dim_expr = constraint.dyn_cast<MappingDimExpr>()) {
    if (dimension_access != target_domain[dim_expr.dimension()]) {
      return op.emitError() << "use of dimension d" << dimension << " in "
                            << origin << " does not match previous occurrences";
    }
  } else {
    return op.emitError() << "cannot unify " << origin
                          << " with previous occurences";
  }

  return mlir::success();
}

}  // namespace sair
