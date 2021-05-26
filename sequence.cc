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

#include "sequence.h"

#include "llvm/ADT/iterator_range.h"
#include "sair_ops.h"

namespace sair {

ComputeOpBackwardSliceAnalysis::ComputeOpBackwardSliceAnalysis(
    SairProgramOp program_op) {
  program_op.walk([this](ComputeOp op) {
    if (frontiers_.count(op.getOperation())) return;
    ComputeFrontier(cast<SairOp>(op.getOperation()));
  });
}

void ComputeOpBackwardSliceAnalysis::ComputeFrontier(SairOp op) {
  // The frontier is computed recursively as we don't expect long chains of
  // non-compute operations between compute operations, particularly in
  // canonicalized form that would have folded projection operations.
  frontiers_.try_emplace(op.getOperation());
  auto add_and_recurse = [&](mlir::Value value) {
    auto defining_op = value.getDefiningOp<SairOp>();
    assert(defining_op);
    if (auto defining_compute_op =
            dyn_cast<ComputeOp>(defining_op.getOperation())) {
      frontiers_[op.getOperation()].insert(defining_compute_op);
      return;
    }
    if (frontiers_.count(defining_op) == 0) ComputeFrontier(defining_op);
    frontiers_[op.getOperation()].merge(frontiers_[defining_op.getOperation()]);
  };

  for (ValueOperand value_operand : op.ValueOperands()) {
    add_and_recurse(value_operand.value());
  }
  for (mlir::Value operand : op.domain()) {
    add_and_recurse(operand);
  }
}

const ComputeOpSet &ComputeOpBackwardSliceAnalysis::BackwardSlice(
    ComputeOp op) const {
  auto it = slice_cache_.find(op.getOperation());
  if (it != slice_cache_.end()) return it->getSecond();

  // Iteratively compute the slice. If the slice of an operand-defining
  // operation is known, merge it into the current slice. Otherwise, merge in
  // the frontier of that operation. Repeat until no new operations are added to
  // the slice, which is a fixed point.
  ComputeOpSet &closure = slice_cache_[op.getOperation()];
  closure.merge(frontiers_.find(op.getOperation())->getSecond());
  size_t orig_size = 0;
  do {
    auto range = closure.Ops();
    auto sub_range =
        llvm::make_range(std::next(range.begin(), orig_size), range.end());
    orig_size = closure.size();
    for (ComputeOp sub_op : sub_range) {
      if (MergeSliceIfAvailable(sub_op, closure)) continue;
      closure.merge(BackwardFrontier(sub_op));
    }
  } while (closure.size() != orig_size);
  return closure;
}

bool ComputeOpBackwardSliceAnalysis::MergeSliceIfAvailable(
    ComputeOp op, ComputeOpSet &slice) const {
  auto it = slice_cache_.find(op.getOperation());
  if (it == slice_cache_.end()) return false;

  slice.merge(it->getSecond());
  return true;
}

SequenceAnalysis::SequenceAnalysis(SairProgramOp program_op) {
  program_op->walk([&](ComputeOp op) {
    if (llvm::Optional<int64_t> sequence_number = op.Sequence()) {
      sequenced_ops_.emplace(*sequence_number, op);
    }
  });
}

SequenceAnalysis::ConstRangeType SequenceAnalysis::Ops() const {
  return ConstRangeType(sequenced_ops_);
}

SequenceAnalysis::ConstRangeType SequenceAnalysis::OpsBefore(
    ComputeOp op) const {
  llvm::Optional<int64_t> sequence_number = op.Sequence();
  if (!sequence_number) {
    return ConstRangeType(sequenced_ops_.end(), sequenced_ops_.end());
  }

  return ConstRangeType(sequenced_ops_.begin(),
                        sequenced_ops_.lower_bound(*sequence_number));
}

}  // namespace sair
