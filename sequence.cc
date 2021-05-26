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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/iterator_range.h"
#include "sair_ops.h"

namespace sair {

namespace {
// A graph with as nodes operations of OpTy. Maintains the order in which the
// nodes were added to enable deterministic traversal order.
template <typename OpTy>
class ConcreteOpGraph {
 public:
  ConcreteOpGraph() {}

  // Insert a new node into the graph.
  bool insert(OpTy key) {
    if (adjacency_.count(key.getOperation())) return false;
    keys_.push_back(key);
    adjacency_.try_emplace(key.getOperation());
    return true;
  }

  // Returns a mutable reference to the adjacency list of the given node.
  ConcreteOpSet<OpTy> &operator[](OpTy key) {
    (void)insert(key);
    return adjacency_[key.getOperation()];
  }

  // Returns the adjacency list of the given node.
  ConcreteOpSet<OpTy> lookup(OpTy key) const {
    return adjacency_.lookup(key.getOperation());
  }

  // Returns `true` if the graph has no nodes.
  bool empty() const { return keys_.empty(); }

  // Returns a list of nodes in the graph.
  llvm::ArrayRef<OpTy> keys() const { return llvm::makeArrayRef(keys_); }

 private:
  llvm::SmallVector<OpTy> keys_;
  llvm::SmallDenseMap<Operation *, ConcreteOpSet<OpTy>> adjacency_;
};

using ComputeOpGraph = ConcreteOpGraph<ComputeOp>;

// A pseudo-container class implementing a DFS postorder iterator of a graph of
// compute ops. Provides traversal iterators through the customary begin/end.
class DFSPostorderTraversal {
 private:
  // DFS traversal state. Maintains an explicit stack to avoid recursive
  // functions on a potentially large number of IR elements.
  struct DFSState {
    llvm::SmallPtrSet<Operation *, 8> visited;
    llvm::SmallVector<ComputeOp> stack;
  };

 public:
  // Constructs the traversal container for the given graph.
  explicit DFSPostorderTraversal(const ComputeOpGraph &graph) : graph_(graph) {}

  // Postorder DFS iterator over the operation graph.
  class iterator {
    friend class DFSPostorderTraversal;

   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = ComputeOp;
    using pointer = ComputeOp;
    using reference = ComputeOp;
    using difference_type = ptrdiff_t;

    // Constructs a null (end) iterator.
    iterator() { SetEmpty(); }

    // Dereferences the iterator.
    ComputeOp operator*() { return current_; }

    // Increments the iterator to point to the next DFS postorder element.
    iterator &operator++() {
      // Null iterator does not need to be incremented.
      if (!container_ || !current_) return *this;

      // If we haven't circled back to the root operation, continue the DFS.
      if (current_ != root_) {
        current_ = VisitDFSPostorder(root_, container_->graph_, state_);
        return *this;
      }

      // Otherwise, go through graph nodes to check for another traversal root,
      // i.e. the op that hasn't been visited yet.
      for (ComputeOp op : container_->graph_.keys()) {
        if (!state_.visited.contains(op)) {
          root_ = op;
          current_ = VisitDFSPostorder(root_, container_->graph_, state_);
          return *this;
        }
      }

      // If there is no unvisited op in the graph, the traversal is complete.
      SetEmpty();
      return *this;
    }

    // Returns `true` if `other` points to the same element of the same
    // container, or if both `this` and `other` are null iterators.
    bool operator==(const iterator &other) const {
      return (current_ == other.current_ && container_ == other.container_) ||
             (IsEmpty() && other.IsEmpty());
    }

    // Returns `false` if `other` points to the same element of the same
    // container, or if both `this` and `other` are null iterators.
    bool operator!=(const iterator &other) const { return !(*this == other); }

   private:
    // Constructs the iterator pointing to the first element of the DFS
    // postorder traversal in the given graph.
    explicit iterator(const DFSPostorderTraversal *container)
        : container_(container) {
      if (!container || container_->graph_.empty()) {
        SetEmpty();
        return;
      }
      root_ = container_->graph_.keys().front();
      current_ = VisitDFSPostorder(root_, container_->graph_, state_);
    }

    // Continues the DFS postorder visitation of `graph` started at `root` with
    // the given `state`. The latter contains the virtual "call" stack of DFS,
    // more specifically the list of operations the visitation of which was
    // postponed until their children are visited. Returns the next operation in
    // DFS postorder and updates `state` for subsequent calls. If `root` is
    // returned, no further ops can be visited.
    ComputeOp VisitDFSPostorder(ComputeOp root, const ComputeOpGraph &graph,
                                DFSState &state) {
      state.stack.push_back(root);
      do {
        // Fetch the op to visit from the top of the stack. If it has unvisited
        // children, put it back on stack, followed by the first child to visit.
        // Next iterations will visit the child and get back to this op.
        ComputeOp current = state.stack.pop_back_val();
        bool all_children_visited = true;
        ComputeOpSet predecessors = graph.lookup(current);
        for (ComputeOp child : predecessors.Ops()) {
          if (state.visited.contains(child)) continue;
          state.stack.push_back(current);
          state.stack.push_back(child);
          all_children_visited = false;
          break;
        }
        // If all children are visited, it's time for the current op to be
        // visited.
        if (all_children_visited) {
          state.visited.insert(current);
          return current;
        }
      } while (!state.stack.empty());
      llvm_unreachable("must have returned root instead");
    }

    // Sets the iterator to empty (end) state.
    void SetEmpty() {
      current_ = nullptr;
      container_ = nullptr;
    }

    // Checks if the iterator is in the empty (end) state.
    bool IsEmpty() const {
      return current_ == nullptr && container_ == nullptr;
    }

    const DFSPostorderTraversal *container_;
    DFSState state_;
    ComputeOp current_;
    ComputeOp root_;
  };

  iterator begin() const { return iterator(this); }

  iterator end() const { return iterator(); }

 private:
  const ComputeOpGraph &graph_;
};

// Updates `graph` to remove self-dependencies.
void RemoveSelfDependencies(ComputeOpGraph &graph) {
  llvm::ArrayRef<ComputeOp> all_ops = graph.keys();

  // Since we are working with predecessor graphs, drop self-links because the
  // op is not expected to be its own predecessor.
  for (ComputeOp op : all_ops) {
    graph[op].erase(op);
  }
}
}  // end namespace

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
  ComputeDefaultSequence(program_op);
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

void SequenceAnalysis::ComputeDefaultSequence(SairProgramOp program) {
  ComputeOpGraph predecessors;
  ComputeOpBackwardSliceAnalysis slice_analysis(program);

  program.walk([&](ComputeOp op) {
    // Put the op in the adjacency list even if it has no predecessors.
    predecessors.insert(op);

    // Add all predecessor compute ops due to use-def chains. Note that we add
    // only the frontier since we will traverse the entire graph in DFS manner,
    // so there's no need to compute the entire slice here.
    predecessors[op] = slice_analysis.BackwardFrontier(op);

    // Add all ops with smaller sequence numbers as known predecessors.
    auto range = llvm::make_second_range(OpsBefore(op));
    predecessors[op].insert(range.begin(), range.end());
  });

  RemoveSelfDependencies(predecessors);

  // Walk the predecessor graph in DFS post-order, meaning that we will visit a
  // compute op after visiting all of its predecessors, and assign new sequence
  // numbers.
  sequenced_ops_.clear();
  for (auto en : llvm::enumerate(DFSPostorderTraversal(predecessors))) {
    sequenced_ops_.emplace(static_cast<int64_t>(en.index()), en.value());
  }
}

}  // namespace sair
