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
#include "llvm/Support/Debug.h"
#include "sair_ops.h"

#define DEBUG_TYPE "sair-sequence"
#define DBGS(X) llvm::dbgs() << "[" DEBUG_TYPE "]"

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
using SairOpGraph = ConcreteOpGraph<SairOp>;

// A pseudo-container class implementing a DFS postorder iterator of a graph of
// compute ops. Provides traversal iterators through the customary begin/end.
template <typename OpTy>
class DFSPostorderTraversal {
 private:
  // DFS traversal state. Maintains an explicit stack to avoid recursive
  // functions on a potentially large number of IR elements.
  struct DFSState {
    llvm::SmallPtrSet<Operation *, 8> visited;
    ConcreteOpSet<OpTy> stack;
  };

 public:
  // Constructs the traversal container for the given graph.
  explicit DFSPostorderTraversal(const ConcreteOpGraph<OpTy> &graph)
      : graph_(graph) {}

  // Postorder DFS iterator over the operation graph.
  class iterator {
    friend class DFSPostorderTraversal;

   public:
    using iterator_category = std::input_iterator_tag;
    using value_type = OpTy;
    using pointer = OpTy;
    using reference = OpTy;
    using difference_type = ptrdiff_t;

    // Constructs a null (end) iterator.
    iterator() { SetEmpty(); }

    // Dereferences the iterator.
    OpTy operator*() { return current_; }

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
      for (OpTy op : container_->graph_.keys()) {
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

    // Returns the range of ops that form a cycle in the graph. Only callable
    // if the iterator has hit the cycle (dereferences to nullptr).
    decltype(std::declval<ConcreteOpSet<OpTy>>().Ops()) FindCycleOnStack()
        const {
      assert(current_ == nullptr && "the iterator hasn't hit the cycle (yet)");
      OpTy pre_cycle_op = state_.stack.back();
      ConcreteOpSet<OpTy> children = container_->graph_.lookup(pre_cycle_op);
      auto stack_range = state_.stack.Ops();
      for (OpTy child : children.Ops()) {
        auto first_occurrence = llvm::find(stack_range, child);
        if (first_occurrence != stack_range.end()) {
          return llvm::make_range(first_occurrence, stack_range.end());
        }
      }
      llvm_unreachable("no cycle found");
    }

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
    OpTy VisitDFSPostorder(OpTy root, const ConcreteOpGraph<OpTy> &graph,
                           DFSState &state) {
      state.stack.insert(root);
      do {
        // Fetch the op to visit from the top of the stack. If it has unvisited
        // children, put it back on stack, followed by the first child to visit.
        // Next iterations will visit the child and get back to this op.
        OpTy current = state.stack.pop_back_val();
        bool all_children_visited = true;
        ConcreteOpSet<OpTy> predecessors = graph.lookup(current);
        for (OpTy child : predecessors.Ops()) {
          if (state.visited.contains(child)) continue;
          state.stack.insert(current);
          // If `child` is already on the stack, we've hit a cycle.
          if (!state.stack.insert(child)) return nullptr;
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
    OpTy current_;
    OpTy root_;
  };

  iterator begin() const { return iterator(this); }

  iterator end() const { return iterator(); }

 private:
  const ConcreteOpGraph<OpTy> &graph_;
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

ComputeOpSet ComputeOpFrontier(SairOp op, ArrayRef<SairFbyOp> ignore = {}) {
  // The frontier is computed recursively as we don't expect long chains of
  // non-compute operations between compute operations, particularly in
  // canonicalized form that would have folded projection operations.
  ComputeOpSet frontier;
  auto add_and_recurse = [&](mlir::Value value) {
    auto defining_op = value.getDefiningOp<SairOp>();
    if (auto defining_compute_op =
            dyn_cast<ComputeOp>(defining_op.getOperation())) {
      frontier.insert(defining_compute_op);
      return;
    }
    frontier.merge(ComputeOpFrontier(defining_op, ignore));
  };
  if (llvm::find(ignore, op) != ignore.end()) {
    add_and_recurse(cast<SairFbyOp>(op.getOperation()).init());
  } else {
    for (ValueOperand value_operand : op.ValueOperands()) {
      add_and_recurse(value_operand.value());
    }
  }
  for (mlir::Value operand : op.domain()) {
    add_and_recurse(operand);
  }
  return frontier;
}

}  // end namespace

SequenceAnalysis::SequenceAnalysis(SairProgramOp program_op) {
  AssertSuccess(Init(program_op, /*report_errors=*/false));
}

std::optional<SequenceAnalysis> SequenceAnalysis::Create(
    SairProgramOp program_op, bool report_errors) {
  SequenceAnalysis analysis;
  if (mlir::failed(analysis.Init(program_op, report_errors))) {
    return std::nullopt;
  }
  return std::move(analysis);
}

mlir::LogicalResult SequenceAnalysis::Init(SairProgramOp program_op,
                                           bool report_errors) {
  program_op->walk([&](ComputeOp op) {
    if (llvm::Optional<int64_t> sequence_number = op.Sequence()) {
      sequenced_ops_.emplace(*sequence_number, op);
    }
  });
  return ComputeDefaultSequence(program_op, report_errors);
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

// Detects use-def cycles in the program and if they can be cut by removing the
// use-def edge of the "value" operand of a "fby" op, adds such ops into
// `fby_ops_to_cut`. Otherwise, returns failure.
static mlir::LogicalResult FindEdgesToCut(
    SairProgramOp program, llvm::SmallVectorImpl<SairFbyOp> &fby_ops_to_cut) {
  // Create a graph with of Sair predecessor ops.
  SairOpGraph predecessors;
  program.walk([&](SairOp op) {
    predecessors.insert(op);
    for (mlir::Value operand : op.domain()) {
      if (auto defining_op = operand.getDefiningOp<SairOp>()) {
        predecessors[op].insert(defining_op);
      }
    }
    for (ValueOperand operand : op.ValueOperands()) {
      predecessors[op].insert(operand.value().getDefiningOp<SairOp>());
    }
  });

  // Find cycles in the predecessor graph. Cycles that don't have an edge ending
  // as "value" of an "fby" operation are not allowed, report failure if found.
  // For cycles that do have such an edge, remove it and start over. Iterate
  // until all cycles are resolved.
  bool found = false;
  do {
    DFSPostorderTraversal<SairOp> traversal(predecessors);
    found = false;

    for (auto it = traversal.begin(), eit = traversal.end(); it != eit; ++it) {
      if (*it != nullptr) continue;

      auto cycle = llvm::to_vector<4>(it.FindCycleOnStack());
      cycle.push_back(cycle.front());
      for (int i = 0, e = cycle.size() - 1; i < e; ++i) {
        if (auto fby_op = dyn_cast<SairFbyOp>(cycle[i].getOperation())) {
          // If the cycle is not caused by the "then" edge of an "fby", we are
          // not interested in cutting it here.
          if (fby_op.Value().value().getDefiningOp() != cycle[i + 1]) continue;
          assert(predecessors[fby_op].contains(cycle[i + 1]));
          predecessors[fby_op].erase(cycle[i + 1]);
          found = true;

          // This is the edge to cut and not consider when computing the
          // backward slice.
          fby_ops_to_cut.push_back(fby_op);
          break;
        }
      }
      // If there is no edge in the cycle that connects an operation to "fby"
      // through its "then" operand, the cycle is invalid in the program.
      if (!found) return mlir::failure();
      break;
    }
  } while (found);
  return mlir::success();
}

bool FindImplicitlySequencedUseDefChain(ComputeOp from, ComputeOp to,
                                        llvm::SmallVectorImpl<SairOp> &stack) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  stack.push_back(cast<SairOp>(from.getOperation()));
  do {
    SairOp current = stack.back();
    bool all_users_visited = true;
    for (mlir::Operation *user : current->getUsers()) {
      if (!visited.insert(user).second) continue;
      auto sair_user = dyn_cast<SairOp>(user);
      if (!sair_user) continue;
      if (user == to) return true;
      if (isa<ComputeOp>(user)) continue;

      stack.push_back(sair_user);
      all_users_visited = false;
      break;
    }
    if (all_users_visited) stack.pop_back();
  } while (!stack.empty());
  return false;
}

mlir::LogicalResult SequenceAnalysis::ComputeDefaultSequence(
    SairProgramOp program, bool report_errors) {
  // This shouldn't fail as long as we control use-def chain order in the input
  // IR. When we don't, this could fail on unexpected use-def cycles, i.e.
  // cycles that are not caused by "fby", and should be reported back to the
  // caller.
  AssertSuccess(FindEdgesToCut(program, fby_ops_to_cut_));

  ComputeOpGraph predecessors;
  program.walk([&](ComputeOp op) {
    // Put the op in the adjacency list even if it has no predecessors.
    predecessors.insert(op);

    // Add all predecessor compute ops due to use-def chains. Note that we add
    // only the frontier since we will traverse the entire graph in DFS manner,
    // so there's no need to compute the entire slice here.
    predecessors[op] =
        ComputeOpFrontier(cast<SairOp>(op.getOperation()), fby_ops_to_cut_);

    // Add all ops with smaller sequence numbers as known predecessors.
    auto range = llvm::make_second_range(OpsBefore(op));
    predecessors[op].insert(range.begin(), range.end());
  });

  RemoveSelfDependencies(predecessors);

  // Walk the predecessor graph in DFS post-order, meaning that we will visit a
  // compute op after visiting all of its predecessors, and assign new sequence
  // numbers.
  sequenced_ops_.clear();
  DFSPostorderTraversal<ComputeOp> traversal(predecessors);
  int64_t counter = 0;
  for (auto it = traversal.begin(), eit = traversal.end(); it != eit; ++it) {
    if (*it != nullptr) {
      sequenced_ops_.emplace(counter++, *it);
      continue;
    }

    // If the traversal hits a cycle (indicated by the iterator pointing to
    // nullptr), this means order of operations implied by use-def chains
    // contradicts that implied by sequence attributes. That is, a use of a
    // value is sequenced before the value is defined. This situation is
    // slightly different from the pure use-def cycle detected above.
    auto cycle = llvm::to_vector<4>(it.FindCycleOnStack());
    LLVM_DEBUG({
      DBGS() << "unexpected cycle detected\n";
      for (ComputeOp cycle_op : cycle) DBGS() << cycle_op << "\n";
    });

    if (!report_errors) return mlir::failure();
    cycle.push_back(cycle.front());
    mlir::InFlightDiagnostic diag =
        cycle.back().emitError()
        << "operation sequencing contradicts use-def chains";
    for (int i = cycle.size() - 2; i >= 0; --i) {
      llvm::SmallVector<SairOp> stack;
      if (FindImplicitlySequencedUseDefChain(cycle[i + 1], cycle[i], stack)) {
        for (SairOp stack_op : llvm::drop_begin(stack)) {
          diag.attachNote(stack_op->getLoc())
              << "implicitly sequenced operation";
        }
        diag.attachNote(cycle[i]->getLoc())
            << "sequenceable operation sequenced by use-def";
      } else {
        diag.attachNote(cycle[i]->getLoc()) << "sequenceable operation";
      }
    }

    return diag;
  }
  return mlir::success();
}

}  // namespace sair
