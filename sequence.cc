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

#include <algorithm>
#include <limits>
#include <map>

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "mlir/Support/LLVM.h"
#include "loop_nest.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "util.h"

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
    if (adjacency_.count(key)) return false;
    keys_.push_back(key);
    adjacency_.try_emplace(key);
    return true;
  }

  // Returns a mutable reference to the adjacency list of the given node.
  llvm::SetVector<OpTy> &operator[](OpTy key) {
    (void)insert(key);
    return adjacency_[key];
  }

  // Returns the adjacency list of the given node.
  llvm::SetVector<OpTy> lookup(OpTy key) const {
    return adjacency_.lookup(key);
  }

  // Returns `true` if the graph has no nodes.
  bool empty() const { return keys_.empty(); }

  // Returns a list of nodes in the graph.
  llvm::ArrayRef<OpTy> keys() const { return llvm::ArrayRef(keys_); }

 private:
  llvm::SmallVector<OpTy> keys_;
  llvm::SmallDenseMap<OpTy, llvm::SetVector<OpTy>> adjacency_;
};

using ComputeOpGraph = ConcreteOpGraph<ComputeOpInstance>;
using OpGraph = ConcreteOpGraph<OpInstance>;

// A pseudo-container class implementing a DFS postorder iterator of a graph of
// compute ops. Provides traversal iterators through the customary begin/end.
template <typename OpTy>
class DFSPostorderTraversal {
 private:
  // DFS traversal state. Maintains an explicit stack to avoid recursive
  // functions on a potentially large number of IR elements.
  struct DFSState {
    llvm::DenseSet<OpTy> visited;
    llvm::SetVector<OpTy> stack;
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
    using pointer = const OpTy *;
    using reference = const OpTy &;
    using difference_type = ptrdiff_t;

    // Constructs a null (end) iterator.
    iterator() { SetEmpty(); }

    // Dereferences the iterator.
    OpTy operator*() { return current_; }

    // Increments the iterator to point to the next DFS postorder element.
    iterator &operator++() {
      // Null iterator does not need to be incremented.
      if (!container_ || current_ == nullptr) return *this;

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
    const llvm::iterator_range<typename llvm::SetVector<OpTy>::iterator>
    FindCycleOnStack() const {
      assert(current_ == nullptr && "the iterator hasn't hit the cycle (yet)");
      OpTy pre_cycle_op = state_.stack.back();
      llvm::SetVector<OpTy> children = container_->graph_.lookup(pre_cycle_op);
      for (OpTy child : children) {
        auto first_occurrence = llvm::find(state_.stack, child);
        if (first_occurrence != state_.stack.end()) {
          return llvm::make_range(first_occurrence, state_.stack.end());
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
        llvm::SetVector<OpTy> predecessors = graph.lookup(current);
        for (OpTy child : predecessors) {
          if (state.visited.contains(child)) continue;
          state.stack.insert(current);
          // If `child` is already on the stack, we've hit a cycle.
          if (!state.stack.insert(child)) return OpTy();
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
      current_ = OpTy();
      container_ = nullptr;
    }

    // Checks if the iterator is in the empty (end) state.
    bool IsEmpty() const { return current_ == OpTy() && container_ == nullptr; }

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
  llvm::ArrayRef<ComputeOpInstance> all_ops = graph.keys();

  // Since we are working with predecessor graphs, drop self-links because the
  // op is not expected to be its own predecessor.
  for (const ComputeOpInstance &op : all_ops) {
    graph[op].remove(op);
  }
}

llvm::SetVector<ComputeOpInstance> ComputeOpFrontier(
    const OpInstance &op, ArrayRef<OpInstance> ignore = {}) {
  // The frontier is computed recursively as we don't expect long chains of
  // non-compute operations between compute operations, particularly in
  // canonicalized form that would have folded projection operations.
  llvm::SetVector<ComputeOpInstance> frontier;
  auto add_and_recurse = [&](std::optional<ResultInstance> value) {
    if (!value.has_value()) return;
    OpInstance defining_op = value->defining_op();
    if (auto defining_compute_op = defining_op.dyn_cast<ComputeOpInstance>()) {
      frontier.insert(defining_compute_op);
      return;
    }
    llvm::SetVector<ComputeOpInstance> def_frontier =
        ComputeOpFrontier(defining_op, ignore);
    frontier.insert(def_frontier.begin(), def_frontier.end());
  };
  if (llvm::find(ignore, op) != ignore.end()) {
    auto fby_op = cast<SairFbyOp>(op.GetDuplicatedOp());
    add_and_recurse(OperandInstance(fby_op.Init(), op).GetValue());
  } else {
    for (OperandInstance operand : op.Operands()) {
      add_and_recurse(operand.GetValue());
    }
  }
  for (ResultInstance operand : op.getDomain()) {
    add_and_recurse(operand);
  }
  return frontier;
}

}  // end namespace

ProgramPoint::ProgramPoint(ComputeOpInstance op, Direction direction,
                           llvm::ArrayRef<mlir::StringAttr> loop_nest)
    : program_(op.program()),
      op_(op),
      direction_(direction),
      loop_nest_(loop_nest) {}

int ProgramPoint::NumCommonLoops(const ProgramPoint &other) const {
  assert(program_ == other.program_);
  auto it = std::mismatch(loop_nest_.begin(), loop_nest_.end(),
                          other.loop_nest_.begin(), other.loop_nest_.end());
  return std::distance(loop_nest_.begin(), it.first);
}

void ProgramPoint::TrimLoopNest(int num_loops) {
  loop_nest_ = loop_nest_.take_front(num_loops);
}

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
  return ComputeDefaultSequence(program_op, report_errors);
}

SequenceAnalysis::RangeType SequenceAnalysis::Ops() const {
  return RangeType(compute_ops_);
}

void SequenceAnalysis::AssignInferred() const {
  int64_t number = 0;
  for (ComputeOpInstance op : Ops()) {
    op.SetDecisions(UpdateSequence(op.GetDecisions(), number++));
  }
}

bool SequenceAnalysis::IsBefore(const ComputeOpInstance &first,
                                const OpInstance &second) const {
  if (first == second) return false;
  int64_t first_number = ExplicitSequenceNumber(first);

  // If both ops are ComputeOps, just check the sequence numbers.
  if (auto second_as_compute = second.dyn_cast<ComputeOpInstance>()) {
    return first_number < ExplicitSequenceNumber(second_as_compute);
  }
  // If the second op is a non-compute, its implicit sequence number is equal to
  // the largest explicit sequence number of its operands; so equal numbers mean
  // the compute op is sequenced before the non-compute op due to a use-def
  // chain between them.
  // NOTE: extending this function to query the order between two non-compute
  // ops will require looking for a potential use-def chain between them.
  return first_number <= ImplicitSequenceNumber(second);
}

bool SequenceAnalysis::IsBefore(ProgramPoint point,
                                const ComputeOpInstance &op) const {
  if (point.operation() == nullptr || point.operation() == op) {
    return point.direction() == Direction::kBefore;
  }
  return IsBefore(point.operation(), op);
}

bool SequenceAnalysis::IsAfter(ProgramPoint point,
                               const ComputeOpInstance &op) const {
  if (point.operation() == nullptr || point.operation() == op) {
    return point.direction() == Direction::kAfter;
  }
  return IsBefore(op, point.operation());
}

void SequenceAnalysis::Insert(const ComputeOpInstance &op, ProgramPoint point) {
  Insert(op, point.operation(), point.direction());
}

void SequenceAnalysis::Insert(const ComputeOpInstance &op,
                              const OpInstance &reference,
                              Direction direction) {
  int64_t sequence_number = -1;
  if (reference != nullptr) {
    if (auto compute_op = reference.dyn_cast<ComputeOpInstance>()) {
      sequence_number = ExplicitSequenceNumber(compute_op);
    } else {
      sequence_number = ImplicitSequenceNumber(reference);
    }
  }

  // Implicit sequence number can be -1 if the reference operation doesn't
  // depend on any explicitly sequenced operation. In this case, insert the
  // operation at the beginning of the program for the "before" direction and at
  // the end for the "after" direction.
  if (sequence_number == -1) {
    sequence_number = direction == Direction::kBefore ? 0 : compute_ops_.size();
  } else if (direction == Direction::kAfter) {
    ++sequence_number;
  }

  for (int64_t number = sequence_number, e = compute_ops_.size(); number < e;
       ++number) {
    op_to_sequence_number_[compute_ops_[number]] = number + 1;
  }
  op_to_sequence_number_.try_emplace(op, sequence_number);
  compute_ops_.insert(compute_ops_.begin() + sequence_number, op);
}

void SequenceAnalysis::Erase(const ComputeOpInstance &op) {
  int64_t sequence_number = ExplicitSequenceNumber(op);
  for (int64_t number = sequence_number + 1, e = compute_ops_.size();
       number < e; ++number) {
    op_to_sequence_number_[compute_ops_[number]] = number - 1;
  }
  op_to_sequence_number_.erase(op);
  compute_ops_.erase(compute_ops_.begin() + sequence_number);
}

int64_t SequenceAnalysis::ImplicitSequenceNumber(const OpInstance &op) const {
  assert(!op.isa<ComputeOpInstance>() &&
         "only non-compute ops have implicit sequence numbers");
  llvm::SetVector<ComputeOpInstance> frontier =
      ComputeOpFrontier(op, fby_ops_to_cut_);
  auto get_explicit_number = [this](ComputeOpInstance compute_op) {
    return ExplicitSequenceNumber(compute_op);
  };
  auto range = llvm::map_range(frontier, get_explicit_number);
  int64_t number = -1;
  for (int64_t sequence : range) number = std::max(number, sequence);
  return number;
}

std::pair<ComputeOpInstance, ComputeOpInstance> SequenceAnalysis::GetSpan(
    llvm::ArrayRef<ComputeOpInstance> ops) const {
  assert(!ops.empty());
  int64_t min = std::numeric_limits<int64_t>::max();
  int64_t max = std::numeric_limits<int64_t>::min();
  auto get_sequence_number = [&](ComputeOpInstance op) {
    return ExplicitSequenceNumber(op);
  };
  for (int64_t sequence_number : llvm::map_range(ops, get_sequence_number)) {
    min = std::min(sequence_number, min);
    max = std::max(sequence_number, max);
  }
  return std::make_pair(compute_ops_[min], compute_ops_[max]);
}

ProgramPoint SequenceAnalysis::FindInsertionPoint(
    const IterationSpaceAnalysis &iter_spaces, const OpInstance &start,
    int num_loops, Direction direction) const {
  // Compute initial sequence number.
  int sequence_number;
  if (auto compute_op = start.dyn_cast<ComputeOpInstance>()) {
    sequence_number = ExplicitSequenceNumber(compute_op);
  } else {
    sequence_number = ImplicitSequenceNumber(start);
    // If the operation is not a ComputeOp and we want to schedule before the
    // operation, then any point that is before the next ComputeOp is fine as
    // the current operation is implicitly scheduled.
    if (sequence_number >= 0 && direction == Direction::kBefore) {
      ++sequence_number;
    }
  }

  llvm::ArrayRef<mlir::StringAttr> start_loop_nest =
      iter_spaces.Get(start).loop_names();
  int num_common_loops = start_loop_nest.size();
  int delta = direction == Direction::kBefore ? -1 : 1;

  sequence_number += delta;
  while (sequence_number >= 0 && sequence_number < compute_ops_.size()) {
    ComputeOpInstance new_op = compute_ops_[sequence_number];
    llvm::ArrayRef<mlir::Attribute> new_loops = new_op.Loops();
    num_common_loops = std::min<int>(new_loops.size(), num_common_loops);
    for (; num_common_loops > 0; --num_common_loops) {
      auto loop = mlir::cast<LoopAttr>(new_loops[num_common_loops - 1]);
      if (loop.name() == start_loop_nest[num_common_loops - 1]) break;
    }
    if (num_common_loops <= num_loops) break;
    sequence_number += delta;
  }

  sequence_number -= delta;

  auto target_loop_nest = start_loop_nest.take_front(num_loops);
  if (sequence_number < 0) {
    return ProgramPoint(start.program(), Direction::kBefore, target_loop_nest);
  } else if (sequence_number >= compute_ops_.size()) {
    return ProgramPoint(start.program(), Direction::kAfter, target_loop_nest);
  } else {
    return ProgramPoint(compute_ops_[sequence_number], direction,
                        target_loop_nest);
  }
}

// Detects use-def cycles in the program and if they can be cut by removing the
// use-def edge of the "value" operand of a "fby" op, adds such ops into
// `fby_ops_to_cut`. Otherwise, returns failure.
static mlir::LogicalResult FindEdgesToCut(
    SairProgramOp program, llvm::SmallVectorImpl<OpInstance> &fby_ops_to_cut,
    bool report_errors) {
  // Create a graph of Sair predecessor ops.
  OpGraph predecessors;
  program.WalkOpInstances([&](const OpInstance &op) {
    predecessors.insert(op);
    for (ResultInstance operand : op.getDomain()) {
      predecessors[op].insert(operand.defining_op());
    }
    for (OperandInstance operand : op.Operands()) {
      auto value = operand.GetValue();
      if (!value.has_value()) continue;
      predecessors[op].insert(value->defining_op());
    }
  });

  // Find cycles in the predecessor graph. Cycles that don't have an edge ending
  // as "value" of an "fby" operation are not allowed, report failure if found.
  // For cycles that do have such an edge, remove it and start over. Iterate
  // until all cycles are resolved.
  bool found = false;
  do {
    DFSPostorderTraversal<OpInstance> traversal(predecessors);
    found = false;

    for (auto it = traversal.begin(), eit = traversal.end(); it != eit; ++it) {
      if (*it != nullptr) continue;

      auto cycle = llvm::to_vector<4>(it.FindCycleOnStack());
      cycle.push_back(cycle.front());
      for (int i = 0, e = cycle.size() - 1; i < e; ++i) {
        // Filter out operations that are not fby operations.
        if (cycle[i].is_copy()) continue;
        mlir::Operation *concrete_op = cycle[i].GetDuplicatedOp();
        auto fby_op = dyn_cast<SairFbyOp>(concrete_op);
        if (fby_op == nullptr) continue;

        OperandInstance operand(fby_op.Value(), cycle[i]);
        auto value = operand.GetValue();
        if (!value.has_value()) continue;

        // If the cycle is not caused by the "then" edge of an "fby", we are
        // not interested in cutting it here.
        if (value->defining_op() != cycle[i + 1]) continue;
        assert(predecessors[cycle[i]].contains(cycle[i + 1]));
        predecessors[cycle[i]].remove(cycle[i + 1]);
        found = true;

        // This is the edge to cut and not consider when computing the
        // backward slice.
        fby_ops_to_cut.push_back(cycle[i]);
        break;
      }
      // If there is no edge in the cycle that connects an operation to "fby"
      // through its "then" operand, the cycle is invalid in the program.
      if (found) break;
      if (!report_errors) return mlir::failure();
      mlir::InFlightDiagnostic diag = cycle.front().EmitError()
                                      << "unexpected use-def cycle";
      for (OpInstance cycle_op : llvm::drop_begin(cycle)) {
        cycle_op.AttachNote(diag) << "operation in the cycle";
      }
      return diag;
    }
  } while (found);
  return mlir::success();
}

bool FindImplicitlySequencedUseDefChain(
    const ComputeOpInstance &from, const ComputeOpInstance &to,
    llvm::SmallVectorImpl<OpInstance> &stack) {
  llvm::DenseSet<OpInstance> visited;
  stack.push_back(from);
  do {
    OpInstance current = stack.back();
    bool all_users_visited = true;
    for (ResultInstance result : current.Results()) {
      for (auto &[user, pos] : result.GetUses()) {
        (void)pos;
        if (!visited.insert(user).second) continue;
        if (user == to) return true;
        if (user.isa<ComputeOpInstance>()) continue;
        stack.push_back(user);
        all_users_visited = false;
        break;
      }
      if (!all_users_visited) break;
    }
    if (all_users_visited) stack.pop_back();
  } while (!stack.empty());
  return false;
}

// Returns an iterator range pointing to `sequenced_ops` of all operations
// sequenced before the given one, in their relative order. All operations are
// given a relative order even if they don't have a sequence attribute attached.
// The sequence number returned in this iteration may differ from that of the
// sequence attribute if the Sair program hasn't been canonicalized.
static llvm::iterator_range<
    std::multimap<int64_t, ComputeOpInstance>::const_iterator>
OpsBefore(const std::multimap<int64_t, ComputeOpInstance> &sequenced_ops,
          ComputeOpInstance op) {
  DecisionsAttr decisions = op.GetDecisions();
  if (decisions.sequence() == nullptr) {
    return llvm::make_range(sequenced_ops.end(), sequenced_ops.end());
  }

  int64_t sequence_number = decisions.sequence().getInt();
  return llvm::make_range(sequenced_ops.begin(),
                          sequenced_ops.lower_bound(sequence_number));
}

mlir::LogicalResult SequenceAnalysis::ComputeDefaultSequence(
    SairProgramOp program, bool report_errors) {
  // We use a standard multimap because (a) the sequence numbers can be shared
  // and (b) we need a deterministic increasing order that is provided by this
  // map and not provided by hash table-based maps.
  std::multimap<int64_t, ComputeOpInstance> initial_sequence;
  program.WalkComputeOpInstances([&](const ComputeOpInstance &op) {
    DecisionsAttr decisions = op.GetDecisions();
    if (decisions.sequence() == nullptr) return;
    initial_sequence.emplace(decisions.sequence().getInt(), op);
  });

  // This shouldn't fail as long as we control use-def chain order in the input
  // IR. When we don't, this could fail on unexpected use-def cycles, i.e.
  // cycles that are not caused by "fby", and should be reported back to the
  // caller.
  if (mlir::failed(FindEdgesToCut(program, fby_ops_to_cut_, report_errors))) {
    return mlir::failure();
  }

  ComputeOpGraph predecessors;
  program.WalkComputeOpInstances([&](const ComputeOpInstance &op) {
    // Put the op in the adjacency list even if it has no predecessors.
    predecessors.insert(op);

    // Add all predecessor compute ops due to use-def chains. Note that we add
    // only the frontier since we will traverse the entire graph in DFS manner,
    // so there's no need to compute the entire slice here.
    predecessors[op] = ComputeOpFrontier(op, fby_ops_to_cut_);

    // Add all ops with smaller sequence numbers as known predecessors.
    auto range = llvm::make_second_range(OpsBefore(initial_sequence, op));
    predecessors[op].insert(range.begin(), range.end());
  });

  RemoveSelfDependencies(predecessors);

  // Walk the predecessor graph in DFS post-order, meaning that we will visit a
  // compute op after visiting all of its predecessors, and assign new sequence
  // numbers.
  DFSPostorderTraversal<ComputeOpInstance> traversal(predecessors);
  compute_ops_.reserve(predecessors.keys().size());
  for (auto it = traversal.begin(), eit = traversal.end(); it != eit; ++it) {
    if (*it != nullptr) {
      op_to_sequence_number_.try_emplace(*it, compute_ops_.size());
      compute_ops_.push_back(*it);
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
      for (const ComputeOpInstance &cycle_op : cycle)
        DBGS() << cycle_op.getOperation() << "\n";
    });

    if (!report_errors) return mlir::failure();
    cycle.push_back(cycle.front());
    mlir::InFlightDiagnostic diag =
        cycle.back().EmitError()
        << "operation sequencing contradicts use-def chains";
    for (int i = cycle.size() - 2; i >= 0; --i) {
      llvm::SmallVector<OpInstance> stack;
      if (FindImplicitlySequencedUseDefChain(cycle[i + 1], cycle[i], stack)) {
        for (OpInstance stack_op : llvm::drop_begin(stack)) {
          stack_op.AttachNote(diag) << "implicitly sequenced operation";
        }
        cycle[i].AttachNote(diag)
            << "sequenceable operation sequenced by use-def";
      } else {
        cycle[i].AttachNote(diag) << "sequenceable operation";
      }
    }

    return diag;
  }
  return mlir::success();
}

}  // namespace sair
