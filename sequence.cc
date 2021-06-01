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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
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

ProgramPoint::ProgramPoint(ComputeOp op, Direction direction,
                           llvm::ArrayRef<mlir::StringAttr> loop_nest)
    : program_(op->getParentOp()),
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
  for (ComputeOp op : Ops()) {
    op.SetSequence(number++);
  }
}

bool SequenceAnalysis::IsBefore(ComputeOp first, SairOp second) const {
  if (first.getOperation() == second.getOperation()) return false;
  int64_t first_number = ExplicitSequenceNumber(first);
  // If both ops are ComputeOps, just check the sequence numbers.
  if (auto second_compute = dyn_cast<ComputeOp>(second.getOperation())) {
    return first_number < ExplicitSequenceNumber(second_compute);
  }
  // If the second op is a non-compute, its implicit sequence number is equal to
  // the largest explicit sequence number of its operands; so equal numbers mean
  // the compute op is sequenced before the non-compute op due to a use-def
  // chain between them.
  // NOTE: extending this function to query the order between two non-compute
  // ops will require looking for a potential use-def chain between them.
  return first_number <= ImplicitSequenceNumber(second);
}

bool SequenceAnalysis::IsBefore(ProgramPoint point, ComputeOp op) const {
  if (point.operation() == nullptr || point.operation() == op) {
    return point.direction() == Direction::kBefore;
  }
  return IsBefore(point.operation(), op);
}

bool SequenceAnalysis::IsAfter(ProgramPoint point, ComputeOp op) const {
  if (point.operation() == nullptr || point.operation() == op) {
    return point.direction() == Direction::kAfter;
  }
  return IsBefore(op, point.operation());
}

void SequenceAnalysis::Insert(ComputeOp op, ComputeOp reference,
                              Direction direction) {
  Insert(op, cast<SairOp>(reference.getOperation()), direction);
}

void SequenceAnalysis::Insert(ComputeOp op, SairOp reference,
                              Direction direction) {
  auto reference_compute_op = dyn_cast<ComputeOp>(reference.getOperation());
  int64_t sequence_number = reference_compute_op
                                ? ExplicitSequenceNumber(reference_compute_op)
                                : ImplicitSequenceNumber(reference);
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
  op_to_sequence_number_.try_emplace(op.getOperation(), sequence_number);
  compute_ops_.insert(compute_ops_.begin() + sequence_number, op);
}

void SequenceAnalysis::Erase(ComputeOp op) {
  int64_t sequence_number = ExplicitSequenceNumber(op);
  for (int64_t number = sequence_number + 1, e = compute_ops_.size();
       number < e; ++number) {
    op_to_sequence_number_[compute_ops_[number]] = number - 1;
  }
  op_to_sequence_number_.erase(op.getOperation());
  compute_ops_.erase(compute_ops_.begin() + sequence_number);
}

void SequenceAnalysis::ImplicitlySequencedOps(
    int64_t sequence_number, llvm::SmallVectorImpl<SairOp> &ops) const {
  assert(sequence_number >= 0 && sequence_number < compute_ops_.size() &&
         "sequence number not in analysis");
  ComputeOp compute_op = compute_ops_[sequence_number];

  // Iterative post-order DFS starting from the compute op.
  llvm::SetVector<Operation *> visited;
  llvm::SetVector<Operation *> stack;
  stack.insert(compute_op);
  do {
    SairOp current = cast<SairOp>(stack.back());
    bool all_visited = true;
    // Schedule users of the current operation for visitation before visiting
    // the current operation. Avoid already visited operations and stack cycles.
    // Stop at an explicitly sequenced (compute) operation.
    for (mlir::Operation *user : current->getUsers()) {
      if (visited.count(user) != 0 || stack.count(user) != 0) continue;
      auto sair_op = dyn_cast<SairOp>(user);
      if (!sair_op || isa<ComputeOp>(user)) continue;
      // The user may depend on another compute operation sequenced after
      // `compute_op` and therefore not be implicitly sequenced after
      // `compute_op`. Don't visit it in this case.
      if (ImplicitSequenceNumber(sair_op) > sequence_number) continue;
      stack.insert(user);
      all_visited = false;
      break;
    }
    if (all_visited) {
      visited.insert(stack.pop_back_val());
    }
  } while (!stack.empty());

  // `visited` contains the operations in DFS postorder of their use-def
  // dependencies, with `current_op` being the last one. Drop it and reverse the
  // order to create a topologically sorted list of operations according to
  // their use-def dependencies.
  visited.pop_back();
  ops.clear();
  llvm::append_range(
      ops, llvm::map_range(llvm::reverse(visited), [](mlir::Operation *op) {
        return cast<SairOp>(op);
      }));
}

int64_t SequenceAnalysis::ImplicitSequenceNumber(SairOp op) const {
  assert(!isa<ComputeOp>(op.getOperation()) &&
         "only non-compute ops have implicit sequence numbers");
  ComputeOpSet frontier = ComputeOpFrontier(op, fby_ops_to_cut_);
  auto range = llvm::map_range(frontier.Ops(), [this](ComputeOp compute_op) {
    return ExplicitSequenceNumber(compute_op);
  });
  int64_t number = -1;
  for (int64_t sequence : range) number = std::max(number, sequence);
  return number;
}

std::pair<ComputeOp, ComputeOp> SequenceAnalysis::GetSpan(
    llvm::ArrayRef<ComputeOp> ops) const {
  assert(!ops.empty());
  int64_t min = std::numeric_limits<int64_t>::max();
  int64_t max = std::numeric_limits<int64_t>::min();
  for (int64_t sequence_number : llvm::map_range(
           ops, [&](ComputeOp op) { return ExplicitSequenceNumber(op); })) {
    min = std::min(sequence_number, min);
    max = std::max(sequence_number, max);
  }
  return std::make_pair(compute_ops_[min], compute_ops_[max]);
}

InsertionPoint SequenceAnalysis::FindInsertionPoint(
    SairOp start, llvm::ArrayRef<mlir::Attribute> current_loop_nest,
    int num_loops, Direction direction) const {
  mlir::Operation *point = start;
  ComputeOp current_op = [&]() {
    if (auto compute_op = dyn_cast<ComputeOp>(start.getOperation())) {
      return compute_op;
    }
    int64_t sequence_number = ImplicitSequenceNumber(start);
    if (sequence_number == -1) return ComputeOp();
    return compute_ops_[sequence_number];
  }();
  if (current_op != nullptr) point = current_op;
  auto target_loop_nest = mlir::ArrayAttr::get(
      start.getContext(), current_loop_nest.take_front(num_loops));

  // Look for a point where only the first `num_loops` of the current loop nest
  // are open.
  while (current_loop_nest.size() > num_loops) {
    // Look for the adjacent in sequence compute op.
    current_op = direction == Direction::kAfter ? NextOp(current_op)
                                                : PrevOp(current_op);
    if (current_op == nullptr) break;

    assert(current_op.loop_nest().hasValue() &&
           "expected loop nests to have been set");

    // Trim current_loop_nest of dimensions that are not open in current_op.
    llvm::ArrayRef<mlir::Attribute> new_loop_nest = current_op.LoopNestLoops();
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

// Detects use-def cycles in the program and if they can be cut by removing the
// use-def edge of the "value" operand of a "fby" op, adds such ops into
// `fby_ops_to_cut`. Otherwise, returns failure.
static mlir::LogicalResult FindEdgesToCut(
    SairProgramOp program, llvm::SmallVectorImpl<SairFbyOp> &fby_ops_to_cut,
    bool report_errors) {
  // Create a graph of Sair predecessor ops.
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
      if (found) break;
      if (!report_errors) return mlir::failure();
      mlir::InFlightDiagnostic diag = cycle.front().emitError()
                                      << "unexpected use-def cycle";
      for (SairOp cycle_op : llvm::drop_begin(cycle)) {
        diag.attachNote(cycle_op->getLoc()) << "operation in the cycle";
      }
      return diag;
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

// Returns an iterator range pointing to `sequenced_ops` of all operations
// sequenced before the given one, in their relative order. All operations are
// given a relative order even if they don't have a sequence attribute attached.
// The sequence number returned in this iteration may differ from that of the
// sequence attribute if the Sair program hasn't been canonicalized.
static llvm::iterator_range<std::multimap<int64_t, ComputeOp>::const_iterator>
OpsBefore(const std::multimap<int64_t, ComputeOp> &sequenced_ops,
          ComputeOp op) {
  llvm::Optional<int64_t> sequence_number = op.Sequence();
  if (!sequence_number) {
    return llvm::make_range(sequenced_ops.end(), sequenced_ops.end());
  }

  return llvm::make_range(sequenced_ops.begin(),
                          sequenced_ops.lower_bound(*sequence_number));
}

mlir::LogicalResult SequenceAnalysis::ComputeDefaultSequence(
    SairProgramOp program, bool report_errors) {
  // We use a standard multimap because (a) the sequence numbers can be shared
  // and (b) we need a deterministic increasing order that is provided by this
  // map and not provided by hash table-based maps.
  std::multimap<int64_t, ComputeOp> initial_sequence;
  program->walk([&](ComputeOp op) {
    if (llvm::Optional<int64_t> sequence_number = op.Sequence()) {
      initial_sequence.emplace(*sequence_number, op);
    }
  });

  // This shouldn't fail as long as we control use-def chain order in the input
  // IR. When we don't, this could fail on unexpected use-def cycles, i.e.
  // cycles that are not caused by "fby", and should be reported back to the
  // caller.
  if (mlir::failed(FindEdgesToCut(program, fby_ops_to_cut_, report_errors))) {
    return mlir::failure();
  }

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
    auto range = llvm::make_second_range(OpsBefore(initial_sequence, op));
    predecessors[op].insert(range.begin(), range.end());
  });

  RemoveSelfDependencies(predecessors);

  // Walk the predecessor graph in DFS post-order, meaning that we will visit a
  // compute op after visiting all of its predecessors, and assign new sequence
  // numbers.
  DFSPostorderTraversal<ComputeOp> traversal(predecessors);
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
