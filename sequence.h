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

#ifndef SAIR_SEQUENCE_H_
#define SAIR_SEQUENCE_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator_range.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"
#include "util.h"

namespace sair {

// Position of an operation relative to another.
enum class Direction { kBefore, kAfter };

// A point in the execution of the program. A point can be:
// - Immediately before or after a Sair operation.
// - Immediately before entering the Sair program.
// - Immediately after exiting the Sair program.
class ProgramPoint {
 public:
  // Constructs a program point that is before or after the whole program.
  ProgramPoint(SairProgramOp program, Direction direction,
               llvm::ArrayRef<mlir::StringAttr> loop_nest = {})
      : program_(program), direction_(direction), loop_nest_(loop_nest) {}

  // Constructs a program point that is before or after `op`. Saves a reference
  // to `loop_nest`.
  ProgramPoint(ComputeOpInstance op, Direction direction,
               llvm::ArrayRef<mlir::StringAttr> loop_nest = {});

  // If null, the point is outside of the sair program. If non-null the point is
  // immediately before or after this operation.
  ComputeOpInstance operation() const { return op_; }

  // Indicates if the point is before or after operation() or before or after
  // the Sair program.
  Direction direction() const { return direction_; }

  // Loop nest the point is nested in.
  llvm::ArrayRef<mlir::StringAttr> loop_nest() const { return loop_nest_; }

  // Reduces the number of loops in loop_nest().
  void TrimLoopNest(int num_loops);

  // Number of common loops between two program points.
  int NumCommonLoops(const ProgramPoint &other) const;

 private:
  SairProgramOp program_;
  ComputeOpInstance op_;
  Direction direction_;
  llvm::ArrayRef<mlir::StringAttr> loop_nest_;
};

class IterationSpaceAnalysis;

// An analysis of the relative positions of Sair operations indicated by their
// sequence attributes.
class SequenceAnalysis {
 public:
  using IterType = llvm::SmallVector<ComputeOpInstance>::const_iterator;
  using RangeType = llvm::iterator_range<IterType>;

  // Performs the analysis in the given Sair program.
  explicit SequenceAnalysis(SairProgramOp program_op);

  // Creates and returns the analysis for the given Sair program, or `nullopt`
  // if the analysis cannot be performed, e.g., if the program has use-def
  // cycles between compute ops.
  static std::optional<SequenceAnalysis> Create(SairProgramOp program_op,
                                                bool report_errors = false);

  // Returns an iterator range for traversing operations in their relative
  // order. All operations are given a relative order even if they don't have a
  // sequence attribute attached. The sequence number returned in this iteration
  // may differ from that of the sequence attribute if the Sair program hasn't
  // been canonicalized.
  RangeType Ops() const;

  // Assings inferred (contiguous) sequence numbers to operations by setting
  // their "sequence" attributes.
  void AssignInferred() const;

  // Returns true if `first` is known to be sequenced before `second`, false
  // otherwise. Note that this currently relies on the default implicit order of
  // sequenced ops so even the ops that do not need to be sequenced in the
  // relative order may be sequenced. This is likely to change in the future.
  bool IsBefore(const ComputeOpInstance &first, const OpInstance &second) const;

  // Returns true if the program point is sequenced before the given op.
  bool IsBefore(ProgramPoint point, const ComputeOpInstance &op) const;

  // Returns true if the program point is sequenced after the given op.
  bool IsAfter(ProgramPoint point, const ComputeOpInstance &op) const;

  // Inserts the given `op` into the analysis, sequencing before or after the
  // `reference` op, depending on `direction`.
  void Insert(const ComputeOpInstance &op, ProgramPoint point);
  void Insert(const ComputeOpInstance &op, const OpInstance &reference,
              Direction direction);

  // Erases the given `op` from the analysis.
  void Erase(const ComputeOpInstance &op);

  // Returns the Sair operation of the given kind preceding `op` if any; steps
  // over the operations of other kinds.
  ComputeOpInstance PrevOp(const ComputeOpInstance &op) const {
    if (op == nullptr) return ComputeOpInstance();
    auto iter = op_to_sequence_number_.find(op);
    assert(iter != op_to_sequence_number_.end() &&
           "op not in sequence analysis");
    if (iter->getSecond() == 0) return ComputeOpInstance();
    return compute_ops_[iter->getSecond() - 1];
  }

  // Returns the Sair operation of the given kind preceding `op` if any; steps
  // over the operations of other kinds.
  ComputeOpInstance NextOp(const ComputeOpInstance &op) const {
    if (op == nullptr) return ComputeOpInstance();
    auto iter = op_to_sequence_number_.find(op);
    assert(iter != op_to_sequence_number_.end());
    if (iter->getSecond() == compute_ops_.size() - 1)
      return ComputeOpInstance();
    return compute_ops_[iter->getSecond() + 1];
  }

  // Returns the pair (first, last) of the given ops according to their sequence
  // numbers.
  std::pair<ComputeOpInstance, ComputeOpInstance> GetSpan(
      llvm::ArrayRef<ComputeOpInstance> ops) const;

  // Finds the first point in the program where it is possible to insert an
  // operation nested in the first `num_loops` of `current_loop_nest`, when
  // starting from `start`.
  ProgramPoint FindInsertionPoint(
      const IterationSpaceAnalysis &iter_spaces, const OpInstance &start,
      int num_loops, Direction direction = Direction::kBefore) const;

 private:
  // Default noop constructor. Init must be called separately.
  SequenceAnalysis() = default;

  // Initializes the analysis for the given program op. This may fail if the
  // program contains use-def loops between compute operations (loops are
  // allowed only through the non-compute by operation).
  mlir::LogicalResult Init(SairProgramOp program_op, bool report_errors);

  // Updates the internal state to have sequence numbers for all compute
  // operations in the program, inferring their relative order from the
  // available sequence attribtues and use-def chains. The relative order is
  // preserved but not the absolute sequence numbers. The traversal order is
  // deterministic but otherwise unspecified for operations that do not have
  // "sequence" attribute and belong to different connected components of the
  // use-def dependency graph.
  mlir::LogicalResult ComputeDefaultSequence(SairProgramOp program,
                                             bool report_errors);

  // Returns the sequence number of the given op.
  int64_t ExplicitSequenceNumber(const ComputeOpInstance &op) const {
    auto it = op_to_sequence_number_.find(op);
    assert(it != op_to_sequence_number_.end() &&
           "op not in the sequence analysis");
    return it->getSecond();
  }

  // Returns the sequence number of the last explicitly sequenceable op that
  // (transitively) produces the operands for this implicitly sequenceable op.
  // In other words, the given op should be sequenced between result and
  // result+1.
  int64_t ImplicitSequenceNumber(const OpInstance &op) const;

  // Sequence state: the position in the vector indicates the sequence number of
  // the operation.
  llvm::SmallVector<ComputeOpInstance> compute_ops_;

  // Lookup cache for the position of the (compute) operation in the vector.
  llvm::DenseMap<ComputeOpInstance, int64_t> op_to_sequence_number_;

  // List of "fby" operations that create a use-def cycle, which can be removed
  // by dropping the use-def edge entering into their "value" operand.
  llvm::SmallVector<OpInstance> fby_ops_to_cut_;
};

}  // namespace sair

#endif  // SAIR_SEQUENCE_H_
