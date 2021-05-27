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

#include <map>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/iterator_range.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"

namespace sair {

// A set of ops of OpTy that preserves the insertion order. Practically, this is
// an llvm::SetVector with additional casting to OpTy since llvm::SetVector
// (precisely, llvm::DenseSet inside it) cannot be constructed for op interface
// classes because their constructors need a non-null operation.
template <typename OpTy>
class ConcreteOpSet {
 public:
  ConcreteOpSet() {}

  // Inserts into the set.
  bool insert(OpTy op) { return contents_.insert(op.getOperation()); }
  template <typename Iterator>
  void insert(Iterator first, Iterator last) {
    contents_.insert(first, last);
  }

  // Merges the given set of ops into the this set of ops.
  void merge(const ConcreteOpSet<OpTy> &other) {
    contents_.insert(other.contents_.begin(), other.contents_.end());
  }

  // Returns `true` if the set has no elements.
  bool empty() const { return contents_.empty(); }

  // Returns the number of ops in this set.
  size_t size() const { return contents_.size(); }

  // Returns `true` if the set contains the given element.
  bool contains(OpTy op) const { return contents_.contains(op.getOperation()); }

  // Removes the most recently added unique element from the set and returns it.
  OpTy pop_back_val() { return cast<OpTy>(contents_.pop_back_val()); }

  // Returns the most recently added unique element of the set.
  OpTy back() const { return cast<OpTy>(contents_.back()); };

  // Returns an iterator range over the elements.
  auto Ops() const {
    return llvm::map_range(contents_,
                           [](Operation *op) { return cast<OpTy>(op); });
  }

  // Erases the given element from the set.
  void erase(OpTy op) { contents_.remove(op.getOperation()); }

 private:
  llvm::SetVector<Operation *> contents_;
};

using ComputeOpSet = ConcreteOpSet<ComputeOp>;

// An analysis of the relative positions of Sair operations indicated by their
// sequence attributes.
class SequenceAnalysis {
 public:
  // We use a standard multimap because (a) the sequence numbers can be shared
  // and (b) we need a deterministic increasing order that is provided by this
  // map and not provided by hash table-based maps.
  using MapType = std::multimap<int64_t, ComputeOp>;
  using ConstRangeType = llvm::iterator_range<MapType::const_iterator>;

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
  ConstRangeType Ops() const;

  // Returns an iterator range of all operations sequenced before the given one,
  // in their relative order. All operations are given a relative order even if
  // they don't have a sequence attribute attached. The sequence number returned
  // in this iteration may differ from that of the sequence attribute if the
  // Sair program hasn't been canonicalized.
  ConstRangeType OpsBefore(ComputeOp op) const;

 private:
  // Default noop constructor. Init must be called separately.
  SequenceAnalysis() = default;

  // Initializes the analysis for the given program op. This may fail if the
  // program contains use-def loops between compute operations (loops are
  // allowed only through the non-compute by operation).
  mlir::LogicalResult Init(SairProgramOp program_op, bool report_errors);

  // Updates `sequenced_ops_` to have sequence numbers for all compute
  // operations in the program, inferring their relative order from the
  // available sequence attribtues and use-def chains. The relative order is
  // preserved but not the absolute sequence numbers. The traversal order is
  // deterministic but otherwise unspecified for operations that do not have
  // "sequence" attribute and belong to different connected components of the
  // use-def dependency graph.
  mlir::LogicalResult ComputeDefaultSequence(SairProgramOp program,
                                             bool report_errors);

  MapType sequenced_ops_;
  llvm::SmallVector<SairFbyOp> fby_ops_to_cut_;
};

}  // namespace sair

#endif  // SAIR_SEQUENCE_H_
