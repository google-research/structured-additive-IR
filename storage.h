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

#ifndef SAIR_STORAGE_H_
#define SAIR_STORAGE_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "loop_nest.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"

namespace sair {

// Verifies that storage attributes in the program are correct. Assumes that
// Sair operands are defined in the same program.
mlir::LogicalResult VerifyStorages(
    SairProgramOp program, const IterationSpaceAnalysis &iteration_spaces);

// Returns the buffer attribute representing a 0-dimensional register.
BufferAttr GetRegister0DBuffer(mlir::MLIRContext *context);

// A buffer declared by one or more storage attributes.
class Buffer {
 public:
  // Create a new buffer written to by the given operation. The operation must
  // have a loop_nest attribute set. `result` is the position of `op` result
  // stored in `buffer`.
  Buffer(mlir::Type element_type, int rank, ComputeOp op, int result,
         const LoopFusionAnalysis &fusion_analysis);
  Buffer(FromToMemRefOp import_op,
         const IterationSpaceAnalysis &iteration_spaces,
         const LoopFusionAnalysis &fusion_analysis);

  // Number of dimensions in the buffer layout.
  int rank() const { return layout_.size(); }

  // Types of the scalars stored in the buffer.
  mlir::Type element_type() const { return element_type_; }

  // Loop nest in which the buffer is defined.
  llvm::ArrayRef<mlir::StringAttr> loop_nest() const { return loop_nest_; }

  // Domain from which the buffer size is derived.
  llvm::ArrayRef<ValueAccess> domain() const { return domain_; }
  llvm::SmallVectorImpl<ValueAccess> &domain() { return domain_; }

  // Mapping from domain dimensions to buffer dimensions.
  llvm::ArrayRef<MappingExpr> layout() const { return layout_; }

  // Indicates if the buffer is declared outside the Sair program.
  bool is_external() const { return import_op_ != nullptr; }

  // In the case where `is_external` is true, operation that imports the memref
  // in the sair program.
  FromToMemRefOp import_op() const { return import_op_; }

  // List of operations that write to the buffer, with the position of the
  // result stored in the buffer. Non-external buffers must have at least one
  // write.
  llvm::ArrayRef<std::pair<ComputeOp, int>> writes() const { return writes_; }

  // List of operations that read from the buffer, with the position of the Sair
  // value operand.
  llvm::ArrayRef<std::pair<ComputeOp, int>> reads() const { return reads_; }

  // Get the location of the first operation defining the buffer.
  mlir::Location getLoc() const { return loc_; }

  // Mapping of domain to layout prefixed by loop nest iterators. The prefix
  // corresponds to the different instances of the buffer.
  MappingAttr PrefixedLayout() const;

  // Registers an operation writting to the buffer.
  void AddWrite(ComputeOp op, int result);

  // Registers an operation reading the buffer.
  void AddRead(ComputeOp op, int operand);

  // Trims the loop-nest to the given size.
  void TrimLoopNest(int new_size);

  // Unifies a dimension of the layout with another expression.
  void UnifyLayoutDim(int layout_dim, MappingExpr expr);

 private:
  mlir::Location loc_;
  mlir::Type element_type_;
  FromToMemRefOp import_op_ = nullptr;

  llvm::SmallVector<mlir::StringAttr> loop_nest_;
  MappingAttr loop_nest_mapping_;

  llvm::SmallVector<ValueAccess> domain_;
  llvm::SmallVector<MappingExpr> layout_;
  llvm::SmallVector<std::pair<ComputeOp, int>> writes_;
  llvm::SmallVector<std::pair<ComputeOp, int>> reads_;
};

// Computes buffers metadata and storage information for each value.
class StorageAnalysis {
 public:
  // Creates and populates the analysis. `operation` must be a sair.program
  // operation. Asserts that the analysis succeeded.
  explicit StorageAnalysis(mlir::Operation *operation);

  // Creates and populates the analysis. Returns `nullopt` and emits an error if
  // the analysis fails because storage attributes are invalid.
  static std::optional<StorageAnalysis> Create(SairProgramOp program);

  // Retrieves the analysis result for a buffer.
  const Buffer &GetBuffer(mlir::StringAttr buffer) const {
    return buffers_.find(buffer)->second;
  }

  // List of buffers indexed by name.
  const llvm::DenseMap<mlir::Attribute, Buffer> &buffers() const {
    return buffers_;
  }

  // Retrieves the storage of a value.
  const ValueStorage &GetStorage(mlir::Value value) const {
    return value_storages_.find(value)->second;
  }

  // Returns a fresh buffer name. May be called multiple times without
  // invalidating the analysis.
  mlir::StringAttr GetFreshBufferName();

 private:
  // Creates an empty analysis.
  StorageAnalysis(mlir::MLIRContext *context) : context_(context){};

  // Populates the analysis.
  mlir::LogicalResult Init(SairProgramOp program);

  mlir::MLIRContext *context_;
  int next_buffer_id_ = 0;
  llvm::DenseMap<mlir::Attribute, Buffer> buffers_;
  llvm::DenseMap<mlir::Value, ValueStorage> value_storages_;
};

}  // namespace sair

#endif  // SAIR_STORAGE_H_
