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

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "sair_attributes.h"
#include "sair_ops.h"
#include "transforms/lowering_pass_classes.h"
#include "utils.h"

namespace sair {
namespace {

// Lowers `op` into a sair.map operation that takes a single value as argument
// and returns it immediately.
void RewriteCopyToMap(SairCopyOp op, mlir::OpBuilder &builder) {
  mlir::OperandRange inputs = op.getOperands().slice(op.domain().size(), 1);
  SairMapOp new_op = builder.create<SairMapOp>(
      op.getLoc(), op.getType(), op.domain(), op.access_pattern_array(), inputs,
      op.shape(), op.loop_nestAttr(), op.memory_spaceAttr());

  // Set the block of code in the sair.map operation to return its argument
  // unchanged.
  builder.setInsertionPointToStart(&new_op.block());
  builder.create<SairReturnOp>(op.getLoc(), new_op.block_inputs().front());
  op.result().replaceAllUsesWith(new_op.getResult(0));
  op.erase();
}

// Lowers a map-reduce `op` into a sair.map by emitting a sair.fby to tie
// together the initial value and partial reduction values, and a sair.proj_last
// to only retain the final reduction value.
void RewriteMapReduceToMap(SairMapReduceOp op, mlir::OpBuilder &builder) {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto parallel_domain = op.parallel_domain();
  auto reduction_domain = op.reduction_domain();

  // The domain of the final sair.map is a concatenation of the parallel and
  // reduction domains of the sair.map_reduce.
  llvm::SmallVector<Value, 8> domain;
  domain.reserve(parallel_domain.size() + reduction_domain.size());
  appendRange(appendRange(domain, parallel_domain), reduction_domain);

  // Split the access pattern array into "initalizer" and "input" parts.
  llvm::ArrayRef<mlir::Attribute> op_access_patterns =
      op.access_pattern_array().getValue();
  auto init_access_patterns = op_access_patterns.drop_back(op.inputs().size());
  auto input_access_patterns = op_access_patterns.take_back(op.inputs().size());

  // Assume the value produced by sair.fby is always indexed using the identity
  // pattern (there is no control of the output pattern, so just make sure the
  // value is read in the identity order as it was produced).
  auto identity_access_pattern =
      AccessPatternAttr::GetIdentity(ctx, domain.size());

  // For each initializer, create a separate sair.fby operation. The results of
  // these operations will be the leading arguments of sair.map in order to keep
  // the order of arguments in its entry block the same as in sair.map_reduce.
  SmallVector<SairFbyOp, 4> fbys;
  SmallVector<Value, 4> map_operands;
  fbys.reserve(op.getNumResults());
  map_operands.reserve(op.getNumResults() + op.inputs().size());
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    Value init_value = op.inits()[i];
    auto access_pattern_attr = builder.getArrayAttr(
        {init_access_patterns[i], identity_access_pattern});
    // This produces a value that of the same rank as the domain.
    auto fby_type = ValueType::get(
        ctx, op.shape(), init_value.getType().cast<ValueType>().ElementType());

    // Use `init_value` as both arguments temporarily, the second argument will
    // be updated later. Keep memory space undefined for the produced value.
    auto fby = builder.create<SairFbyOp>(loc, fby_type, parallel_domain,
                                         reduction_domain, access_pattern_attr,
                                         init_value, init_value, nullptr);
    fbys.push_back(fby);
    map_operands.push_back(fby.getResult());
  }

  // Forward sair.map_reduce inputs as trailing arguments of the sair.map.
  appendRange(map_operands, op.inputs());

  // The values produced by sair.fby are accessed using identity patterns and
  // the original inputs retain their patterns.
  SmallVector<mlir::Attribute, 4> access_patterns(fbys.size(),
                                                  identity_access_pattern);
  appendRange(access_patterns, input_access_patterns);
  auto map_access_pattern = builder.getArrayAttr(access_patterns);

  // The shapes of new result types are same as the op shapes since we
  // reintroduced the reduction dimensions in them.
  auto result_types = llvm::to_vector<4>(
      llvm::map_range(op.getResultTypes(), [&](mlir::Type type) -> mlir::Type {
        return ValueType::get(type.getContext(), op.shape(),
                              type.cast<ValueType>().ElementType());
      }));

  // Keep memory space undefined for the produced value.
  auto map = builder.create<SairMapOp>(loc, result_types, domain,
                                       map_access_pattern, map_operands,
                                       op.shape(), op.loop_nestAttr(), nullptr);
  map.getRegion().takeBody(op.getRegion());

  // For each original result of sair.map_reduce, create a sair.proj_last that
  // only retains the final value of the reduction and use its result instead.
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    // Close the cycling definition of the sair.fby op.
    fbys[i].Value().set_value(map.getResult(i));

    // Place this result into the same memory space as we need to respect the
    // previous choice, unlike the temporaries we introduced above.
    mlir::ArrayAttr memory_space =
        op.IsMemorySpaceSet(i) ? builder.getI64ArrayAttr(*op.GetMemorySpace(i))
                               : nullptr;
    auto proj = builder.create<SairProjLastOp>(
        loc, op.getResultTypes()[i], op.parallel_domain(),
        op.reduction_domain(), builder.getArrayAttr(identity_access_pattern),
        map.getResult(i), op.shape(), memory_space);
    op.results()[i].replaceAllUsesWith(proj.result());
  }
  op.erase();
}

class LowerToMap : public LowerToMapPassBase<LowerToMap> {
  // Converts sair.copy operations into sair.map operations. This is a hook for
  // the MLIR pass infrastructure.
  void runOnFunction() override {
    mlir::MLIRContext *context = &getContext();
    getFunction().walk([context](Operation *op) {
      mlir::OpBuilder builder(context);
      builder.setInsertionPoint(op);
      if (auto copy = dyn_cast<SairCopyOp>(op)) {
        RewriteCopyToMap(copy, builder);
      } else if (auto reduce = dyn_cast<SairMapReduceOp>(op)) {
        RewriteMapReduceToMap(reduce, builder);
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerToMapPass() {
  return std::make_unique<LowerToMap>();
}

}  // namespace sair
