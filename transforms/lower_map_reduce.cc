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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_ops.h"
#include "transforms/lowering_pass_classes.h"
#include "util.h"

namespace sair {
namespace {

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
  llvm::append_range(domain, parallel_domain);
  llvm::append_range(domain, reduction_domain);

  // Split the mapping array into "initalizer" and "input" parts.
  llvm::ArrayRef<mlir::Attribute> op_mappings = op.mapping_array().getValue();
  auto init_mappings = op_mappings.drop_back(op.inputs().size());
  auto input_mappings = op_mappings.take_back(op.inputs().size());

  // Assume the value produced by sair.fby is always indexed using the identity
  // mapping (there is no control of the output mapping, so just make sure the
  // value is read in the identity order as it was produced).
  auto identity_mapping = MappingAttr::GetIdentity(ctx, domain.size());

  // For each initializer, create a separate sair.fby operation. The results of
  // these operations will be the leading arguments of sair.map in order to keep
  // the order of arguments in its entry block the same as in sair.map_reduce.
  SmallVector<SairFbyOp, 4> fbys;
  SmallVector<Value, 4> map_operands;
  fbys.reserve(op.getNumResults());
  map_operands.reserve(op.getNumResults() + op.inputs().size());
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    Value init_value = op.inits()[i];
    auto mapping_attr =
        builder.getArrayAttr({init_mappings[i], identity_mapping});
    // This produces a value that of the same rank as the domain.
    auto fby_type = ValueType::get(
        op.shape(), init_value.getType().cast<ValueType>().ElementType());

    // Use `init_value` as both arguments temporarily, the second argument will
    // be updated later.
    auto fby = builder.create<SairFbyOp>(
        loc, fby_type, parallel_domain, reduction_domain, mapping_attr,
        init_value, init_value, /*copies=*/nullptr);
    fbys.push_back(fby);
    map_operands.push_back(fby.getResult());
  }

  // Forward sair.map_reduce inputs as trailing arguments of the sair.map.
  llvm::append_range(map_operands, op.inputs());

  // The values produced by sair.fby are accessed using identity mappings and
  // the original inputs retain their mappings.
  SmallVector<mlir::Attribute, 4> mappings(fbys.size(), identity_mapping);
  llvm::append_range(mappings, input_mappings);
  auto map_mapping = builder.getArrayAttr(mappings);

  // The shapes of new result types are same as the op shapes since we
  // reintroduced the reduction dimensions in them.
  auto result_types = llvm::to_vector<4>(
      llvm::map_range(op.getResultTypes(), [&](mlir::Type type) -> mlir::Type {
        return ValueType::get(op.shape(), type.cast<ValueType>().ElementType());
      }));

  // Keep memory space undefined for the produced value.
  auto map = builder.create<SairMapOp>(loc, result_types, domain, map_mapping,
                                       map_operands, op.shape(),
                                       op.decisionsAttr(), /*copies=*/nullptr);
  map.getRegion().takeBody(op.getRegion());

  // For each original result of sair.map_reduce, create a sair.proj_last that
  // only retains the final value of the reduction and use its result instead.
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    // Close the cycling definition of the sair.fby op.
    fbys[i].Value().set_value(map.getResult(i));

    auto copies = builder.getArrayAttr(op.GetCopies(i));
    auto proj = builder.create<SairProjLastOp>(
        loc, op.getResultTypes()[i], op.parallel_domain(),
        op.reduction_domain(), builder.getArrayAttr(identity_mapping),
        map.getResult(i), op.shape(),
        /*copies=*/builder.getArrayAttr({copies}));
    op.results()[i].replaceAllUsesWith(proj.result());
  }

  op.erase();
}

class LowerMapReduce : public LowerMapReducePassBase<LowerMapReduce> {
  // Converts
  //
  // <res> = sair.map_reduce[<D0>] <inits> reduce[<D1>] <values> <body>
  //
  // into
  //
  // <tmp0> = sair.fby[<D0>] <inits> then[<D1>] <tmp1>
  // <tmp1> = sair.map[<D0>, <D1>] <tmp0>, <values> <body>
  // <res> = sair.proj_last[<D0>] last[<D1>] <tmp1>
  void runOnFunction() override {
    mlir::MLIRContext *context = &getContext();
    getFunction().walk([context](SairMapReduceOp op) {
      mlir::OpBuilder builder(context);
      builder.setInsertionPoint(op);
      RewriteMapReduceToMap(op, builder);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateLowerMapReducePass() {
  return std::make_unique<LowerMapReduce>();
}

}  // namespace sair
