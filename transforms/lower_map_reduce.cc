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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_ops.h"

namespace sair {

#define GEN_PASS_DEF_LOWERMAPREDUCEPASS
#include "transforms/lowering.h.inc"

namespace {

// Creates an `instances` attribute (array of decisions) that has as many
// entries as the `instances` attribute of `op` and only has the operands field
// filled. The callback receives as input the instance position, the list of
// operand attributes of the corresponding instance of `op`, and a vector to
// populate with new operand attributes. If the vector remains empty, the
// operands field will be set to nullptr.
mlir::ArrayAttr CreateOperandsOnlyInstances(
    SairMapReduceOp op,
    llvm::function_ref<void(int, llvm::ArrayRef<mlir::Attribute>,
                            llvm::SmallVectorImpl<mlir::Attribute> &)>
        callback) {
  mlir::MLIRContext *ctx = op->getContext();
  SmallVector<mlir::Attribute> instances;
  instances.reserve(op.NumInstances());
  for (unsigned i = 0, e = op.NumInstances(); i < e; ++i) {
    DecisionsAttr map_reduce_decisions = op.GetDecisions(i);
    SmallVector<mlir::Attribute> operand_attrs;
    if (map_reduce_decisions.operands() != nullptr) {
      callback(i, map_reduce_decisions.operands().getValue(), operand_attrs);
    }
    auto operands_attr = operand_attrs.empty()
                             ? mlir::ArrayAttr()
                             : mlir::ArrayAttr::get(ctx, operand_attrs);
    auto decisions = DecisionsAttr::get(
        /*sequence=*/nullptr, /*loop_nest=*/nullptr,
        /*storage=*/nullptr, /*expansion=*/nullptr,
        /*copy_of=*/nullptr,
        /*operands=*/operands_attr, ctx);
    instances.push_back(decisions);
  }
  return mlir::ArrayAttr::get(ctx, instances);
}

// Lowers a map-reduce `op` into a sair.map by emitting a sair.fby to tie
// together the initial value and partial reduction values, and a sair.proj_last
// to only retain the final reduction value.
void RewriteMapReduceToMap(SairMapReduceOp op, mlir::OpBuilder &builder) {
  MLIRContext *ctx = op.getContext();
  Location loc = op.getLoc();
  auto parallel_domain = op.getParallelDomain();
  auto reduction_domain = op.getReductionDomain();

  // The domain of the final sair.map is a concatenation of the parallel and
  // reduction domains of the sair.map_reduce.
  llvm::SmallVector<Value, 8> domain;
  domain.reserve(parallel_domain.size() + reduction_domain.size());
  llvm::append_range(domain, parallel_domain);
  llvm::append_range(domain, reduction_domain);

  // Split the mapping array into "initalizer" and "input" parts.
  llvm::ArrayRef<mlir::Attribute> op_mappings = op.getMappingArray().getValue();
  auto init_mappings = op_mappings.drop_back(op.getInputs().size());
  auto input_mappings = op_mappings.take_back(op.getInputs().size());

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
  map_operands.reserve(op.getNumResults() + op.getInputs().size());
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    Value init_value = op.getInits()[i];
    auto mapping_attr =
        builder.getArrayAttr({init_mappings[i], identity_mapping});
    // This produces a value that of the same rank as the domain.
    auto fby_type = ValueType::get(
        op.getShape(), init_value.getType().cast<ValueType>().ElementType());

    // Same operands as in map_reduce are used for domain dimensions and the
    // init. The fby value is taken from the instance with the same position as
    // the current instance to match the map constructed below.
    mlir::ArrayAttr instances = CreateOperandsOnlyInstances(
        op, [&](int instance, llvm::ArrayRef<mlir::Attribute> old_operand_attrs,
                llvm::SmallVectorImpl<mlir::Attribute> &operand_attrs) {
          int domain_size = op.domain().size();
          llvm::append_range(operand_attrs,
                             old_operand_attrs.take_front(domain_size));
          operand_attrs.append({old_operand_attrs[domain_size + i],
                                InstanceAttr::get(ctx, instance)});
        });

    // Use `init_value` as both arguments temporarily, the second argument will
    // be updated later.
    auto fby = builder.create<SairFbyOp>(
        loc, fby_type, parallel_domain, reduction_domain, mapping_attr,
        init_value, init_value, instances, /*copies=*/nullptr);
    fbys.push_back(fby);
    map_operands.push_back(fby.getResult());
  }

  // Forward sair.map_reduce inputs as trailing arguments of the sair.map.
  llvm::append_range(map_operands, op.getInputs());

  // The values produced by sair.fby are accessed using identity mappings and
  // the original inputs retain their mappings.
  SmallVector<mlir::Attribute, 4> mappings(fbys.size(), identity_mapping);
  llvm::append_range(mappings, input_mappings);
  auto map_mapping = builder.getArrayAttr(mappings);

  // The shapes of new result types are same as the op shapes since we
  // reintroduced the reduction dimensions in them.
  auto result_types = llvm::to_vector<4>(
      llvm::map_range(op.getResultTypes(), [&](mlir::Type type) -> mlir::Type {
        return ValueType::get(op.getShape(),
                              type.cast<ValueType>().ElementType());
      }));

  // The new map op instances similar to that of map_reduce, but the operands
  // must be adapted. The domain and input operands are the same, and the
  // operands that correspond to fby values refer to a co-indexed instance of
  // the corresponding fby op.
  SmallVector<mlir::Attribute> map_instances;
  map_instances.reserve(op.NumInstances());
  for (unsigned i = 0, e = op.NumInstances(); i < e; ++i) {
    DecisionsAttr old_decisions = op.GetDecisions(i);
    if (old_decisions.operands() == nullptr) {
      map_instances.push_back(old_decisions);
      continue;
    }

    SmallVector<mlir::Attribute> operand_attrs;
    operand_attrs.reserve(map_operands.size());
    llvm::append_range(
        operand_attrs,
        old_decisions.operands().getValue().take_front(domain.size()));
    operand_attrs.append(fbys.size(), InstanceAttr::get(ctx, i));
    llvm::append_range(
        operand_attrs,
        old_decisions.operands().getValue().take_back(op.getInputs().size()));

    auto decisions = DecisionsAttr::get(
        old_decisions.sequence(), old_decisions.loop_nest(),
        old_decisions.storage(), old_decisions.expansion(),
        old_decisions.copy_of(), mlir::ArrayAttr::get(ctx, operand_attrs), ctx);
    map_instances.push_back(decisions);
  }

  auto map = builder.create<SairMapOp>(
      loc, result_types, domain, map_mapping, map_operands, op.getShape(),
      mlir::ArrayAttr::get(ctx, map_instances), /*copies=*/nullptr);
  map.getRegion().takeBody(op.getRegion());

  // For each original result of sair.map_reduce, create a sair.proj_last that
  // only retains the final value of the reduction and use its result instead.
  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    // Close the cycling definition of the sair.fby op.
    fbys[i].Value().set_value(map.getResult(i));

    // Same operands as map_reduce are used for domain dimensions in
    // projections. Values are taken from the co-indexed instance of map.
    mlir::ArrayAttr instances = CreateOperandsOnlyInstances(
        op, [&](int instance, llvm::ArrayRef<mlir::Attribute> old_operand_attrs,
                llvm::SmallVectorImpl<mlir::Attribute> &operand_attrs) {
          int domain_size = op.domain().size();
          llvm::append_range(operand_attrs,
                             old_operand_attrs.take_front(domain_size));
          operand_attrs.push_back(InstanceAttr::get(ctx, instance));
        });

    auto copies = builder.getArrayAttr(op.GetCopies(i));
    auto proj = builder.create<SairProjLastOp>(
        loc, op.getResultTypes()[i], op.getParallelDomain(),
        op.getReductionDomain(), builder.getArrayAttr(identity_mapping),
        map.getResult(i), op.getShape(), instances,
        /*copies=*/builder.getArrayAttr({copies}));
    op.getResults()[i].replaceAllUsesWith(proj.getResult());
  }

  op.erase();
}

class LowerMapReduce : public impl::LowerMapReducePassBase<LowerMapReduce> {
  // Converts
  //
  // <res> = sair.map_reduce[<D0>] <inits> reduce[<D1>] <values> <body>
  //
  // into
  //
  // <tmp0> = sair.fby[<D0>] <inits> then[<D1>] <tmp1>
  // <tmp1> = sair.map[<D0>, <D1>] <tmp0>, <values> <body>
  // <res> = sair.proj_last[<D0>] last[<D1>] <tmp1>
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    getOperation().walk([context](SairMapReduceOp op) {
      mlir::OpBuilder builder(context);
      builder.setInsertionPoint(op);
      RewriteMapReduceToMap(op, builder);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLowerMapReducePass() {
  return std::make_unique<LowerMapReduce>();
}

}  // namespace sair
