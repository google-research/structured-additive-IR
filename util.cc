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

#include "util.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OperationSupport.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"

namespace sair {
namespace {

// Helper class to build range parameters.
class RangeParameterBuilder {
 public:
  // Creates a builder that will compute range parameters by inserting
  // operations in `body` and adding arguments to `arguments`, where `body` is
  // the body of a sair.map operation with `arguments` passed as arguments.
  // `current_to_source` is a mapping from the domain of the sair.map operation
  // to source_domain.
  RangeParameterBuilder(mlir::Location loc,
                        llvm::ArrayRef<ValueAccess> source_domain,
                        MappingAttr current_to_source, MapBodyBuilder &body,
                        mlir::OpBuilder &builder);

  // Size of the domain of the current operation.
  int current_domain_size() const { return current_to_source_.UseDomainSize(); }

  // Adds an argument to the current operation and return the corresponding
  // scalar value in the operation body.
  mlir::OpFoldResult AddArgument(const ValueOrConstant &value);

  // Returns parameters for the dimension obtained by applying expr to the
  // source domain.
  RangeParameters Get(MappingExpr expr);
  RangeParameters Get(MappingDimExpr expr);
  RangeParameters Get(MappingStripeExpr expr);
  RangeParameters Get(MappingUnStripeExpr expr);

 private:
  mlir::Location loc_;
  llvm::ArrayRef<ValueAccess> source_domain_;
  MappingAttr current_to_source_;
  MapBodyBuilder &body_;
  mlir::OpBuilder &builder_;
};

RangeParameterBuilder::RangeParameterBuilder(
    mlir::Location loc, llvm::ArrayRef<ValueAccess> source_domain,
    MappingAttr current_to_source, MapBodyBuilder &body,
    mlir::OpBuilder &builder)
    : loc_(loc),
      source_domain_(source_domain),
      current_to_source_(current_to_source),
      body_(body),
      builder_(builder) {}

mlir::OpFoldResult RangeParameterBuilder::AddArgument(
    const ValueOrConstant &value) {
  if (value.is_constant()) return value.constant();
  return body_.AddOperand(value.value());
}

RangeParameters RangeParameterBuilder::Get(MappingExpr expr) {
  return mlir::TypeSwitch<MappingExpr, RangeParameters>(expr)
      .Case<MappingDimExpr>([this](MappingDimExpr expr) { return Get(expr); })
      .Case<MappingStripeExpr>(
          [this](MappingStripeExpr expr) { return Get(expr); })
      .Case<MappingUnStripeExpr>(
          [this](MappingUnStripeExpr expr) { return Get(expr); });
}

RangeParameters RangeParameterBuilder::Get(MappingDimExpr expr) {
  const ValueAccess &dimension = source_domain_[expr.dimension()];
  auto range_op = mlir::cast<RangeOp>(dimension.value.getDefiningOp());
  MappingAttr mapping = current_to_source_.Compose(
      dimension.mapping.ResizeUseDomain(current_to_source_.size()));
  assert(mapping.IsSurjective());

  return {.begin = AddArgument(range_op.LowerBound().Map(mapping)),
          .end = AddArgument(range_op.UpperBound().Map(mapping)),
          .step = range_op.Step()};
}

RangeParameters RangeParameterBuilder::Get(MappingStripeExpr expr) {
  // Compute range parameters for the operand.
  RangeParameters operand_parameters = Get(expr.operand());
  int step = expr.factors().back() * operand_parameters.step;

  // If the stripe covers the entire operand range, no additional
  // computation is needed.
  if (expr.factors().size() == 1) {
    return {operand_parameters.begin, operand_parameters.end, step};
  }
  int size = expr.factors()[expr.factors().size() - 2];

  // Compute the begin index. For this, look for the unstripe operation
  // corresponding to `this` in the inverse mapping, and find the
  // expression of the outer stripe dimension.
  auto inverse_expr = expr.operand()
                          .FindInInverse(current_to_source_.Dimensions())
                          .cast<MappingUnStripeExpr>();
  auto begin_map = mlir::AffineMap::get(
      current_domain_size(), 0,
      inverse_expr.operands()[expr.factors().size() - 2].AsAffineExpr());
  mlir::Value begin =
      builder_.create<mlir::AffineApplyOp>(loc_, begin_map, body_.indices());

  // Compute the end index as `min(begin + size, operand_size)`.
  mlir::Type index_type = builder_.getIndexType();
  auto size_op = builder_.create<mlir::ConstantOp>(
      loc_, index_type, builder_.getIndexAttr(size * operand_parameters.step));
  auto uncapped_end =
      builder_.create<mlir::AddIOp>(loc_, index_type, begin, size_op);
  mlir::Value operand_end;
  if (operand_parameters.end.is<mlir::Attribute>()) {
    operand_end = builder_.create<mlir::ConstantOp>(
        loc_, index_type, operand_parameters.end.get<mlir::Attribute>());
  } else {
    operand_end = operand_parameters.end.get<mlir::Value>();
  }
  auto is_capped = builder_.create<mlir::CmpIOp>(loc_, CmpIPredicate::ult,
                                                 operand_end, uncapped_end);
  mlir::Value end = builder_.create<mlir::SelectOp>(
      loc_, builder_.getIndexType(), is_capped, operand_end, uncapped_end);

  return {begin, end, step};
}

RangeParameters RangeParameterBuilder::Get(MappingUnStripeExpr expr) {
  RangeParameters params = Get(expr.operands()[0]);
  params.step /= expr.factors().front();
  return params;
}

}  // namespace

void InsertionPoint::Set(mlir::OpBuilder &builder) const {
  if (direction == Direction::kAfter) {
    builder.setInsertionPointAfter(operation);
  } else {
    builder.setInsertionPoint(operation);
  }
}

void ForwardAttributes(mlir::Operation *old_op, mlir::Operation *new_op,
                       llvm::ArrayRef<llvm::StringRef> ignore) {
  for (auto [name, attr] : old_op->getAttrs()) {
    if (new_op->hasAttr(name) || llvm::find(ignore, name) != ignore.end())
      continue;
    new_op->setAttr(name, attr);
  }
}

void SetInArrayAttr(mlir::Operation *operation, llvm::StringRef attr_name,
                    int array_size, int element, mlir::Attribute value) {
  mlir::MLIRContext *context = operation->getContext();
  llvm::SmallVector<mlir::Attribute, 4> values;

  auto old_attr = operation->getAttr(attr_name);
  if (old_attr == nullptr) {
    values.resize(array_size, mlir::UnitAttr::get(context));
  } else {
    auto old_array_attr = old_attr.cast<mlir::ArrayAttr>();
    assert(old_array_attr.size() == array_size);
    llvm::append_range(values, old_array_attr.getValue());
  }

  values[element] = value;
  operation->setAttr(attr_name, mlir::ArrayAttr::get(context, values));
}

mlir::Value Materialize(mlir::Location loc, mlir::OpFoldResult value,
                        mlir::OpBuilder &builder) {
  if (value.is<mlir::Value>()) return value.get<mlir::Value>();
  return builder.create<mlir::ConstantOp>(loc, value.get<mlir::Attribute>());
}

llvm::SmallVector<RangeParameters> GetRangeParameters(
    mlir::Location loc, MappingAttr mapping,
    llvm::ArrayRef<ValueAccess> source_domain, MappingAttr current_to_source,
    MapBodyBuilder &current_body, mlir::OpBuilder &builder) {
  assert(mapping.IsSurjective());
  assert(mapping.IsFullySpecified());
  assert(mapping.UseDomainSize() == source_domain.size());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&current_body.block());
  RangeParameterBuilder param_builder(loc, source_domain, current_to_source,
                                      current_body, builder);
  llvm::SmallVector<RangeParameters> range_parameters;
  for (MappingExpr expr : mapping) {
    range_parameters.push_back(param_builder.Get(expr));
  }
  return range_parameters;
}

//===----------------------------------------------------------------------===//
// MapBodyBuilder
//===----------------------------------------------------------------------===//

MapBodyBuilder::MapBodyBuilder(int domain_size, mlir::MLIRContext *context)
    : domain_size_(domain_size) {
  auto *block = new Block();
  region_.getBlocks().insert(region_.end(), block);
  auto index_type = mlir::IndexType::get(context);
  for (int i = 0; i < domain_size; ++i) {
    block->addArgument(index_type);
  }
}

mlir::Value MapBodyBuilder::index(int dimension) {
  assert(dimension < domain_size_);
  return block().getArgument(dimension);
}

mlir::ValueRange MapBodyBuilder::indices() {
  return block().getArguments().take_front(domain_size_);
}

mlir::Value MapBodyBuilder::block_input(int position) {
  assert(position < operands_.size());
  return block().getArgument(domain_size_ + position);
}

mlir::ValueRange MapBodyBuilder::block_inputs() {
  return block().getArguments().drop_front(domain_size_);
}

mlir::Value MapBodyBuilder::AddOperand(ValueAccess operand) {
  operands_.push_back(operand);
  auto value_type = operand.value.getType().cast<ValueType>();
  return block().addArgument(value_type.ElementType());
}

}  // namespace sair
