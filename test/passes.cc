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

#include "test/passes.h"

#include "mlir/IR/Builders.h"
#include "sair_attributes.h"
#include "sair_dialect.h"

namespace sair {

#define GEN_PASS_CLASSES
#include "test/passes.h.inc"

// Retrieves the attribute `name` from `op` and converts it into a vector of
// `T`.
template <typename T>
static llvm::SmallVector<T, 4> GetAttrVector(llvm::StringRef name,
                                             mlir::Operation *op) {
  auto array = op->getAttrOfType<mlir::ArrayAttr>(name);
  assert(array != nullptr);
  llvm::SmallVector<T, 4> vector;
  vector.reserve(array.size());
  for (mlir::Attribute element : array.getValue()) {
    vector.push_back(element.cast<T>());
  }
  return vector;
}

// Converts an array of attributes of type `T` to an array attribute.
template <typename T>
mlir::ArrayAttr GetArrayAttr(llvm::SmallVectorImpl<T> &attributes,
                             mlir::Builder &builder) {
  llvm::SmallVector<mlir::Attribute, 4> array(attributes.begin(),
                                              attributes.end());
  return builder.getArrayAttr(array);
}

// Walks a module and dispatch each operation to an access pattern expression
// method call based on the operation name.
class TestAccessPatternExprsPass
    : public TestAccessPatternExprsPassBase<TestAccessPatternExprsPass> {
 public:
  mlir::Attribute DispatchTest(llvm::StringRef op_name, AccessPatternExpr expr,
                               mlir::Operation *op) {
    mlir::Builder builder(&getContext());
    if (op_name == "is_fully_specified") {
      return builder.getBoolAttr(expr.IsFullySpecified());
    } else if (op_name == "make_fully_specified") {
      int num_dimensions = expr.MinDomainSize();
      return expr.MakeFullySpecified(num_dimensions);
    } else if (op_name == "substitute_dims") {
      return expr.SubstituteDims(
          GetAttrVector<AccessPatternExpr>("substitutions", op));
    } else if (op_name == "set_inverse") {
      llvm::SmallVector<AccessPatternExpr, 4> inverses =
          GetAttrVector<AccessPatternExpr>("inverses", op);
      auto context = op->getAttrOfType<AccessPatternExpr>("context");
      assert(context != nullptr);
      if (mlir::failed(expr.SetInverse(context, inverses))) return nullptr;
      return GetArrayAttr(inverses, builder);
    } else if (op_name == "unify") {
      auto other = op->getAttrOfType<AccessPatternExpr>("other");
      assert(other != nullptr);
      return expr.Unify(other);
    } else if (op_name == "unification_constraints") {
      auto other = op->getAttrOfType<AccessPatternExpr>("other");
      auto domain_size = op->getAttrOfType<mlir::IntegerAttr>("domain_size");
      assert(other != nullptr);
      assert(domain_size != nullptr);
      llvm::SmallVector<AccessPatternExpr, 4> constraints(
          domain_size.getInt(), AccessPatternNoneExpr::get(&getContext()));
      if (mlir::failed(expr.UnificationConstraints(other, constraints))) {
        return nullptr;
      }
      return GetArrayAttr(constraints, builder);
    } else if (op_name == "find_in_inverse") {
      auto inverse = op->getAttrOfType<AccessPatternAttr>("inverse");
      assert(inverse != nullptr);
      return expr.FindInInverse(inverse.Dimensions());
    } else if (op_name == "min_domain_size") {
      return builder.getIndexAttr(expr.MinDomainSize());
    }
    llvm_unreachable("unknown test name");
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::Operation *op) {
      if (op->getName().getDialect() != "test") return;
      llvm::StringRef name = op->getName().stripDialect();
      auto expr = op->getAttrOfType<AccessPatternExpr>("expr");
      assert(expr != nullptr);
      mlir::Attribute result = DispatchTest(name, expr, op);
      // Replace nullptr by unit so that we can serialize it.
      if (result == nullptr) {
        result = mlir::UnitAttr::get(&getContext());
      }
      // Clear attributes except label.
      mlir::Attribute label = op->getAttr("label");
      op->setAttrs({});
      op->setAttr("result", result);
      if (label != nullptr) {
        op->setAttr("label", label);
      }
    });
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTestAccessPatternExprsPass() {
  return std::make_unique<TestAccessPatternExprsPass>();
}

// Walks a module and dispatch each operation to a DomainShapeAttr method call
// based on the operation name.
class TestDomainShapePass
    : public TestDomainShapePassBase<TestDomainShapePass> {
 public:
  mlir::Attribute DispatchTest(llvm::StringRef op_name, DomainShapeAttr shape,
                               mlir::Operation *op) {
    mlir::Builder builder(&getContext());
    if (op_name == "accessed_shape") {
      auto pattern = op->getAttrOfType<AccessPatternAttr>("pattern");
      assert(pattern != nullptr);
      return shape.AccessedShape(pattern);
    }
    llvm_unreachable("unknown test name");
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::Operation *op) {
      if (op->getName().getDialect() != "test") return;
      llvm::StringRef name = op->getName().stripDialect();
      auto shape = op->getAttrOfType<DomainShapeAttr>("shape");
      assert(shape != nullptr);
      mlir::Attribute result = DispatchTest(name, shape, op);
      // Replace nullptr by unit so that we can serialize it.
      if (result == nullptr) {
        result = mlir::UnitAttr::get(&getContext());
      }
      // Clear attributes except label.
      mlir::Attribute label = op->getAttr("label");
      op->setAttrs({});
      op->setAttr("result", result);
      if (label != nullptr) {
        op->setAttr("label", label);
      }
    });
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTestDomainShapePass() {
  return std::make_unique<TestDomainShapePass>();
}

}  // namespace sair
