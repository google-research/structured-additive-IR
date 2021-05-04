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

// Walks a module and dispatch each operation to an mapping expression
// method call based on the operation name.
class TestMappingExprsPass
    : public TestMappingExprsPassBase<TestMappingExprsPass> {
 public:
  mlir::Attribute DispatchTest(llvm::StringRef op_name, MappingExpr expr,
                               mlir::Operation *op) {
    mlir::Builder builder(&getContext());
    if (op_name == "has_none_exprs") {
      return builder.getBoolAttr(expr.HasNoneExprs());
    } else if (op_name == "substitute_dims") {
      return expr.SubstituteDims(
          GetAttrVector<MappingExpr>("substitutions", op));
    } else if (op_name == "set_inverse") {
      llvm::SmallVector<MappingExpr, 4> inverses =
          GetAttrVector<MappingExpr>("inverses", op);
      auto context = op->getAttrOfType<MappingExpr>("context");
      assert(context != nullptr);
      if (mlir::failed(expr.SetInverse(context, inverses))) return nullptr;
      return GetArrayAttr(inverses, builder);
    } else if (op_name == "unify") {
      auto other = op->getAttrOfType<MappingExpr>("other");
      assert(other != nullptr);
      return Unify(expr, other);
    } else if (op_name == "unification_constraints") {
      auto other = op->getAttrOfType<MappingExpr>("other");
      auto domain_size = op->getAttrOfType<mlir::IntegerAttr>("domain_size");
      assert(other != nullptr);
      assert(domain_size != nullptr);
      llvm::SmallVector<MappingExpr, 4> constraints(
          domain_size.getInt(), MappingNoneExpr::get(&getContext()));
      if (mlir::failed(UnificationConstraints(expr, other, constraints))) {
        return nullptr;
      }
      return GetArrayAttr(constraints, builder);
    } else if (op_name == "find_in_inverse") {
      auto inverse = op->getAttrOfType<MappingAttr>("inverse");
      assert(inverse != nullptr);
      return expr.FindInInverse(inverse.Dimensions());
    } else if (op_name == "min_domain_size") {
      return builder.getIndexAttr(expr.MinDomainSize());
    } else if (op_name == "as_affine_expr") {
      int domain_size = expr.MinDomainSize();
      auto map = mlir::AffineMap::get(domain_size, 0, expr.AsAffineExpr());
      return mlir::AffineMapAttr::get(map);
    } else if (op_name == "canonicalize") {
      return expr.Canonicalize();
    }
    llvm_unreachable("unknown test name");
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() != "test") return;
      llvm::StringRef name = op->getName().stripDialect();
      auto expr = op->getAttrOfType<MappingExpr>("expr");
      assert(expr != nullptr);
      mlir::Attribute result = DispatchTest(name, expr, op);
      // Replace nullptr by unit so that we can serialize it.
      if (result == nullptr) {
        result = mlir::UnitAttr::get(&getContext());
      }
      // Clear attributes except label.
      mlir::Attribute label = op->getAttr("label");
      mlir::NamedAttrList attrs;
      attrs.set("result", result);
      if (label != nullptr) {
        attrs.set("label", label);
      }
      op->setAttrs(attrs.getDictionary(op->getContext()));
    });
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTestMappingExprsPass() {
  return std::make_unique<TestMappingExprsPass>();
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
      auto mapping = op->getAttrOfType<MappingAttr>("mapping");
      assert(mapping != nullptr);
      return shape.AccessedShape(mapping);
    }
    llvm_unreachable("unknown test name");
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() != "test") return;
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
      mlir::NamedAttrList attrs;
      attrs.set("result", result);
      if (label != nullptr) {
        attrs.set("label", label);
      }
      op->setAttrs(attrs.getDictionary(op->getContext()));
    });
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateTestDomainShapePass() {
  return std::make_unique<TestDomainShapePass>();
}

}  // namespace sair
