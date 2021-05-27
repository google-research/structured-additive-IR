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

#include "sair_types.h"

#include <tuple>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "sair_attributes.h"
#include "sair_dialect.h"

namespace sair {

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::SairShapedType. Instances of
// this class are allocated by MLIR type system in a dedicated arena. Not
// intended for direct use.
class impl::ShapedTypeStorage : public mlir::TypeStorage {
 public:
  // Key type uniquely identifying ShapedTypeStorage for MLIR type unique-ing.
  // This specific name is required by mlir::TypeUniquer.
  using KeyTy = DomainShapeAttr;

  // Creates a ShapedTypeStorage using the provided allocator, hook for MLIR
  // type system support.
  static impl::ShapedTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator,  // NOLINT
      const KeyTy &key) {
    return new (allocator.allocate<ShapedTypeStorage>()) ShapedTypeStorage(key);
  }

  // Compares the ShapedTypeStorage identification key with this object.
  bool operator==(const KeyTy &key) const { return key == shape_; }
  // Returns the hash key for MLIR type unique-ing.
  static unsigned hashKey(const KeyTy &key) { return hash_value(key); }

  // Returns the shape of the dimensions this range type depends upon.
  DomainShapeAttr shape() const { return shape_; }

 protected:
  // Constructs a storage object from the provided key. Such objects must not be
  // constructed directly but rather created by MLIR's type system within an
  // arena allocator by calling ::construct.
  //
  // Is protected so that users are forced to call the construct method, but
  // sub-classes can still call the base class constructor.
  explicit ShapedTypeStorage(const KeyTy &key) : shape_(key) {}

 private:
  // The shape of the dimensions the range depends on.
  DomainShapeAttr shape_;
};

// Forwards the request to the implementation class.
DomainShapeAttr ShapedType::Shape() const {
  return TypeSwitch<ShapedType, DomainShapeAttr>(*this)
      .Case<DynRangeType>([](DynRangeType type) { return type.Shape(); })
      .Case<ValueType>([](ValueType type) { return type.Shape(); })
      .Case<StaticRangeType>([this](StaticRangeType type) {
        return DomainShapeAttr::get(getContext(), {});
      });
}

//===----------------------------------------------------------------------===//
// DynRangeType
//===----------------------------------------------------------------------===//

// Forwards the construction to the MLIR type system with SairTypes::Range tag.
DynRangeType DynRangeType::get(DomainShapeAttr shape) {
  return Base::get(shape.getContext(), shape);
}

DomainShapeAttr DynRangeType::Shape() const { return getImpl()->shape(); }

//===----------------------------------------------------------------------===//
// StaticRangeType
//===----------------------------------------------------------------------===//

class impl::StaticRangeTypeStorage : public mlir::TypeStorage {
 public:
  // (size, step) identification key for MLIR type uniquer.
  using KeyTy = std::pair<int, int>;

  static StaticRangeTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<StaticRangeTypeStorage>())
        StaticRangeTypeStorage(key.first, key.second);
  }

  bool operator==(const KeyTy &key) const {
    return key.first == size_ && key.second == step_;
  }

  // Range size.
  int size() const { return size_; }

  // Range step.
  int step() const { return step_; }

 private:
  StaticRangeTypeStorage(int size, int step) : size_(size), step_(step) {}

  int size_;
  int step_;
};

StaticRangeType StaticRangeType::get(int size, int step,
                                     mlir::MLIRContext *context) {
  return Base::get(context, size, step);
}

StaticRangeType StaticRangeType::getChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error, int size,
    int step, mlir::MLIRContext *context) {
  return Base::getChecked(emit_error, context, size, step);
}

int StaticRangeType::size() const { return getImpl()->size(); }

int StaticRangeType::step() const { return getImpl()->step(); }

mlir::LogicalResult StaticRangeType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error, int size,
    int step) {
  if (size < 1 || step < 1) {
    return emit_error() << "expected positive step and size";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ValueType
//===----------------------------------------------------------------------===//

// Private implementation/storage class for sair::ValueType. Instances of this
// class are allocated by MLIR type system in a dedicated arena. Not intended
// for direct use.
class impl::ValueTypeStorage : public impl::ShapedTypeStorage {
 public:
  // Key type uniquely identifying ValueTypeStorage for MLIR unique-ing. This
  // specific name is required by mlir::TypeUniquer.
  using KeyTy = std::tuple<impl::ShapedTypeStorage::KeyTy, mlir::Type>;

  // Creates a ValueTypeStorage using the provided allocator, hook for MLIR type
  // system support.
  static impl::ValueTypeStorage *construct(
      mlir::TypeStorageAllocator &allocator,  // NOLINT
      const KeyTy &key) {
    return new (allocator.allocate<ValueTypeStorage>())
        ValueTypeStorage(std::get<0>(key), std::get<1>(key));
  }

  // Compares the ValueTypeStorage identification key with this object.
  bool operator==(const KeyTy &key) const {
    return impl::ShapedTypeStorage::operator==(std::get<0>(key)) &&
           std::get<1>(key) == element_type_;
  }

  // Computes the hash of a key.
  static unsigned hashKey(const KeyTy &key) {
    unsigned base = impl::ShapedTypeStorage::hashKey(std::get<0>(key));
    return llvm::hash_combine(base, std::get<1>(key));
  }

  // Returns the type of the value scalar elements.
  mlir::Type element_type() const { return element_type_; }

 private:
  // Constructs a storage object from the shape of the iteration domain and
  // element type.  Such objects must not be constructed directly but rather
  // created by MLIR's type system within an arena allocator by calling
  // ::construct.
  ValueTypeStorage(DomainShapeAttr domain, mlir::Type element_type)
      : ShapedTypeStorage(domain), element_type_(element_type) {}

  // Type of the scalar elements in the value.
  mlir::Type element_type_;
};

// Forwards the construction to the MLIR type system with Sair::Value tag.
ValueType ValueType::get(DomainShapeAttr domain, mlir::Type element_type) {
  return Base::get(domain.getContext(), std::make_tuple(domain, element_type));
}

// Forwards the construction to the MLIR type system with Sair::Value tag.
ValueType ValueType::get(mlir::Type element_type) {
  return get(DomainShapeAttr::get(element_type.getContext()), element_type);
}

// Forwards the request to the implementation class.
mlir::Type ValueType::ElementType() const { return getImpl()->element_type(); }

DomainShapeAttr ValueType::Shape() const { return getImpl()->shape(); }

ValueType ValueType::AccessedType(MappingAttr mapping) const {
  return ValueType::get(Shape().AccessedShape(mapping), ElementType());
}

//===----------------------------------------------------------------------===//
// SairDialect
//===----------------------------------------------------------------------===//

void SairDialect::registerTypes() {
  addTypes<DynRangeType, StaticRangeType, ValueType>();
}

}  // namespace sair
