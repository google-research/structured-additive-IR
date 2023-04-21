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

#include "sair_ops.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "loop_nest.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_op_interfaces.h"
#include "sair_types.h"
#include "sequence.h"
#include "storage.h"
#include "util.h"

namespace sair {

namespace {

// Parses a Sair value access if present. Returns std::nullopt if no Sair value
// access is present, and a ParseResult indicating the parsing status otherwise.
// Populates "value" and "mapping" with an operand placeholder and a
// mapping attribute on success.
OptionalParseResult ParseOptionalValueAccess(
    int num_dimensions, mlir::OpAsmParser &parser,
    mlir::OpAsmParser::UnresolvedOperand &value, MappingAttr &mapping) {
  OptionalParseResult has_operand = parser.parseOptionalOperand(value);
  if (!has_operand.has_value() || mlir::failed(has_operand.value()))
    return has_operand;

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (!(mapping = ParseOptionalMapping(parser, num_dimensions))) {
    return mlir::failure();
  }
  if (mapping.HasNoneExprs() || mapping.HasUnknownExprs()) {
    return parser.emitError(loc)
           << "expected mapping to a concrete element, got 'none' or '?'";
  }
  return mlir::success(mapping != nullptr);
}

// Parses a potentially empty list of Sair value operands with corresponding
// mappings.
//
// value-list ::= epsilon
//              | ssa-value mapping (`,` ssa-value mapping)*
ParseResult ParseOperandList(
    int num_dimensions, mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<MappingAttr> &mappings) {
  // Try parsing a value access. If there is no operand in the parsing stream,
  // interpret it as having parsed an empty operand list and succeed.
  mlir::OpAsmParser::UnresolvedOperand first_operand;
  MappingAttr first_mapping;
  OptionalParseResult has_first_operand = ParseOptionalValueAccess(
      num_dimensions, parser, first_operand, first_mapping);
  if (!has_first_operand.has_value()) {
    return mlir::success();
  }
  if (mlir::failed(has_first_operand.value())) {
    return mlir::failure();
  }

  operands.emplace_back(first_operand);
  mappings.emplace_back(first_mapping);

  // If there was an operand, attempt parsing succeeding operands that are
  // separated by commas.
  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (mlir::failed(ParseValueAccess(num_dimensions, parser,
                                      operands.emplace_back(),
                                      mappings.emplace_back()))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Checks the type of an operand and the shape of its mapping. Registers the
// operand in result.
ParseResult ResolveOperand(const mlir::OpAsmParser::UnresolvedOperand &operand,
                           MappingAttr mapping, DomainShapeAttr shape,
                           mlir::Type element_type, mlir::OpAsmParser &parser,
                           mlir::OperationState &result) {
  AttrLocation loc(parser.getEncodedSourceLoc(operand.location),
                   "operand mapping");
  if (mlir::failed(VerifyMappingShape(loc, mapping, shape))) {
    return mlir::failure();
  }
  auto type = ValueType::get(shape, element_type).AccessedType(mapping);
  return parser.resolveOperand(operand, type, result.operands);
}
}  // namespace

// Parses the range operator. This operation has an iteration domain and
// accesses a single Sair value with index elements. The syntax for the range
// operator is the following.
//
//   range-op ::= `sair.dyn_range` domain  ssa-value mapping
//
ParseResult SairDynRangeOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  mlir::Builder &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  llvm::SmallVector<MappingAttr, 2> mappings;
  DynRangeType type;

  if (ParseDomain(parser, domain) ||
      ParseOperandList(domain.size(), parser, operands, mappings)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword(RangeOp::kStepAttrName))) {
    auto index_type = parser.getBuilder().getIndexType();
    mlir::Attribute step;
    if (mlir::failed(parser.parseAttribute(
            step, index_type, RangeOp::kStepAttrName, result.attributes))) {
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType<DynRangeType>(type) ||
      parser.addTypeToList(type, result.types) ||
      ResolveDomain(parser, type.Shape(), domain, result)) {
    return failure();
  }

  llvm::ArrayRef<mlir::Attribute> mapping_attrs(mappings.begin(),
                                                mappings.size());
  result.addAttribute(
      SairOp::kMappingAttrName,
      ArrayAttr::get(parser.getBuilder().getContext(), mapping_attrs));
  result.addAttribute(
      SairDynRangeOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(domain.size()),
                                static_cast<int32_t>(operands.size() - 1), 1}));

  for (auto [operand, mapping] : llvm::zip(operands, mappings)) {
    if (mlir::failed(ResolveOperand(operand, mapping, type.Shape(),
                                    builder.getIndexType(), parser, result))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Parses the static_range operation. This operation takes a single integer
// attribute as argument and returns a Sair range. The syntax is the following.
//
// static-range-op ::= `sair.static_range` : !sair.static_range< size, step >
//
ParseResult SairStaticRangeOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  StaticRangeType type;
  return failure(parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseColonType<StaticRangeType>(type) ||
                 parser.addTypeToList(type, result.types));
}

// Parses the placeholder dimension. This operation has an iteration domain and
// returns a range value. The syntax is the following.
//
// placeholder-op ::= `sair.placeholder` domain : range-type
//
ParseResult SairPlaceholderOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> domain;
  DimensionType type;

  return mlir::failure(ParseDomain(parser, domain) ||
                       parser.parseOptionalAttrDict(result.attributes) ||
                       parser.parseColonType<DimensionType>(type) ||
                       parser.addTypeToList(type, result.types) ||
                       ResolveDomain(parser, type.Shape(), domain, result));
}

// Parses the copy operation. This operation has an iteration domain and
// accesses a single Sair value. The syntax for the operation is the following.
//
// copy-op ::= `sair.copy` domain ssa-value mapping attributes
//
ParseResult SairCopyOp::parse(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  mlir::OpAsmParser::UnresolvedOperand value;
  MappingAttr mapping;
  ValueType type;

  if (ParseDomain(parser, domain) ||
      ParseValueAccess(domain.size(), parser, value, mapping) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType<ValueType>(type) ||
      parser.addTypeToList(type, result.types) ||
      ResolveDomain(parser, type.Shape(), domain, result)) {
    return failure();
  }

  result.addAttribute(SairOp::kMappingAttrName,
                      parser.getBuilder().getArrayAttr({mapping}));
  return ResolveOperand(value, mapping, type.Shape(), type.ElementType(),
                        parser, result);
}

// Parses the sair.from_scalar operation, that takes a single argument and
// returns 0D sair value that encapsulates the argument type. The syntax of the
// operation is the following.
//
// from-scalar-op ::= `sair.from_memref` ssa-value attribute-dict
//   `:` sair-value-type
//
ParseResult SairFromScalarOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand operand;
  ValueType result_type;
  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType<ValueType>(result_type) ||
      parser.addTypeToList(result_type, result.types)) {
    return failure();
  }

  return parser.resolveOperand(operand, result_type.ElementType(),
                               result.operands);
}

// Parses the FromMemRef operation. This operations takes an iteration domain
// and a memref as argument and returns a Sair value. This syntax is the
// following.
//
// from-memref-op ::= 'sair.from_memref' parallel-domain memref-operand
//    'memref' memref-domain attr-dict : shape, memref-type
//
ParseResult SairFromMemRefOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  mlir::OpAsmParser::UnresolvedOperand memref;
  MappingAttr mapping;
  mlir::MemRefType memref_type;
  DomainShapeAttr shape;

  if (mlir::failed(ParseDomain(parser, domain))) return mlir::failure();
  int parallel_domain_size = domain.size();
  if (ParseValueAccess(domain.size(), parser, memref, mapping) ||
      parser.parseKeyword("memref") || ParseDomain(parser, domain) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseAttribute(shape) || parser.parseComma() ||
      parser.parseType(memref_type) ||
      ResolveDomain(parser, shape, domain, result)) {
    return mlir::failure();
  }

  // It is irrelevant which Op class we use to get the attribute name because it
  // comes from a trait. However, we cannot call a trait method directly.
  result.addAttribute(
      SairFromMemRefOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(parallel_domain_size),
           static_cast<int32_t>(domain.size() - parallel_domain_size),
           static_cast<int32_t>(1)}));

  mapping = mapping.ResizeUseDomain(domain.size());
  result.addAttribute(SairOp::kMappingAttrName,
                      parser.getBuilder().getArrayAttr({mapping}));

  auto result_type = ValueType::get(shape, memref_type.getElementType());
  return mlir::failure(
      ResolveOperand(memref, mapping, shape, memref_type, parser, result) ||
      parser.addTypeToList(result_type, result.types));
}

// Parses the LoadFromMemRef operation. This operations takes an iteration
// domain and a memref as argument and returns a Sair value. The syntax is the
// following.
//
// from-memref-op ::= 'sair.load_from_memref' domain memref-operand
//    attr-dict : memref-type -> value_type
//
ParseResult SairLoadFromMemRefOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  mlir::OpAsmParser::UnresolvedOperand memref;
  MappingAttr mapping;
  mlir::MemRefType memref_type;
  ValueType value_type;

  if (ParseDomain(parser, domain) ||
      ParseValueAccess(domain.size(), parser, memref, mapping) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(memref_type) || parser.parseArrow() ||
      parser.parseType(value_type) ||
      parser.addTypeToList(value_type, result.types) ||
      ResolveDomain(parser, value_type.Shape(), domain, result)) {
    return failure();
  }

  result.addAttribute(SairOp::kMappingAttrName,
                      parser.getBuilder().getArrayAttr({mapping}));
  return ResolveOperand(memref, mapping, value_type.Shape(), memref_type,
                        parser, result);
}

// Parses the ToMemRef operation. This operation takes an iteration domain, a
// Sair value and a memref as argument and returns nothing. Its syntax is the
// following.
//
// to-memref-op ::= 'sair.from_memref' parallel-domain memref-operand
//    'memref' memref-domain value-operand attr-dict : shape, memref-type
//
ParseResult SairToMemRefOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  mlir::OpAsmParser::UnresolvedOperand memref, value;
  MappingAttr memref_mapping, value_mapping;
  DomainShapeAttr shape;
  mlir::MemRefType memref_type;

  if (mlir::failed(ParseDomain(parser, domain))) return mlir::failure();
  int parallel_domain_size = domain.size();
  if (ParseValueAccess(domain.size(), parser, memref, memref_mapping) ||
      parser.parseKeyword("memref") || ParseDomain(parser, domain) ||
      ParseValueAccess(domain.size(), parser, value, value_mapping) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseAttribute(shape, SairDialect::kShapeAttrName,
                            result.attributes) ||
      parser.parseComma() || parser.parseType(memref_type) ||
      ResolveDomain(parser, shape, domain, result)) {
    return mlir::failure();
  }

  memref_mapping = memref_mapping.ResizeUseDomain(domain.size());
  result.addAttribute(
      SairOp::kMappingAttrName,
      parser.getBuilder().getArrayAttr({memref_mapping, value_mapping}));

  // It is irrelevant which Op class we use to get the attribute name because it
  // comes from a trait. However, we cannot call a trait method directly.
  result.addAttribute(
      SairFromMemRefOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(parallel_domain_size),
           static_cast<int32_t>(domain.size() - parallel_domain_size),
           static_cast<int32_t>(1), static_cast<int32_t>(1)}));

  return failure(ResolveOperand(memref, memref_mapping, shape, memref_type,
                                parser, result) ||
                 ResolveOperand(value, value_mapping, shape,
                                memref_type.getElementType(), parser, result));
}

// Parses the StoreToMemRef operation. The syntax is the following.
//
// store-to-memref-op ::= 'sair.store_to_memref' domain memref-operand ','
//   value-operand  attr-dict : shape, memref_type
//
ParseResult SairStoreToMemRefOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  mlir::OpAsmParser::UnresolvedOperand memref, value;
  MappingAttr memref_mapping, value_mapping;
  mlir::MemRefType memref_type;
  DomainShapeAttr shape;

  if (ParseDomain(parser, domain) ||
      ParseValueAccess(domain.size(), parser, memref, memref_mapping) ||
      parser.parseComma() ||
      ParseValueAccess(domain.size(), parser, value, value_mapping) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseAttribute(shape, SairDialect::kShapeAttrName,
                            result.attributes) ||
      parser.parseComma() || parser.parseType(memref_type) ||
      ResolveDomain(parser, shape, domain, result)) {
    return failure();
  }

  result.addAttribute(
      SairOp::kMappingAttrName,
      parser.getBuilder().getArrayAttr({memref_mapping, value_mapping}));

  return failure(ResolveOperand(memref, memref_mapping, shape, memref_type,
                                parser, result) ||
                 ResolveOperand(value, value_mapping, shape,
                                memref_type.getElementType(), parser, result));
}

namespace {
constexpr llvm::StringRef kOfKeyword = "of";

// Parses an operation of the form:
//
// proj ::= dialect-namespace '.' op-name domain 'of' domain operand
//    attr-dict? ':' shape ',' element-type
//
mlir::ParseResult parseProjectionOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  mlir::OpAsmParser::UnresolvedOperand value;
  MappingAttr mapping;
  DomainShapeAttr shape;
  mlir::Type element_type;

  if (failed(ParseDomain(parser, domain))) return mlir::failure();

  int num_parallel_dimensions = domain.size();
  if (parser.parseKeyword(kOfKeyword) || ParseDomain(parser, domain) ||
      ParseValueAccess(domain.size(), parser, value, mapping) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseAttribute(shape, SairDialect::kShapeAttrName,
                            result.attributes) ||
      parser.parseComma() || parser.parseType(element_type) ||
      ResolveDomain(parser, shape, domain, result)) {
    return mlir::failure();
  }

  mapping = mapping.ResizeUseDomain(domain.size());
  result.addAttribute(SairOp::kMappingAttrName,
                      parser.getBuilder().getArrayAttr({mapping}));
  DomainShapeAttr result_shape = shape.Prefix(num_parallel_dimensions);
  result.addTypes(ValueType::get(result_shape, element_type));

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int32_t.
  int num_projection_dimensions = domain.size() - num_parallel_dimensions;
  result.addAttribute(SairProjAnyOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(num_parallel_dimensions),
                           static_cast<int32_t>(num_projection_dimensions),
                           static_cast<int32_t>(1)}));

  return ResolveOperand(value, mapping, shape, element_type, parser, result);
}
}  // namespace

mlir::ParseResult SairProjAnyOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return parseProjectionOp(parser, result);
}

mlir::ParseResult SairProjLastOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  return parseProjectionOp(parser, result);
}

// Parses the sair.return operation, with the following syntax.
//
// return-op ::= `sair.return` operands attr-dict (`:` operands-types)?
//
mlir::ParseResult SairReturnOp::parse(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SmallVector<mlir::Type, 4> operand_types;
  return failure(parser.parseOperandList(operands) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 parser.parseOptionalColonTypeList(operand_types) ||
                 parser.resolveOperands(operands, operand_types,
                                        parser.getNameLoc(), result.operands));
}

// Parses the sair.exit operation, with the follwing syntax.
//
// exit-op ::= `sair.exit` operands attr-dict? (':' element-types)?
//
mlir::ParseResult SairExitOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SmallVector<mlir::Type, 4> element_types;
  llvm::SmallVector<MappingAttr, 4> mappings;
  llvm::SMLoc type_loc;
  if (ParseOperandList(0, parser, operands, mappings) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&type_loc) ||
      parser.parseOptionalColonTypeList(element_types)) {
    return mlir::failure();
  }

  llvm::ArrayRef<mlir::Attribute> mapping_attrs(mappings.begin(),
                                                mappings.end());
  result.addAttribute(
      SairOp::kMappingAttrName,
      ArrayAttr::get(parser.getBuilder().getContext(), mapping_attrs));

  assert(mappings.size() == operands.size());
  if (element_types.size() != operands.size()) {
    return parser.emitError(type_loc)
           << "expected " << operands.size() << " types";
  }

  mlir::Builder &builder = parser.getBuilder();
  auto domain_shape = DomainShapeAttr::get(builder.getContext());
  for (auto [operand, element_type, mapping] :
       llvm::zip(operands, element_types, mappings)) {
    if (failed(ResolveOperand(operand, mapping, domain_shape, element_type,
                              parser, result))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Parses the sair.alloc operation with the following syntax.
//
// alloc-op ::= `sair.alloc` value-list attr-dict? : type
//
mlir::ParseResult SairAllocOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain, values;
  llvm::SmallVector<mlir::Attribute, 4> access_patterns;
  mlir::OpAsmParser::UnresolvedOperand value;
  MappingAttr pattern;
  if (failed(ParseDomain(parser, domain))) return mlir::failure();

  mlir::OptionalParseResult parse_result =
      ParseOptionalValueAccess(domain.size(), parser, value, pattern);
  if (parse_result.has_value() && mlir::failed(*parse_result)) {
    return failure();
  }

  if (parse_result.has_value() && mlir::succeeded(*parse_result)) {
    access_patterns.push_back(pattern);
    values.push_back(value);
    while (mlir::succeeded(parser.parseOptionalComma())) {
      if (mlir::failed(ParseValueAccess(domain.size(), parser,
                                        values.emplace_back(), pattern))) {
        return mlir::failure();
      }
      access_patterns.push_back(pattern);
    }
  }

  result.attributes.append(SairAllocOp::getOperandSegmentSizeAttr(),
                           parser.getBuilder().getDenseI32ArrayAttr(
                               {static_cast<int32_t>(domain.size()),
                                static_cast<int32_t>(values.size())}));
  result.attributes.append(SairOp::kMappingAttrName,
                           parser.getBuilder().getArrayAttr(access_patterns));

  ValueType resultType;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(resultType) ||
      parser.addTypeToList(resultType, result.types) ||
      ResolveDomain(parser, resultType.Shape(), domain, result)) {
    return mlir::failure();
  }

  auto index_type = parser.getBuilder().getIndexType();
  for (auto [value, mapping_attr] : llvm::zip(values, access_patterns)) {
    auto mapping = mapping_attr.cast<MappingAttr>();
    if (mlir::failed(ResolveOperand(value, mapping, resultType.Shape(),
                                    index_type, parser, result))) {
      return failure();
    }
  }
  return success();
}

// Parses the sair.free operation with the following syntax.
//
// free-op ::= `sair.free` domain value attr-dict : type
//
mlir::ParseResult SairFreeOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;

  mlir::OpAsmParser::UnresolvedOperand value;
  MappingAttr mapping;
  ValueType value_type;

  if (ParseDomain(parser, domain) ||
      ParseValueAccess(domain.size(), parser, value, mapping) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(value_type) ||
      ResolveDomain(parser, value_type.Shape(), domain, result) ||
      ResolveOperand(value, mapping, value_type.Shape(),
                     value_type.ElementType(), parser, result)) {
    return mlir::failure();
  }

  result.addAttribute(SairOp::kMappingAttrName,
                      parser.getBuilder().getArrayAttr(mapping));
  return mlir::success();
}

// Parses the sair.fby operation, with the following syntax.
//
// fby-op ::= `sair.fby` domain init `then` domain value attr-dict : type
//
mlir::ParseResult SairFbyOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  mlir::OpAsmParser::UnresolvedOperand init, value;
  MappingAttr init_mapping, value_mapping;
  ValueType type;

  if (failed(ParseDomain(parser, domain))) return mlir::failure();
  int num_parallel_dimensions = domain.size();
  if (ParseValueAccess(num_parallel_dimensions, parser, init, init_mapping) ||
      parser.parseKeyword(SairFbyOp::kThenKeyword) ||
      ParseDomain(parser, domain) ||
      ParseValueAccess(domain.size(), parser, value, value_mapping) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      ResolveDomain(parser, type.Shape(), domain, result) ||
      parser.addTypeToList(type, result.types)) {
    return mlir::failure();
  }

  result.addAttribute(
      SairOp::kMappingAttrName,
      mlir::ArrayAttr::get(
          type.getContext(),
          {init_mapping.ResizeUseDomain(domain.size()), value_mapping}));

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int32_t.
  result.addAttribute(
      SairMapOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(num_parallel_dimensions),
           static_cast<int32_t>(domain.size() - num_parallel_dimensions), 1,
           1}));

  return failure(ResolveOperand(init, init_mapping, type.Shape(),
                                type.ElementType(), parser, result) ||
                 ResolveOperand(value, value_mapping, type.Shape(),
                                type.ElementType(), parser, result));
}

namespace {
// Prints a Sair value access list. Takes the list of values and respective
// mappings as arguments. Expects "values" and "mappings" to be ranges
// of equal length.
void PrintValueAccessList(const ValueOperandRange operands,
                          mlir::OpAsmPrinter &printer) {
  llvm::interleaveComma(operands, printer, [&](ValueOperand operand) {
    PrintValueAccess(operand, printer);
  });
}
}  // namespace

// Prints the range operation.
void SairDynRangeOp::print(OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  printer << " ";
  if (LowerBound().is_value()) {
    PrintValueAccess(ValueOperands()[0], printer);
    printer << ", ";
  }
  PrintValueAccess(ValueOperands().back(), printer);
  if (getStep() != 1) {
    printer << " " << RangeOp::kStepAttrName << " " << getStep();
  }
  printer.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {RangeOp::kStepAttrName, SairOp::kMappingAttrName,
       SairDynRangeOp::getOperandSegmentSizeAttr()});
  printer << " : " << getType();
}

// Prints the sair.static_range operation.
void SairStaticRangeOp::print(OpAsmPrinter &printer) {
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getType();
}

void SairPlaceholderOp::print(mlir::OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getRange().getType();
}

// Prints the copy operation.
void SairCopyOp::print(OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  printer << " ";
  PrintValueAccess(Value(), printer);
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {SairOp::kMappingAttrName});
  printer << " : " << getType();
}

// Prints the sair.from_scalar operation.
void SairFromScalarOp::print(OpAsmPrinter &printer) {
  printer << " " << getValue();
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getType();
}

// Prints the from_memref operation.
void SairFromMemRefOp::print(OpAsmPrinter &printer) {
  PrintDomain(getParallelDomain(), printer);
  printer << " ";
  PrintValueAccess(MemRef(), printer);
  printer << " memref";
  PrintDomain(getMemrefDomain(), printer, getParallelDomain().size());
  // It is irrelevant which Op class we use to get the attribute name because it
  // comes from a trait. However, we cannot call a trait method directly.
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {SairFromMemRefOp::getOperandSegmentSizeAttr(),
                                 SairOp::kMappingAttrName});
  printer << " : " << getShape() << ", " << MemRefType();
}

// Prints the load_from_memref operation.
void SairLoadFromMemRefOp::print(OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  printer << " ";
  PrintValueAccess(MemRef(), printer);
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {SairOp::kMappingAttrName});
  printer << " : " << MemRefType() << " -> " << getType();
}

// Prints the to_memref operation.
void SairToMemRefOp::print(OpAsmPrinter &printer) {
  PrintDomain(getParallelDomain(), printer);
  printer << " ";
  PrintValueAccess(MemRef(), printer);
  printer << " memref";
  PrintDomain(getMemrefDomain(), printer, getParallelDomain().size());
  printer << " ";
  PrintValueAccess(Value(), printer);
  // It is irrelevant which Op class we use to get the attribute name because it
  // comes from a trait. However, we cannot call a trait method directly.
  printer.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {SairFromMemRefOp::getOperandSegmentSizeAttr(),
       SairDialect::kShapeAttrName, SairOp::kMappingAttrName});
  printer << " : " << getShape() << ", " << MemRefType();
}

// Prints the store_to_memref operation.
void SairStoreToMemRefOp::print(OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  printer << " ";
  PrintValueAccess(MemRef(), printer);
  printer << ", ";
  PrintValueAccess(Value(), printer);
  printer.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {SairOp::kMappingAttrName, SairDialect::kShapeAttrName});
  printer << " : " << getShape() << ", " << MemRefType();
}

namespace {
// Prints a projection operation.
template<typename Op>
void PrintProjectionOp(Op op, OpAsmPrinter &printer) {
  PrintDomain(op.getParallelDomain(), printer);
  printer << " " << kOfKeyword;
  PrintDomain(op.getProjectionDomain(), printer, op.getParallelDomain().size());
  printer << " ";
  PrintValueAccess(op.Value(), printer);
  printer.printOptionalAttrDict(
      op->getAttrs(), {SairFromMemRefOp::getOperandSegmentSizeAttr(),
                       SairDialect::kShapeAttrName, SairOp::kMappingAttrName});
  printer << " : " << op.getShape() << ", "
          << op.getResult().getType().template cast<ValueType>().ElementType();
}
}  // namespace

// Prints the proj_any operation.
void SairProjAnyOp::print(OpAsmPrinter &printer) {
  PrintProjectionOp(*this, printer);
}

// Prints the proj_last operation.
void SairProjLastOp::print(OpAsmPrinter &printer) {
  PrintProjectionOp(*this, printer);
}

// Prints the sair.return operation.
void SairReturnOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperands(getOperands());
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  if (getOperands().empty()) return;
  printer << " : ";
  llvm::interleaveComma(getOperands().getTypes(), printer,
                        [&](mlir::Type type) { printer.printType(type); });
}

// Prints the sair.exit operation.
void SairExitOp::print(OpAsmPrinter &printer) {
  printer << " ";
  PrintValueAccessList(ValueOperands(), printer);
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {SairOp::kMappingAttrName});
  if (getInputs().empty()) return;
  printer << " : ";
  llvm::interleaveComma(
      getOperands().getTypes(), printer, [&](mlir::Type type) {
        printer.printType(type.cast<ValueType>().ElementType());
      });
}

// Prints the sair.fby operation.
void SairFbyOp::print(OpAsmPrinter &printer) {
  PrintDomain(getParallelDomain(), printer);
  printer << " ";
  PrintValueAccess(Init(), printer);
  printer << " " << SairFbyOp::kThenKeyword;
  PrintDomain(getSequentialDomain(), printer, getParallelDomain().size());
  printer << " ";
  PrintValueAccess(Value(), printer);

  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {
                                    SairMapOp::getOperandSegmentSizeAttr(),
                                    SairOp::kMappingAttrName,
                                });
  printer << " : ";
  printer.printType(getType());
}

void SairAllocOp::print(mlir::OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  if (!ValueOperands().empty()) printer << " ";
  llvm::interleaveComma(ValueOperands(), printer, [&](ValueOperand value) {
    PrintValueAccess(value, printer);
  });
  printer.printOptionalAttrDict(
      getOperation()->getAttrs(),
      {SairOp::kMappingAttrName, SairAllocOp::getOperandSegmentSizeAttr()});
  printer << " : " << getResult().getType();
}

void SairFreeOp::print(mlir::OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  printer << " ";
  PrintValueAccess(Value(), printer);
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {SairOp::kMappingAttrName});
  mlir::Type element_type = Value().GetType().ElementType();
  printer << " : " << ValueType::get(getShape(), element_type);
}

namespace {
mlir::LogicalResult VerifyLoadFromStoreToMemRef(mlir::Operation *op,
                                                mlir::MemRefType memref_type,
                                                ValueType value_type,
                                                MappingAttr layout) {
  if (memref_type.getElementType() != value_type.ElementType()) {
    return op->emitError()
           << "memref and value type must have the same element type";
  }
  if (memref_type.getRank() != layout.size()) {
    return op->emitError() << "memref and layout must have the same rank";
  }
  if (layout.HasNoneExprs()) {
    return op->emitError() << "layout must be surjective";
  }
  return mlir::success();
}

static mlir::LogicalResult VerifyFromToMemRef(mlir::Operation *op,
                                              int parallel_domain_size,
                                              DomainShapeAttr shape,
                                              mlir::Value memref,
                                              mlir::Value value) {
  auto memref_type =
      memref.getType().cast<ValueType>().ElementType().cast<MemRefType>();
  auto value_type = value.getType().cast<ValueType>();
  if (memref_type.getElementType() != value_type.ElementType()) {
    return op->emitError()
           << "memref and value must have the same element type";
  }
  int memref_domain_size = shape.NumDimensions() - parallel_domain_size;
  if (memref_type.getRank() != memref_domain_size) {
    return op->emitError() << "expected memref of rank " << memref_domain_size
                           << ", got " << memref_type.getRank();
  }
  for (const DomainShapeDim shape_dim :
       shape.Dimensions().drop_front(parallel_domain_size)) {
    int max_dependency = shape_dim.DependencyMask().find_last();
    if (max_dependency >= parallel_domain_size) {
      return op->emitError()
             << "memref domain dimensions cannot depend on each other";
    }
  }

  return mlir::success();
}

}  // namespace

mlir::LogicalResult SairFromScalarOp::verify() {
  SairFromScalarOp op = *this;
  mlir::Type expected_type =
      op.getResult().getType().cast<ValueType>().ElementType();
  if (op.getValue().getType() != expected_type) {
    return op.emitError() << "expects different type: '"
                          << op.getValue().getType() << "' vs '"
                          << expected_type << "'";
  }
  return mlir::success();
}

mlir::LogicalResult SairExitOp::verify() {
  SairExitOp op = *this;
  auto program_op = op->getParentOfType<SairProgramOp>();
  assert(program_op);

  if (op.getNumOperands() != program_op.getNumResults()) {
    return op.emitError() << "expected " << program_op.getNumResults()
                          << " operands, found " << op.getNumOperands();
  }

  for (auto p : llvm::zip(op.getOperandTypes(), program_op.getResultTypes())) {
    mlir::Type given_type = std::get<0>(p).cast<ValueType>().ElementType();
    mlir::Type expected_type = std::get<1>(p);
    if (expected_type != given_type) {
      return op.emitError()
             << "sair.exit operands must match the return type of the "
                "sair.program: expected "
             << expected_type << ", found " << given_type;
    }
  }

  if (op.getInstances().has_value() && op.getInstances().value().empty()) {
    return op.emitOpError() << "must have an instance";
  }

  return mlir::success();
}

mlir::LogicalResult SairAllocOp::verify() {
  SairAllocOp op = *this;
  if (op.getDynamicSizes().size() != op.MemType().getNumDynamicDims()) {
    return op.emitError() << "expected " << op.MemType().getNumDynamicDims()
                          << " dynamic size operands";
  }
  return mlir::success();
}

mlir::LogicalResult SairLoadFromMemRefOp::verify() {
  return VerifyLoadFromStoreToMemRef(*this, MemRefType(),
                                     getType().cast<ValueType>(), getLayout());
}

mlir::LogicalResult SairStoreToMemRefOp::verify() {
  return VerifyLoadFromStoreToMemRef(*this, MemRefType(), Value().GetType(),
                                     getLayout());
}

mlir::LogicalResult SairFromMemRefOp::verify() {
  return VerifyFromToMemRef(*this, getParallelDomain().size(), getShape(),
                            getMemref(), getResult());
}

mlir::LogicalResult SairToMemRefOp::verify() {
  return VerifyFromToMemRef(*this, getParallelDomain().size(), getShape(),
                            getMemref(), getValue());
}

template <typename OpTy>
llvm::SmallBitVector FromToMemRefLikeDimsDependingOnOperands(OpTy op,
                                                             int sair_operand) {
  llvm::SmallBitVector mask(op.getDomain().size());
  if (sair_operand == 0) {
    mask.set(op.getParallelDomain().size(), op.getDomain().size());
  }
  return mask;
}

llvm::SmallBitVector SairFromMemRefOp::DimsDependingOnOperand(
    int sair_operand) {
  return FromToMemRefLikeDimsDependingOnOperands(*this, sair_operand);
}

llvm::SmallBitVector SairToMemRefOp::DimsDependingOnOperand(int sair_operand) {
  return FromToMemRefLikeDimsDependingOnOperands(*this, sair_operand);
}

ParseResult ParseDomain(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &dimensions) {
  if (failed(parser.parseOptionalLSquare())) return success();
  do {
    std::string dim_name = "d" + std::to_string(dimensions.size());
    if (failed(parser.parseKeyword(dim_name)) || failed(parser.parseColon()) ||
        failed(parser.parseOperand(dimensions.emplace_back()))) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRSquare();
}

ParseResult ResolveDomain(
    mlir::OpAsmParser &parser, DomainShapeAttr expected_shape,
    llvm::ArrayRef<mlir::OpAsmParser::UnresolvedOperand> domain,
    mlir::OperationState &result) {
  std::vector<Type> domain_type;
  for (const DomainShapeDim &dim : expected_shape.Dimensions()) {
    domain_type.push_back(dim.type());
  }
  return parser.resolveOperands(domain, domain_type, parser.getNameLoc(),
                                result.operands);
}

ParseResult ParseValueAccess(int num_dimensions, mlir::OpAsmParser &parser,
                             mlir::OpAsmParser::UnresolvedOperand &value,
                             MappingAttr &mapping) {
  OptionalParseResult has_value_access =
      ParseOptionalValueAccess(num_dimensions, parser, value, mapping);
  if (!has_value_access.has_value()) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected a sair value access";
  }
  return has_value_access.value();
}

void PrintValueAccess(ValueOperand value, OpAsmPrinter &printer) {
  printer << value.value();
  if (value.Mapping().empty()) return;
  printer << "(";
  PrintMapping(value.Mapping(), printer.getStream());
  printer << ")";
}

void PrintDomain(mlir::Operation::operand_range dimensions,
                 OpAsmPrinter &printer, int offset) {
  if (!dimensions.empty()) {
    printer << "[";
    llvm::interleaveComma(dimensions, printer, [&](mlir::Value operand) {
      printer << "d" << offset++ << ":";
      printer.printOperand(operand);
    });
    printer << "]";
  }
}

bool IsSameElementType(mlir::Value lhs, mlir::Value rhs) {
  return lhs.getType().cast<ValueType>().ElementType() ==
         rhs.getType().cast<ValueType>().ElementType();
}

// Parses a Sair MapOp. The expected syntax is as folows.
//
// op ::= `sair.map` domain value-list `attributes` attr-dict region
//        `:` shape `,` functional-type
ParseResult SairMapOp::parse(mlir::OpAsmParser &parser,
                             mlir::OperationState &result) {
  // First, parse the domain and store the dimension names.
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  if (mlir::failed(ParseDomain(parser, domain))) {
    return mlir::failure();
  }

  // Parse a non-empty list of operands and store them to have their types
  // resolved when the type information is available.
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SmallVector<MappingAttr, 4> mappings;
  if (mlir::failed(
          ParseOperandList(domain.size(), parser, operands, mappings))) {
    return mlir::failure();
  }

  // Store the operand mappings as an attribute.
  llvm::ArrayRef<mlir::Attribute> mapping_attrs(mappings.begin(),
                                                mappings.size());
  result.addAttribute(
      SairOp::kMappingAttrName,
      ArrayAttr::get(parser.getBuilder().getContext(), mapping_attrs));

  // Parse an optional attribute dictionary.
  if (mlir::failed(
          parser.parseOptionalAttrDictWithKeyword(result.attributes))) {
    return mlir::failure();
  }

  // Parse the remaining part of the operation and build the domain shape. Note
  // that 'std::nullopt' is passed as region arguments and types to indicate to
  // MLIR that the region is expected to have a named entry block that specifies
  // the names and types of those arguments.
  mlir::Region *body = result.addRegion();
  DomainShapeAttr domain_shape;
  mlir::FunctionType function_type;
  llvm::SMLoc type_loc;
  if (parser.parseRegion(*body) || parser.parseColon() ||
      parser.parseAttribute(domain_shape, SairDialect::kShapeAttrName,
                            result.attributes) ||
      parser.parseComma() || parser.getCurrentLocation(&type_loc) ||
      parser.parseType(function_type) ||
      ResolveDomain(parser, domain_shape, domain, result)) {
    return mlir::failure();
  }

  if (operands.size() != function_type.getNumInputs()) {
    return parser.emitError(type_loc,
                            "expected as many input types as operands");
  }

  // Resolve operand types: they are expected to have a shape derived from the
  // domain shape by applying the mapping, and the same element type as
  // region arguments.
  mlir::Builder &builder = parser.getBuilder();
  for (int i = 0, e = function_type.getNumInputs(); i < e; ++i) {
    if (mlir::failed(ResolveOperand(operands[i], mappings[i], domain_shape,
                                    function_type.getInput(i), parser,
                                    result))) {
      return mlir::failure();
    }
  }

  // Construct result types: they have the domain shape and the element types
  // provided in the syntax.
  for (int i = 0, e = function_type.getNumResults(); i < e; ++i) {
    mlir::Type type = ValueType::get(domain_shape, function_type.getResult(i));

    result.addTypes(type);
  }

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int32_t.
  result.addAttribute(
      SairMapOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(domain.size()),
                                static_cast<int32_t>(operands.size())}));

  return mlir::success();
}

// Builds a sair.map operation and setups its block with the right arguments.
// Input values must have !sair.value types.
void SairMapOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::TypeRange result_types, mlir::ValueRange domain,
                      llvm::ArrayRef<ValueAccess> inputs, DomainShapeAttr shape,
                      mlir::ArrayAttr instances, mlir::ArrayAttr copies) {
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Attribute> mappings;
  for (const auto &input : inputs) {
    values.push_back(input.value);
    mappings.push_back(input.mapping);
  }
  auto mappings_attr = builder.getArrayAttr(mappings);
  return SairMapOp::build(builder, result, result_types, domain, mappings_attr,
                          values, shape, instances, copies);
}

// Builds a sair.map operation and setups its block with the right arguments.
// Input values must have !sair.value types.
void SairMapOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::TypeRange result_types, mlir::ValueRange domain,
                      mlir::ArrayAttr mappings_array, mlir::ValueRange inputs,
                      DomainShapeAttr shape, mlir::ArrayAttr instances,
                      mlir::ArrayAttr copies) {
  result.addTypes(result_types);
  result.addOperands(domain);
  result.addOperands(inputs);
  result.addAttribute(SairOp::kMappingAttrName, mappings_array);
  result.addAttribute(SairDialect::kShapeAttrName, shape);
  if (instances != nullptr) {
    result.addAttribute(SairOp::kInstancesAttrName, instances);
  }
  if (copies != nullptr) {
    result.addAttribute(ValueProducerOp::kCopiesAttrName, copies);
  }

  auto operand_segment_sizes =
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(domain.size()),
                                static_cast<int32_t>(inputs.size())});
  result.addAttribute(SairMapOp::getOperandSegmentSizeAttr(),
                      operand_segment_sizes);

  mlir::Region *region = result.addRegion();
  mlir::Block *block = new Block();
  region->push_back(block);
  for (int i = 0, e = domain.size(); i < e; ++i) {
    block->addArgument(builder.getIndexType(), result.location);
  }
  for (mlir::Value input : inputs) {
    mlir::Type element_type = input.getType().cast<ValueType>().ElementType();
    block->addArgument(element_type, result.location);
  }
}

// Extracts the element types from the Sair value types of the given 'values'
// and appends them to 'result'.
void ExtractElementTypes(mlir::ValueRange values,
                         llvm::SmallVectorImpl<mlir::Type> &result) {
  // Exit early if values is empty since values.getTypes() expect values to be
  // non-empty.
  if (values.empty()) return;
  mlir::TypeRange types = values.getTypes();
  result.reserve(types.size());
  for (mlir::Type type : types) {
    auto value_type = type.cast<ValueType>().ElementType();
    result.push_back(value_type);
  }
}

// Prints a Sair MapOp using the 'printer' provided.
void SairMapOp::print(OpAsmPrinter &printer) {
  PrintDomain(getDomain(), printer);
  printer << " ";

  PrintValueAccessList(ValueOperands(), printer);

  // Print the attributes except those that are handled specially in the syntax.
  printer.printOptionalAttrDictWithKeyword(
      getOperation()->getAttrs(),
      {SairMapOp::getOperandSegmentSizeAttr(), SairDialect::kShapeAttrName,
       SairOp::kMappingAttrName});

  printer << ' ';
  printer.printRegion(getBody());
  printer << " : ";
  printer.printAttribute(getShape());
  printer << ", ";

  // Print operand and result element types as a single function type.
  llvm::SmallVector<mlir::Type, 4> input_types;
  ExtractElementTypes(getInputs(), input_types);
  llvm::SmallVector<mlir::Type, 4> output_types;
  ExtractElementTypes(getResults(), output_types);
  printer.printFunctionalType(input_types, output_types);
}

// Verifies the well-formedness of the body of a Sair operation "op". Assumes
// the body is a single block in a unique region. Verifies that the block has
// as many leading arguments of "index" type as "op"s domain has dimensions, and
// that the trailing arguments have the "operand_types". Reports error to the
// default MLIR stream.
static mlir::LogicalResult VerifyBodyArgsTypes(SairOp op,
                                               mlir::TypeRange operand_types) {
  mlir::Operation *operation = op.getOperation();
  mlir::Block *body = &operation->getRegion(0).front();
  mlir::TypeRange body_arg_types = body->getArgumentTypes();
  int num_domain_dimensions = op.getDomain().size();

  int expected_num_body_args = num_domain_dimensions + operand_types.size();
  if (body_arg_types.size() != expected_num_body_args) {
    return op.emitOpError()
           << "expects " << expected_num_body_args << " body arguments";
  }

  if (!llvm::all_of(body_arg_types.take_front(num_domain_dimensions),
                    [](mlir::Type t) { return t.isIndex(); })) {
    return op.emitOpError() << "expects first " << num_domain_dimensions
                            << " body arguments to have type 'index'";
  }

  if (!std::equal(operand_types.begin(), operand_types.end(),
                  body_arg_types.drop_front(num_domain_dimensions).begin())) {
    return op.emitOpError() << "expects trailing body arguments to have the "
                               "same element type as operands";
  }

  return mlir::success();
}

// Verifies that the body of a Sair operation "op" is terminated by sair.return,
// and that the types of sair.return operands have types as elemental types of
// "op" results.
static mlir::LogicalResult VerifyBodyTerminator(Operation *op) {
  mlir::Block *body = &op->getRegion(0).front();

  auto return_op = llvm::dyn_cast_or_null<SairReturnOp>(body->getTerminator());
  if (!return_op) {
    return op->emitOpError() << "expects body to be terminated with '"
                             << SairReturnOp::getOperationName() << "'";
  }

  llvm::SmallVector<mlir::Type, 4> result_types;
  ExtractElementTypes(op->getResults(), result_types);
  mlir::TypeRange return_operand_types = return_op.getOperands().getTypes();
  if (!std::equal(result_types.begin(), result_types.end(),
                  return_operand_types.begin(), return_operand_types.end())) {
    return op->emitOpError(
                 "expects element types of results to match operand types of "
                 "the body terminator")
               .attachNote(return_op.getLoc())
           << "body terminator";
  }

  return mlir::success();
}

// Verifies that a Sair MapOp is well-formed. Prints error messages to the MLIR
// default stream on any failure and returns immediately.
mlir::LogicalResult SairMapOp::verify() {
  SairMapOp op = *this;
  // Check body region argument types.
  llvm::SmallVector<mlir::Type, 4> types;
  ExtractElementTypes(op.getInputs(), types);
  auto sair_op = cast<SairOp>(op.getOperation());
  if (mlir::failed(VerifyBodyArgsTypes(sair_op, types))) {
    return mlir::failure();
  }

  // Check terminator and result types.
  if (mlir::failed(VerifyBodyTerminator(op))) {
    return mlir::failure();
  }

  return mlir::success();
}

llvm::SmallBitVector SairMapReduceOp::DimsDependingOnOperand(int sair_operand) {
  llvm::SmallBitVector mask(getDomain().size());
  if (sair_operand < getInits().size()) {
    mask.set(getParallelDomain().size(), getDomain().size());
  }
  return mask;
}

// Parses a Sair MapReduce operation. The expected syntax is as follows.
//
// op ::= `sair.map_reduce` domain value-list `reduce` domain value-list
//        (`attributes` attr-dict)? region `:` shape `,` functional-type
ParseResult SairMapReduceOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  // First, parse the parallel part of the domain and store the dimension names.
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> domain;
  if (mlir::failed(ParseDomain(parser, domain))) {
    return mlir::failure();
  }
  int num_parallel_dimensions = domain.size();

  // Parse a list of reduction initializer operands and store them to have their
  // types resolved when the type information is available.
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 4> operands;
  llvm::SmallVector<MappingAttr, 4> mappings;
  if (mlir::failed(
          ParseOperandList(domain.size(), parser, operands, mappings))) {
    return mlir::failure();
  }
  int num_reduction_init_operands = operands.size();

  // Parse the reduction part of the domain. Store operands and dimenion names
  // in the same list.
  if (parser.parseKeyword(SairMapReduceOp::kReduceKeyword) ||
      mlir::failed(ParseDomain(parser, domain)) ||
      mlir::failed(
          ParseOperandList(domain.size(), parser, operands, mappings)) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  // Resize the use domain in init operands.
  for (int i = 0; i < num_reduction_init_operands; ++i) {
    mappings[i] = mappings[i].ResizeUseDomain(domain.size());
  }
  // Store the operand mappings as an attribute.
  llvm::ArrayRef<mlir::Attribute> mapping_attrs(mappings.begin(),
                                                mappings.size());
  result.addAttribute(
      SairOp::kMappingAttrName,
      ArrayAttr::get(parser.getBuilder().getContext(), mapping_attrs));

  // Parse the remaining part of the operation and build the domain shape. Note
  // that 'std::nullopt' is passed as region arguments and types to indicate to
  // MLIR that the region is expected to have a named entry block that specifies
  // the names and types of those arguments.
  mlir::Region *body = result.addRegion();
  DomainShapeAttr domain_shape;
  mlir::FunctionType function_type;
  llvm::SMLoc type_loc;
  if (parser.parseRegion(*body) || parser.parseColon() ||
      parser.parseAttribute(domain_shape, SairDialect::kShapeAttrName,
                            result.attributes) ||
      parser.parseComma() || parser.getCurrentLocation(&type_loc) ||
      parser.parseType(function_type) ||
      ResolveDomain(parser, domain_shape, domain, result)) {
    return mlir::failure();
  }

  // Check the number of arguments in the trailing functional type. Operand type
  // resolution relies on this number being correct.
  int num_input_operands = operands.size() - num_reduction_init_operands;
  if (function_type.getNumInputs() != num_input_operands) {
    return parser.emitError(type_loc)
           << "expected " << num_input_operands
           << " arguments in the trailing function type";
  }
  if (function_type.getNumResults() != num_reduction_init_operands) {
    return parser.emitError(type_loc)
           << "expected " << num_reduction_init_operands
           << " results in the trailing function type";
  }

  // The elemental types of operands correspond to the results of the trailing
  // functional type followed by its arguments. (Leading operands correspond to
  // reduction initializers that have the same elemental type as operation
  // results, so the syntax avoids repetition).
  llvm::SmallVector<mlir::Type, 4> operand_element_types =
      llvm::to_vector<4>(function_type.getResults());
  llvm::append_range(operand_element_types, function_type.getInputs());

  // Resolve operand types.
  mlir::Builder &builder = parser.getBuilder();
  for (int i = 0, e = operand_element_types.size(); i < e; ++i) {
    if (mlir::failed(ResolveOperand(operands[i], mappings[i], domain_shape,
                                    operand_element_types[i], parser,
                                    result))) {
      return mlir::failure();
    }
  }

  // Construct result value types. Result types use only non-reduction domain.
  auto parallel_domain_shape = DomainShapeAttr::get(
      builder.getContext(),
      domain_shape.Dimensions().take_front(num_parallel_dimensions));
  for (int i = 0, e = function_type.getNumResults(); i < e; ++i) {
    mlir::Type type =
        ValueType::get(parallel_domain_shape, function_type.getResult(i));
    result.addTypes(type);
  }

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int32_t.
  std::array<int32_t, 4> segment_sizes(
      {num_parallel_dimensions,
       static_cast<int32_t>(domain.size()) - num_parallel_dimensions,
       num_reduction_init_operands,
       static_cast<int32_t>(operands.size()) - num_reduction_init_operands});
  result.addAttribute(SairMapOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segment_sizes));

  return mlir::success();
}

// Prints a Sair MapReduce operation using the "printer" provided.
void SairMapReduceOp::print(mlir::OpAsmPrinter &printer) {
  // Print the parallel part of the domain.
  PrintDomain(getParallelDomain(), printer);
  printer << " ";
  int num_inits = getInits().size();
  PrintValueAccessList(ValueOperands().take_front(num_inits), printer);

  // Print the reduction part of the domain.
  printer << " " << SairMapReduceOp::kReduceKeyword;
  PrintDomain(getReductionDomain(), printer, getParallelDomain().size());
  printer << " ";
  PrintValueAccessList(ValueOperands().drop_front(num_inits), printer);

  // Print the attributes and the body.
  printer.printOptionalAttrDictWithKeyword(
      getOperation()->getAttrs(),
      {SairMapOp::getOperandSegmentSizeAttr(), SairDialect::kShapeAttrName,
       SairOp::kMappingAttrName});
  printer << " ";
  printer.printRegion(getBody());

  // Print the trailing type using operand and result element types as a single
  // functional type.
  printer << " : ";
  printer.printAttribute(getShape());
  printer << ", ";

  llvm::SmallVector<mlir::Type, 4> input_types;
  llvm::SmallVector<mlir::Type, 4> init_types;
  ExtractElementTypes(getInputs(), input_types);
  ExtractElementTypes(getInits(), init_types);
  printer.printFunctionalType(input_types, init_types);
}

// Reduction dimensions come after parallel dimensions and should not be
// referenced in the mapping.
mlir::LogicalResult VerifyReductionMapping(MappingAttr mapping,
                                           int num_parallel_dimensions) {
  llvm::SmallBitVector dependencies = mapping.DependencyMask();
  return mlir::success(dependencies.find_next(num_parallel_dimensions) == -1);
}

// Verifies that a Sair MapReduce operation is well-formed. Prints error
// messages to the MLIR default stream on any failure and returns immediately.
mlir::LogicalResult SairMapReduceOp::verify() {
  SairMapReduceOp op = *this;
  // Check body region argument types.
  llvm::SmallVector<mlir::Type, 4> types;
  ExtractElementTypes(op.getInits(), types);
  ExtractElementTypes(op.getInputs(), types);
  auto sair_op = cast<SairOp>(op.getOperation());
  if (mlir::failed(VerifyBodyArgsTypes(sair_op, types))) {
    return mlir::failure();
  }

  // Check terminator and result types.
  if (mlir::failed(VerifyBodyTerminator(op))) {
    return mlir::failure();
  }
  return mlir::success();
}

llvm::SmallBitVector SairProjLastOp::ResultsDimDependencies() {
  llvm::SmallBitVector mask(getDomain().size());
  mask.set(getParallelDomain().size(), getDomain().size());
  return mask;
}

// Parses a SairProgramOp using "parser" and populates the "result" with data
// sufficient for MLIR to construct the operation. The expected syntax is as
// follows.
//
// op ::= `sair.program` (`attributes` attr-dict)? region
mlir::ParseResult SairProgramOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::Region *body = result.addRegion();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*body) ||
      parser.parseOptionalColonTypeList(result.types)) {
    return mlir::failure();
  }
  return mlir::success();
}

// Prints the given SairProgramOp using "printer".
void SairProgramOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  if (getResults().empty()) return;
  printer << " : ";
  llvm::interleaveComma(getResultTypes(), printer,
                        [&](mlir::Type type) { printer.printType(type); });
}

// Verifies the well-formedness of the given SairProgramOp, in particular that
// all its non-terminator ops are Sair ops, and the correctness of lowering
// attributes that operate across operations: buffer, sequence and loop_nest.
mlir::LogicalResult SairProgramOp::verify() {
  SairProgramOp program = *this;
  mlir::Block *body = &program.getBody().front();
  for (mlir::Operation &nested_operation : *body) {
    if (!isa<SairOp>(nested_operation)) {
      return program.emitOpError("expected only Sair operations in the body")
                 .attachNote(nested_operation.getLoc())
             << "found";
    }
    mlir::RegisteredOperationName info = *nested_operation.getRegisteredInfo();
    // Run operation verifier here so we can safely access their fields.
    if (mlir::failed(info.verifyInvariants(&nested_operation))) {
      return mlir::failure();
    }
  }

  auto sequence_analysis_res =
      SequenceAnalysis::Create(program, /*report_errors=*/true);
  if (!sequence_analysis_res.has_value()) return mlir::failure();

  // Check that the terminator operands are coherent with the results.
  if (body->empty() || !llvm::isa<SairExitOp>(body->back())) {
    return program.emitError() << "expected a sair.exit terminator";
  }

  IterationSpaceAnalysis iteration_spaces(program);
  SequenceAnalysis sequence_analysis(program);
  auto fusion_analysis_res =
      LoopFusionAnalysis::Create(program, sequence_analysis);
  if (!fusion_analysis_res.has_value()) return mlir::failure();
  LoopFusionAnalysis fusion_analysis = std::move(fusion_analysis_res).value();

  if (mlir::failed(VerifyLoopNests(program, fusion_analysis, iteration_spaces,
                                   sequence_analysis))) {
    return mlir::failure();
  }
  if (mlir::failed(VerifyStorages(program, fusion_analysis, iteration_spaces,
                                  sequence_analysis))) {
    return mlir::failure();
  }
  return VerifyExpansionPatterns(program);
}

void SairProgramOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result,
                          mlir::TypeRange result_types) {
  result.addTypes(result_types);
  result.addRegion()->push_back(new Block());
}

mlir::WalkResult SairProgramOp::TryWalkComputeOpInstances(
    llvm::function_ref<mlir::WalkResult(ComputeOpInstance &)> walker) {
  for (mlir::Operation &operation : getBody().front()) {
    auto compute_op = dyn_cast<ComputeOp>(&operation);
    if (compute_op != nullptr) {
      auto sair_op = cast<SairOp>(&operation);
      for (int i = 0, e = sair_op.NumInstances(); i < e; ++i) {
        ComputeOpInstance instance(compute_op, i);
        if (walker(instance).wasInterrupted()) {
          return mlir::WalkResult::interrupt();
        }
      }
    }

    auto value_producer = dyn_cast<ValueProducerOp>(&operation);
    if (value_producer == nullptr) continue;
    for (int i = 0, e = operation.getNumResults(); i < e; ++i) {
      int num_copies = value_producer.GetCopies(i).size();
      for (int j = 0; j < num_copies; ++j) {
        ComputeOpInstance instance(value_producer, i, j);
        if (walker(instance).wasInterrupted()) {
          return mlir::WalkResult::interrupt();
        }
      }
    }
  }

  return mlir::WalkResult::advance();
}

void SairProgramOp::WalkComputeOpInstances(
    llvm::function_ref<void(ComputeOpInstance &)> walker) {
  TryWalkComputeOpInstances([=](ComputeOpInstance &op) {
    walker(op);
    return mlir::WalkResult::advance();
  });
}

mlir::WalkResult SairProgramOp::TryWalkOpInstances(
    llvm::function_ref<mlir::WalkResult(OpInstance &)> walker) {
  for (mlir::Operation &operation : getBody().front()) {
    auto compute_op = dyn_cast<ComputeOp>(&operation);
    if (compute_op != nullptr) {
      auto sair_op = cast<SairOp>(&operation);
      for (int i = 0, e = sair_op.NumInstances(); i < e; ++i) {
        ComputeOpInstance instance(compute_op, i);
        if (walker(instance).wasInterrupted()) {
          return mlir::WalkResult::interrupt();
        }
      }
    } else {
      OpInstance instance(cast<SairOp>(&operation));
      if (walker(instance).wasInterrupted()) {
        return mlir::WalkResult::interrupt();
      }
    }

    auto value_producer = dyn_cast<ValueProducerOp>(&operation);
    if (value_producer == nullptr) continue;
    for (int i = 0, e = operation.getNumResults(); i < e; ++i) {
      int num_copies = value_producer.GetCopies(i).size();
      for (int j = 0; j < num_copies; ++j) {
        ComputeOpInstance instance(value_producer, i, j);
        if (walker(instance).wasInterrupted()) {
          return mlir::WalkResult::interrupt();
        }
      }
    }
  }

  return mlir::WalkResult::advance();
}

void SairProgramOp::WalkOpInstances(
    llvm::function_ref<void(OpInstance &)> walker) {
  TryWalkOpInstances([=](OpInstance &op) {
    walker(op);
    return mlir::WalkResult::advance();
  });
}

// Builds a sair.exit operation with empty mappings. This is the
// implementation of an MLIR generated class.
void SairExitOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                       mlir::ValueRange operands) {
  result.addOperands(operands);
  mlir::MLIRContext *context = builder.getContext();
  MappingAttr mapping =
      MappingAttr::get(context, /*domain_size =*/0, /*mapping =*/{});
  mlir::SmallVector<mlir::Attribute, 4> mappings(operands.size(), mapping);
  auto mappings_attr = mlir::ArrayAttr::get(context, mappings);
  result.addAttribute(SairOp::kMappingAttrName, mappings_attr);
}

OperandRange ChainOperandRanges(OperandRange first, OperandRange second) {
  if (first.empty()) return second;
  if (second.empty()) return first;

  mlir::Operation *operation = first.getBase()->getOwner();
  int begin = first.getBeginOperandIndex();
  assert(second.getBase()->getOwner() == operation);
  assert(begin + first.size() == second.getBeginOperandIndex());

  int size = first.size() + second.size();
  return operation->getOperands().slice(begin, size);
}

llvm::SmallBitVector SairFbyOp::DimsDependingOnOperand(int sair_operand) {
  llvm::SmallBitVector mask(getDomain().size());
  if (sair_operand == 0) {
    mask.set(getParallelDomain().size(), getDomain().size());
  }
  return mask;
}

llvm::SmallBitVector SairFbyOp::CarryingDimensions(int sair_operand) {
  llvm::SmallBitVector mask(getDomain().size());
  if (sair_operand == 1) {
    mask.set(getParallelDomain().size(), getDomain().size());
  }
  return mask;
}

llvm::SmallVector<int, 2> SairDynRangeOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

llvm::SmallVector<int, 2> SairPlaceholderOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

llvm::SmallVector<int, 2> SairCopyOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

llvm::SmallVector<int, 2> SairFromScalarOp::SubDomains() { return {}; }

llvm::SmallVector<int, 2> SairFromMemRefOp::SubDomains() {
  return {static_cast<int>(getParallelDomain().size()),
          static_cast<int>(getMemrefDomain().size())};
}

llvm::SmallVector<int, 2> SairLoadFromMemRefOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

llvm::SmallVector<int, 2> SairToMemRefOp::SubDomains() {
  return {static_cast<int>(getParallelDomain().size()),
          static_cast<int>(getMemrefDomain().size())};
}

llvm::SmallVector<int, 2> SairStoreToMemRefOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

llvm::SmallVector<int, 2> SairMapOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

llvm::SmallVector<int, 2> SairMapReduceOp::SubDomains() {
  return {static_cast<int>(getParallelDomain().size()),
          static_cast<int>(getReductionDomain().size())};
}

llvm::SmallVector<int, 2> SairProjLastOp::SubDomains() {
  return {static_cast<int>(getParallelDomain().size()),
          static_cast<int>(getProjectionDomain().size())};
}

llvm::SmallVector<int, 2> SairProjAnyOp::SubDomains() {
  return {static_cast<int>(getParallelDomain().size()),
          static_cast<int>(getProjectionDomain().size())};
}

llvm::SmallVector<int, 2> SairFbyOp::SubDomains() {
  return {static_cast<int>(getParallelDomain().size()),
          static_cast<int>(getSequentialDomain().size())};
}

llvm::SmallVector<int, 2> SairStaticRangeOp::SubDomains() { return {}; }

llvm::SmallVector<int, 2> SairExitOp::SubDomains() { return {}; }

llvm::SmallVector<int, 2> SairAllocOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

llvm::SmallVector<int, 2> SairFreeOp::SubDomains() {
  return {static_cast<int>(getDomain().size())};
}

DomainShapeAttr SairFreeOp::getShape() {
  ValueOperand value = Value();
  return value.GetType().Shape().AccessedShape(value.Mapping().Inverse());
}

// Takes a mapping `lhs` and an array of mappings `rhs_array`. Returns a new
// array containing the composition of `lhs` with each element of `rhs_array`.
static mlir::ArrayAttr ComposeMappings(MappingAttr lhs,
                                       mlir::ArrayAttr rhs_array) {
  return MkArrayAttrMapper<MappingAttr>([&](MappingAttr rhs) {
    return lhs.Compose(rhs).Canonicalize();
  })(rhs_array);
}

// Translate an array of instances to a new domain. `mapping` is a mapping from
// the new domain to the old one. Returns null if the array is null.
static mlir::ArrayAttr ComposeInstances(MappingAttr mapping,
                                        mlir::ArrayAttr instances) {
  mlir::MLIRContext *context = mapping.getContext();
  auto map_loop = [=](LoopAttr loop) {
    MappingExpr new_iter =
        loop.iter().SubstituteDims(mapping.Dimensions()).Canonicalize();
    return LoopAttr::get(loop.name(), new_iter, loop.unroll(), context);
  };
  return MkArrayAttrMapper(MapLoopNest(MkArrayAttrMapper<LoopAttr>(map_loop)))(
      instances);
}

SairOp SairDynRangeOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  llvm_unreachable(
      "not called by NormalizeLoops because the op defines a dimension");
}

SairOp SairStaticRangeOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  llvm_unreachable(
      "not called by NormalizeLoops because the op defines a dimension");
}

SairOp SairPlaceholderOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  llvm_unreachable(
      "not called by NormalizeLoops because the op defines a dimension");
}

SairOp SairCopyOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 1);
  assert(!HasCopies());
  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  auto new_type =
      ValueType::get(new_shape, getType().cast<ValueType>().ElementType());

  auto new_instances = ComposeInstances(new_to_old_mapping, getInstancesAttr());
  auto new_op =
      builder.create<SairCopyOp>(getLoc(), new_type, new_domains[0],
                                 new_mappings, getValue(), new_instances,
                                 /*copies=*/nullptr);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairFromScalarOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  llvm_unreachable(
      "not called by NormalizeLoops because the op has a 0D domain");
}

SairOp SairFromMemRefOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  llvm_unreachable("must be erased before calling loop normalization");
}

SairOp SairLoadFromMemRefOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 1);
  assert(!HasCopies());
  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());

  MappingAttr new_layout = new_to_old_mapping.Compose(getLayout());
  auto return_type =
      ValueType::get(new_shape, getType().cast<ValueType>().ElementType());
  auto new_instances = ComposeInstances(new_to_old_mapping, getInstancesAttr());
  auto new_op = builder.create<SairLoadFromMemRefOp>(
      getLoc(), return_type, new_domains[0], new_mappings, getMemref(),
      new_layout, new_instances, /*copies=*/nullptr);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairToMemRefOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  llvm_unreachable("must be erased before calling loop normalization");
}

SairOp SairStoreToMemRefOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 1);
  assert(!HasCopies());

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  MappingAttr new_layout = new_to_old_mapping.Compose(getLayout());
  auto new_instances = ComposeInstances(new_to_old_mapping, getInstancesAttr());
  auto new_op = builder.create<SairStoreToMemRefOp>(
      getLoc(), new_domains[0], new_mappings, getMemref(), getValue(),
      new_layout, new_shape, new_instances, /*copies=*/nullptr);
  return llvm::cast<SairOp>(new_op.getOperation());
}

// Moves the body of a sair.map or sair.map_reduce operation and update indices
// to match the new domain.
static void MoveMapBody(mlir::Location loc, mlir::Block &old_body,
                        mlir::Block &new_body, MappingAttr new_to_old_mapping,
                        mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&new_body);
  int new_domain_size = new_to_old_mapping.UseDomainSize();
  for (auto [index, expr] :
       llvm::zip(old_body.getArguments(), new_to_old_mapping.Dimensions())) {
    auto map = mlir::AffineMap::get(new_domain_size, 0, expr.AsAffineExpr());
    auto new_index = builder.create<mlir::affine::AffineApplyOp>(
        loc, map, new_body.getArguments().take_front(new_domain_size));
    index.replaceAllUsesWith(new_index);
  }

  auto old_args = old_body.getArguments().drop_front(new_to_old_mapping.size());
  auto new_args = new_body.getArguments().drop_front(new_domain_size);
  for (auto [old_arg, new_arg] : llvm::zip(old_args, new_args)) {
    old_arg.replaceAllUsesWith(new_arg);
  }
  new_body.getOperations().splice(new_body.end(), old_body.getOperations());
}

SairOp SairMapOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 1);
  assert(!HasCopies());

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  llvm::SmallVector<mlir::Type, 4> new_return_types;
  new_return_types.reserve(getResults().size());
  for (mlir::Type type : getResultTypes()) {
    new_return_types.push_back(
        ValueType::get(new_shape, type.cast<ValueType>().ElementType()));
  }
  auto new_instances = ComposeInstances(new_to_old_mapping, getInstancesAttr());
  auto new_op = builder.create<SairMapOp>(
      getLoc(), new_return_types, new_domains[0], new_mappings, getInputs(),
      new_shape, new_instances, /*copies=*/nullptr);
  MoveMapBody(getLoc(), block(), new_op.block(), new_to_old_mapping, builder);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairMapReduceOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 2);
  assert(!HasCopies());

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  llvm::SmallVector<mlir::Type, 4> new_return_types;
  new_return_types.reserve(getResults().size());
  for (mlir::Type type : getResultTypes()) {
    new_return_types.push_back(
        ValueType::get(new_shape.Prefix(new_domains[0].size()),
                       type.cast<ValueType>().ElementType()));
  }
  auto new_instances = ComposeInstances(new_to_old_mapping, getInstancesAttr());
  auto new_op = builder.create<SairMapReduceOp>(
      getLoc(), new_return_types, new_domains[0], new_domains[1], new_mappings,
      getInits(), getInputs(), new_shape, new_instances, /*copies=*/nullptr);
  // Create the map body.
  llvm::SmallVector<mlir::Type> block_arg_types(new_shape.NumDimensions(),
                                                builder.getIndexType());
  block_arg_types.reserve(block_arg_types.size() +
                          new_op.ValueOperands().size());
  for (auto operand : new_op.ValueOperands()) {
    block_arg_types.push_back(operand.GetType().ElementType());
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&new_op.getBody(), {}, block_arg_types,
                      SmallVector<Location>(block_arg_types.size(), getLoc()));
  MoveMapBody(getLoc(), block(), new_op.block(), new_to_old_mapping, builder);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairProjLastOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 2);
  assert(!HasCopies());

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  auto new_return_type =
      ValueType::get(new_shape.Prefix(new_domains[0].size()),
                     getType().cast<ValueType>().ElementType());
  auto new_op = builder.create<SairProjLastOp>(
      getLoc(), new_return_type, new_domains[0], new_domains[1], new_mappings,
      getValue(), new_shape, /*instances=*/nullptr, /*copies=*/nullptr);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairProjAnyOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 2);
  assert(!HasCopies());

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  auto new_return_type =
      ValueType::get(new_shape.Prefix(new_domains[0].size()),
                     getType().cast<ValueType>().ElementType());
  auto new_op = builder.create<SairProjAnyOp>(
      getLoc(), new_return_type, new_domains[0], new_domains[1], new_mappings,
      getValue(), new_shape, /*instances=*/nullptr, /*copies=*/nullptr);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairFbyOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 2);

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  auto new_return_type =
      ValueType::get(new_shape, getType().cast<ValueType>().ElementType());
  auto new_op = builder.create<SairFbyOp>(
      getLoc(), new_return_type, new_domains[0], new_domains[1], new_mappings,
      getInit(), getValue(), /*instances=*/nullptr, /*copies=*/nullptr);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairExitOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  llvm_unreachable(
      "not called by NormalizeLoops because sair.exit has a 0D domain");
}

SairOp SairAllocOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 1);
  assert(!HasCopies());

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  auto new_return_type = ValueType::get(new_shape, MemType());
  auto new_instances = ComposeInstances(new_to_old_mapping, getInstancesAttr());
  auto new_op = builder.create<SairAllocOp>(
      getLoc(), new_return_type, new_domains[0], new_mappings,
      getDynamicSizes(), new_instances, /*copies=*/nullptr);
  return llvm::cast<SairOp>(new_op.getOperation());
}

SairOp SairFreeOp::ReCreateWithNewDomain(
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> new_domains,
    DomainShapeAttr new_shape, MappingAttr new_to_old_mapping,
    mlir::OpBuilder &builder) {
  assert(new_domains.size() == 1);

  mlir::ArrayAttr new_mappings =
      ComposeMappings(new_to_old_mapping, getMappingArray());
  auto new_instances = ComposeInstances(new_to_old_mapping, getInstancesAttr());
  auto new_op = builder.create<SairFreeOp>(
      getLoc(), new_domains[0], new_mappings, getValue(), new_instances);
  return llvm::cast<SairOp>(new_op.getOperation());
}

}  // namespace sair

#define GET_OP_CLASSES
#include "sair_ops.cc.inc"
