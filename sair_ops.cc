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
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "loop_nest.h"
#include "sair_attributes.h"
#include "sair_dialect.h"
#include "sair_op_interfaces.h"
#include "sair_traits.h"
#include "sair_types.h"

namespace sair {

namespace {

// Parses a Sair value access if present. Returns llvm::None if no Sair value
// access is present, and a ParseResult indicating the parsing status otherwise.
// Populates "value" and "mapping" with an operand placeholder and a
// mapping attribute on success.
OptionalParseResult ParseOptionalValueAccess(
    int num_dimensions, mlir::OpAsmParser &parser,
    mlir::OpAsmParser::OperandType &value, MappingAttr &mapping) {
  OptionalParseResult has_operand = parser.parseOptionalOperand(value);
  if (!has_operand.hasValue() || mlir::failed(has_operand.getValue()))
    return has_operand;

  llvm::SMLoc loc = parser.getCurrentLocation();
  if (!(mapping = ParseOptionalMapping(parser, num_dimensions))) {
    return mlir::failure();
  }
  if (!mapping.IsFullySpecified()) {
    return parser.emitError(loc)
           << "expected mapping to a concrete element, got 'none'";
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
    llvm::SmallVectorImpl<mlir::OpAsmParser::OperandType> &operands,
    llvm::SmallVectorImpl<MappingAttr> &mappings) {
  // Try parsing a value access. If there is no operand in the parsing stream,
  // interpret it as having parsed an empty operand list and succeed.
  mlir::OpAsmParser::OperandType first_operand;
  MappingAttr first_mapping;
  OptionalParseResult has_first_operand = ParseOptionalValueAccess(
      num_dimensions, parser, first_operand, first_mapping);
  if (!has_first_operand.hasValue()) {
    return mlir::success();
  }
  if (mlir::failed(has_first_operand.getValue())) {
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

// Parses the range operator. This operation has an iteration domain and
// accesses a single Sair value with index elements. The syntax for the range
// operator is the following.
//
//   range-op ::= `sair.dyn_range` domain  ssa-value mapping
//
ParseResult ParseDynRangeOp(mlir::OpAsmParser &parser,
                            mlir::OperationState &result) {
  mlir::Builder &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operands;
  llvm::SmallVector<MappingAttr, 2> mappings;
  RangeType type;

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

  if (parser.parseColonType<RangeType>(type) ||
      parser.addTypeToList(type, result.types) ||
      ResolveDomain(parser, type.Shape(), domain, result)) {
    return failure();
  }

  llvm::ArrayRef<mlir::Attribute> mapping_attrs(mappings.begin(),
                                                mappings.size());
  result.addAttribute(
      SairDialect::kMappingAttrName,
      ArrayAttr::get(mapping_attrs, parser.getBuilder().getContext()));
  result.addAttribute(
      SairDynRangeOp::getOperandSegmentSizeAttr(),
      builder.getI64VectorAttr({static_cast<int64_t>(domain.size()),
                                static_cast<int64_t>(operands.size() - 1), 1}));

  ValueType index_value_type = builder.getType<ValueType>(
      type.Shape(), builder.getType<mlir::IndexType>());
  for (auto [operand, mapping] : llvm::zip(operands, mappings)) {
    if (mlir::failed(parser.resolveOperand(
            operand, index_value_type.AccessedType(mapping),
            result.operands))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Parses the static_range operation. This operation takes a single integer
// attribute as argument and returns a Sair range. The syntax is the following.
//
// static-range-op ::= `sair.static_range` int : !sair.range
//
ParseResult ParseStaticRangeOp(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  mlir::Attribute size;
  mlir::Type index_type = parser.getBuilder().getIndexType();
  RangeType type;
  if (failed(
          parser.parseAttribute(size, index_type, "size", result.attributes))) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword(RangeOp::kStepAttrName))) {
    mlir::Attribute step;
    if (mlir::failed(parser.parseAttribute(
            step, index_type, RangeOp::kStepAttrName, result.attributes))) {
      return failure();
    }
  }

  return failure(parser.parseColonType<RangeType>(type) ||
                 parser.addTypeToList(type, result.types));
}

// Parses the copy operation. This operation has an iteration domain and
// accesses a single Sair value. The syntax for the operation is the following.
//
// copy-op ::= `sair.copy` domain ssa-value mapping attributes
//
ParseResult ParseCopyOp(mlir::OpAsmParser &parser,
                        mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  mlir::OpAsmParser::OperandType value;
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

  result.addAttribute(SairDialect::kMappingAttrName,
                      parser.getBuilder().getArrayAttr({mapping}));

  return parser.resolveOperand(value, type.AccessedType(mapping),
                               result.operands);
}

// Parses the sair.from_scalar operation, that takes a single argument and
// returns 0D sair value that encapsulates the argument type. The syntax of the
// operation is the following.
//
// from-scalar-op ::= `sair.from_memref` ssa-value attribute-dict
//   `:` sair-value-type
//
ParseResult ParseFromScalarOp(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  mlir::OpAsmParser::OperandType operand;
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
ParseResult ParseFromMemRefOp(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  mlir::OpAsmParser::OperandType memref;
  MappingAttr mapping;
  MemRefType memref_type;
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

  result.addAttribute(
      SairFromMemRefOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getI64VectorAttr(
          {static_cast<int64_t>(parallel_domain_size),
           static_cast<int64_t>(domain.size() - parallel_domain_size),
           static_cast<int64_t>(1)}));

  mapping = mapping.ResizeUseDomain(domain.size());
  result.addAttribute(SairDialect::kMappingAttrName,
                      parser.getBuilder().getArrayAttr({mapping}));

  mlir::MLIRContext *context = parser.getBuilder().getContext();
  auto memref_value_type =
      ValueType::get(context, shape.AccessedShape(mapping), memref_type);
  auto result_type =
      ValueType::get(context, shape, memref_type.getElementType());
  return mlir::failure(
      parser.resolveOperand(memref, memref_value_type, result.operands) ||
      parser.addTypeToList(result_type, result.types));
}

// Parses the ToMemRef operation. This operation takes an iteration domain, a
// Sair value and a memref as argument and returns nothing. Its syntax is the
// following.
//
// to-memref-op ::= 'sair.from_memref' parallel-domain memref-operand
//    'memref' memref-domain value-operand attr-dict : shape, memref-type
//
ParseResult ParseToMemRefOp(mlir::OpAsmParser &parser,
                            mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  mlir::OpAsmParser::OperandType memref, value;
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
      SairDialect::kMappingAttrName,
      parser.getBuilder().getArrayAttr({memref_mapping, value_mapping}));

  result.addAttribute(
      SairFromMemRefOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getI64VectorAttr(
          {static_cast<int64_t>(parallel_domain_size),
           static_cast<int64_t>(domain.size() - parallel_domain_size),
           static_cast<int64_t>(1), static_cast<int64_t>(1)}));

  mlir::MLIRContext *context = parser.getBuilder().getContext();
  auto value_type = ValueType::get(context, shape.AccessedShape(value_mapping),
                                   memref_type.getElementType());
  auto memref_value_type =
      ValueType::get(context, shape.AccessedShape(memref_mapping), memref_type);
  return failure(
      parser.resolveOperand(memref, memref_value_type, result.operands) ||
      parser.resolveOperand(value, value_type, result.operands));
}

constexpr llvm::StringRef kOfKeyword = "of";

// Parses an operation of the form:
//
// proj ::= dialect-namespace '.' op-name domain 'of' domain operand
//    attr-dict? ':' shape ',' element-type
//
ParseResult ParseProjection(mlir::OpAsmParser &parser,
                            mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  mlir::OpAsmParser::OperandType value;
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
  result.addAttribute(SairDialect::kMappingAttrName,
                      parser.getBuilder().getArrayAttr({mapping}));
  mlir::MLIRContext *context = parser.getBuilder().getContext();
  DomainShapeAttr result_shape = shape.Prefix(num_parallel_dimensions);
  result.addTypes(ValueType::get(context, result_shape, element_type));

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int64_t.
  int num_projection_dimensions = domain.size() - num_parallel_dimensions;
  result.addAttribute(SairProjAnyOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getI64VectorAttr(
                          {static_cast<int64_t>(num_parallel_dimensions),
                           static_cast<int64_t>(num_projection_dimensions),
                           static_cast<int64_t>(1)}));

  ValueType type =
      ValueType::get(context, shape, element_type).AccessedType(mapping);
  return parser.resolveOperand(value, type, result.operands);
}

// Parses the sair.return operation, with the following syntax.
//
// return-op ::= `sair.return` operands attr-dict (`:` operands-types)?
//
ParseResult ParseReturnOp(mlir::OpAsmParser &parser,
                          mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
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
ParseResult ParseExitOp(mlir::OpAsmParser &parser,
                        mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
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
      SairDialect::kMappingAttrName,
      ArrayAttr::get(mapping_attrs, parser.getBuilder().getContext()));

  assert(mappings.size() == operands.size());
  if (element_types.size() != operands.size()) {
    return parser.emitError(type_loc)
           << "expected " << operands.size() << " types";
  }

  mlir::Builder &builder = parser.getBuilder();
  auto domain_shape = DomainShapeAttr::get(builder.getContext());
  for (auto [operand, element_type, mapping] :
       llvm::zip(operands, element_types, mappings)) {
    mlir::Type expected_type = builder.getType<ValueType>(
        domain_shape.AccessedShape(mapping), element_type);
    if (failed(
            parser.resolveOperand(operand, expected_type, result.operands))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

// Parses the sair.fby operation, with the following syntax.
//
// fby-op ::= `sair.fby` domain init `then` domain value attr-dict : type
//
static mlir::ParseResult ParseFbyOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  mlir::OpAsmParser::OperandType init, value;
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
      SairDialect::kMappingAttrName,
      mlir::ArrayAttr::get(
          {init_mapping.ResizeUseDomain(domain.size()), value_mapping},
          type.getContext()));

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int64_t.
  result.addAttribute(
      SairMapOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getI64VectorAttr(
          {static_cast<int64_t>(num_parallel_dimensions),
           static_cast<int64_t>(domain.size() - num_parallel_dimensions), 1,
           1}));

  ValueType init_type = type.AccessedType(init_mapping);
  ValueType value_type = type.AccessedType(value_mapping);
  return failure(parser.resolveOperand(init, init_type, result.operands) ||
                 parser.resolveOperand(value, value_type, result.operands));
}

// Prints a Sair value access list. Takes the list of values and respective
// mappings as arguments. Expects "values" and "mappings" to be ranges
// of equal length.
static void PrintValueAccessList(const ValueOperandRange operands,
                                 mlir::OpAsmPrinter &printer) {
  llvm::interleaveComma(operands, printer, [&](ValueOperand operand) {
    PrintValueAccess(operand, printer);
  });
}

// Prints the range operation.
void Print(SairDynRangeOp op, OpAsmPrinter &printer) {
  printer << SairDynRangeOp::getOperationName();
  PrintDomain(op.domain(), printer);
  printer << " ";
  if (op.LowerBound().is_value()) {
    PrintValueAccess(op.ValueOperands()[0], printer);
    printer << ", ";
  }
  PrintValueAccess(op.ValueOperands().back(), printer);
  if (op.step() != 1) {
    printer << " " << RangeOp::kStepAttrName << " " << op.step();
  }
  printer << " : " << op.getType();
}

// Prints the sair.static_range operation.
void Print(SairStaticRangeOp op, OpAsmPrinter &printer) {
  printer << SairStaticRangeOp::getOperationName() << " " << op.size();
  if (op.step() != 1) {
    printer << " " << RangeOp::kStepAttrName << " " << op.step();
  }
  printer << " : " << op.getType();
}

// Prints the copy operation.
void Print(SairCopyOp op, OpAsmPrinter &printer) {
  printer << SairCopyOp::getOperationName();
  PrintDomain(op.domain(), printer);
  printer << " ";
  PrintValueAccess(op.Value(), printer);
  printer.printOptionalAttrDict(op.getAttrs(), {SairDialect::kMappingAttrName});
  printer << " : " << op.getType();
}

// Prints the sair.from_scalar operation.
void Print(SairFromScalarOp op, OpAsmPrinter &printer) {
  printer << SairFromScalarOp::getOperationName() << " " << op.value();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.getType();
}

// Prints the from_memref operation.
void Print(SairFromMemRefOp op, OpAsmPrinter &printer) {
  printer << SairFromMemRefOp::getOperationName();
  PrintDomain(op.parallel_domain(), printer);
  printer << " ";
  PrintValueAccess(op.MemRef(), printer);
  printer << " memref";
  PrintDomain(op.memref_domain(), printer, op.parallel_domain().size());
  printer.printOptionalAttrDict(op.getAttrs(),
                                {SairFromMemRefOp::getOperandSegmentSizeAttr(),
                                 SairDialect::kMappingAttrName});
  printer << " : " << op.shape() << ", " << op.MemRefType();
}

// Prints the to_memref operation.
void Print(SairToMemRefOp op, OpAsmPrinter &printer) {
  printer << SairToMemRefOp::getOperationName();
  PrintDomain(op.parallel_domain(), printer);
  printer << " ";
  PrintValueAccess(op.MemRef(), printer);
  printer << " memref";
  PrintDomain(op.memref_domain(), printer, op.parallel_domain().size());
  printer << " ";
  PrintValueAccess(op.Value(), printer);
  printer.printOptionalAttrDict(
      op.getAttrs(),
      {SairFromMemRefOp::getOperandSegmentSizeAttr(),
       SairDialect::kShapeAttrName, SairDialect::kMappingAttrName});
  printer << " : " << op.shape() << ", " << op.MemRefType();
}

// Prints a projection operation.
template<typename Op>
void PrintProjectionOp(Op op, OpAsmPrinter &printer) {
  printer << Op::getOperationName();
  PrintDomain(op.parallel_domain(), printer);
  printer << " " << kOfKeyword;
  PrintDomain(op.projection_domain(), printer, op.parallel_domain().size());
  printer << " ";
  PrintValueAccess(op.Value(), printer);
  printer << " : " << op.shape() << ", "
          << op.result().getType().template cast<ValueType>().ElementType();
}

// Prints the proj_any operation.
void Print(SairProjAnyOp op, OpAsmPrinter &printer) {
  PrintProjectionOp(op, printer);
}

// Prints the proj_last operation.
void Print(SairProjLastOp op, OpAsmPrinter &printer) {
  PrintProjectionOp(op, printer);
}

// Prints the sair.return operation.
void Print(SairReturnOp op, OpAsmPrinter &printer) {
  printer << SairReturnOp::getOperationName() << " ";
  printer.printOperands(op.operands());
  printer.printOptionalAttrDict(op.getAttrs());
  if (op.operands().empty()) return;
  printer << " : ";
  llvm::interleaveComma(op.getOperands().getTypes(), printer,
                        [&](mlir::Type type) { printer.printType(type); });
}

// Prints the sair.exit operation.
void Print(SairExitOp op, OpAsmPrinter &printer) {
  printer << SairExitOp::getOperationName() << " ";
  PrintValueAccessList(op.ValueOperands(), printer);
  printer.printOptionalAttrDict(op.getAttrs(), {SairDialect::kMappingAttrName});
  if (op.inputs().empty()) return;
  printer << " : ";
  llvm::interleaveComma(
      op.getOperands().getTypes(), printer, [&](mlir::Type type) {
        printer.printType(type.cast<ValueType>().ElementType());
      });
}

// Prints the sair.fby operation.
void Print(SairFbyOp op, OpAsmPrinter &printer) {
  printer << SairFbyOp::getOperationName();
  PrintDomain(op.parallel_domain(), printer);
  printer << " ";
  PrintValueAccess(op.Init(), printer);
  printer << " " << SairFbyOp::kThenKeyword;
  PrintDomain(op.sequential_domain(), printer, op.parallel_domain().size());
  printer << " ";
  PrintValueAccess(op.Value(), printer);

  printer.printOptionalAttrDict(op.getAttrs(),
                                {
                                    SairMapOp::getOperandSegmentSizeAttr(),
                                    SairDialect::kMappingAttrName,
                                });
  printer << " : ";
  printer.printType(op.getType());
}

mlir::LogicalResult Verify(SairFromScalarOp op) {
  mlir::Type expected_type =
      op.result().getType().cast<ValueType>().ElementType();
  if (op.value().getType() != expected_type) {
    return op.emitError() << "expects different type: '" << op.value().getType()
                          << "' vs '" << expected_type << "'";
  }
  return mlir::success();
}

static mlir::LogicalResult VerifyFromToMemRef(
    mlir::Operation *op, int parallel_domain_size, DomainShapeAttr shape,
    mlir::Value memref, mlir::Value value, mlir::AffineMap access_map) {
  auto memref_type =
      memref.getType().cast<ValueType>().ElementType().cast<MemRefType>();
  auto value_type = value.getType().cast<ValueType>();
  if (memref_type.getElementType() != value_type.ElementType()) {
    return op->emitError()
           << "memref and value must have the same element type";
  }
  int memref_domain_size = shape.NumDimensions() - parallel_domain_size;
  if (access_map.getNumDims() != memref_domain_size) {
    return op->emitError() << "access_map has " << access_map.getNumDims()
                           << " dimensions, expected " << memref_domain_size;
  }
  if (memref_type.getRank() != access_map.getNumResults()) {
    return op->emitError() << "expected memref of rank "
                           << access_map.getNumResults() << ", got "
                           << memref_type.getRank();
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

mlir::LogicalResult Verify(SairExitOp op) {
  auto program_op = op.getParentOfType<SairProgramOp>();
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

  return mlir::success();
}

}  // namespace

llvm::SmallBitVector SairFromMemRefOp::DimsDependingOnOperand(
    int sair_operand) {
  llvm::SmallBitVector mask(domain().size());
  if (sair_operand == 0) {
    mask.set(parallel_domain().size(), domain().size());
  }
  return mask;
}

llvm::SmallBitVector SairToMemRefOp::DimsDependingOnOperand(int sair_operand) {
  llvm::SmallBitVector mask(domain().size());
  if (sair_operand == 0) {
    mask.set(parallel_domain().size(), domain().size());
  }
  return mask;
}

ParseResult ParseDomain(
    mlir::OpAsmParser &parser,
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> &dimensions) {
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

ParseResult ResolveDomain(mlir::OpAsmParser &parser,
                          DomainShapeAttr expected_shape,
                          llvm::ArrayRef<mlir::OpAsmParser::OperandType> domain,
                          mlir::OperationState &result) {
  std::vector<Type> domain_type;
  for (const DomainShapeDim &dim : expected_shape.Dimensions()) {
    domain_type.push_back(dim.type());
  }
  return parser.resolveOperands(domain, domain_type, parser.getNameLoc(),
                                result.operands);
}

ParseResult ParseValueAccess(int num_dimensions, mlir::OpAsmParser &parser,
                             mlir::OpAsmParser::OperandType &value,
                             MappingAttr &mapping) {
  OptionalParseResult has_value_access =
      ParseOptionalValueAccess(num_dimensions, parser, value, mapping);
  if (!has_value_access.hasValue()) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected a sair value access";
  }
  return has_value_access.getValue();
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
ParseResult ParseMapOp(mlir::OpAsmParser &parser,
                       mlir::OperationState &result) {
  // First, parse the domain and store the dimension names.
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  if (mlir::failed(ParseDomain(parser, domain))) {
    return mlir::failure();
  }

  // Parse a non-empty list of operands and store them to have their types
  // resolved when the type information is available.
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
  llvm::SmallVector<MappingAttr, 4> mappings;
  if (mlir::failed(
          ParseOperandList(domain.size(), parser, operands, mappings))) {
    return mlir::failure();
  }

  // Store the operand mappings as an attribute.
  llvm::ArrayRef<mlir::Attribute> mapping_attrs(mappings.begin(),
                                                mappings.size());
  result.addAttribute(
      SairDialect::kMappingAttrName,
      ArrayAttr::get(mapping_attrs, parser.getBuilder().getContext()));

  // Parse an optional attribute dictionary.
  if (mlir::failed(
          parser.parseOptionalAttrDictWithKeyword(result.attributes))) {
    return mlir::failure();
  }

  // Parse the remaining part of the operation and build the domain shape. Note
  // that 'llvm::None' is passed as region arguments and types to indicate to
  // MLIR that the region is expected to have a named entry block that specifies
  // the names and types of those arguments.
  mlir::Region *body = result.addRegion();
  DomainShapeAttr domain_shape;
  mlir::FunctionType function_type;
  if (parser.parseRegion(*body, llvm::None, llvm::None) ||
      parser.parseColon() ||
      parser.parseAttribute(domain_shape, SairDialect::kShapeAttrName,
                            result.attributes) ||
      parser.parseComma() || parser.parseType(function_type) ||
      ResolveDomain(parser, domain_shape, domain, result)) {
    return mlir::failure();
  }

  // Resolve operand types: they are expected to have a shape derived from the
  // domain shape by applying the mapping, and the same element type as
  // region arguments.
  mlir::Builder &builder = parser.getBuilder();
  for (int i = 0, e = function_type.getNumInputs(); i < e; ++i) {
    mlir::Type type = builder.getType<ValueType>(
        domain_shape.AccessedShape(mappings[i]), function_type.getInput(i));
    if (mlir::failed(
            parser.resolveOperand(operands[i], type, result.operands))) {
      return mlir::failure();
    }
  }

  // Construct result types: they have the domain shape and the element types
  // provided in the syntax.
  for (int i = 0, e = function_type.getNumResults(); i < e; ++i) {
    mlir::Type type =
        builder.getType<ValueType>(domain_shape, function_type.getResult(i));
    result.addTypes(type);
  }

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int64_t.
  result.addAttribute(
      SairMapOp::getOperandSegmentSizeAttr(),
      builder.getI64VectorAttr({static_cast<int64_t>(domain.size()),
                                static_cast<int64_t>(operands.size())}));

  return mlir::success();
}

// Builds a sair.map operation and setups its block with the right arguments.
// Input values must have !sair.value types.
void SairMapOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                      mlir::TypeRange result_types, mlir::ValueRange domain,
                      mlir::ArrayAttr mappings_array, mlir::ValueRange inputs,
                      DomainShapeAttr shape, ArrayAttr loop_nest,
                      ArrayAttr memory_space) {
  result.addTypes(result_types);
  result.addOperands(domain);
  result.addOperands(inputs);
  result.addAttribute(SairDialect::kMappingAttrName, mappings_array);
  result.addAttribute(SairDialect::kShapeAttrName, shape);
  if (loop_nest != nullptr) {
    result.addAttribute(ComputeOp::kLoopNestAttrName, loop_nest);
  }
  if (memory_space != nullptr) {
    result.addAttribute(ValueProducerOp::kMemorySpaceAttrName, memory_space);
  }

  auto operand_segment_sizes =
      builder.getI32VectorAttr({static_cast<int32_t>(domain.size()),
                                static_cast<int32_t>(inputs.size())});
  result.addAttribute(SairMapOp::getOperandSegmentSizeAttr(),
                      operand_segment_sizes);

  mlir::Region *region = result.addRegion();
  mlir::Block *block = new Block();
  region->push_back(block);
  for (int i = 0, e = domain.size(); i < e; ++i) {
    block->addArgument(builder.getIndexType());
  }
  for (mlir::Value input : inputs) {
    mlir::Type element_type = input.getType().cast<ValueType>().ElementType();
    block->addArgument(element_type);
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
void Print(SairMapOp op, OpAsmPrinter &printer) {
  printer << SairMapOp::getOperationName();
  PrintDomain(op.domain(), printer);
  printer << " ";

  PrintValueAccessList(op.ValueOperands(), printer);

  // Print the attributes except those that are handled specially in the syntax.
  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      {SairMapOp::getOperandSegmentSizeAttr(), SairDialect::kShapeAttrName,
       SairDialect::kMappingAttrName});

  printer.printRegion(op.body());
  printer << " : ";
  printer.printAttribute(op.shape());
  printer << ", ";

  // Print operand and result element types as a single function type.
  llvm::SmallVector<mlir::Type, 4> input_types;
  ExtractElementTypes(op.inputs(), input_types);
  llvm::SmallVector<mlir::Type, 4> output_types;
  ExtractElementTypes(op.results(), output_types);
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
  int num_domain_dimensions = op.domain().size();

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
  mlir::TypeRange return_operand_types = return_op.operands().getTypes();
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
mlir::LogicalResult Verify(SairMapOp op) {
  // Check body region argument types.
  llvm::SmallVector<mlir::Type, 4> types;
  ExtractElementTypes(op.inputs(), types);
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

// Indicates if the operand at the given index is used to initialize to
// reduction.
bool SairMapReduceOp::IsInitOperand(int position) {
  if (inits().empty()) return false;
  const int inits_begin = inits().getBeginOperandIndex();
  const int inits_end = inits_begin + inits().size();
  return inits_begin <= position && position < inits_end;
}

llvm::SmallBitVector SairMapReduceOp::DimsDependingOnOperand(int sair_operand) {
  llvm::SmallBitVector mask(domain().size());
  if (sair_operand < inits().size()) {
    mask.set(parallel_domain().size(), domain().size());
  }
  return mask;
}

// Parses a Sair MapReduce operation. The expected syntax is as follows.
//
// op ::= `sair.map_reduce` domain value-list `reduce` domain value-list
//        (`attributes` attr-dict)? region `:` shape `,` functional-type
ParseResult ParseMapReduceOp(mlir::OpAsmParser &parser,
                             mlir::OperationState &result) {
  // First, parse the parallel part of the domain and store the dimension names.
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> domain;
  if (mlir::failed(ParseDomain(parser, domain))) {
    return mlir::failure();
  }
  int num_parallel_dimensions = domain.size();

  // Parse a list of reduction initializer operands and store them to have their
  // types resolved when the type information is available.
  llvm::SmallVector<mlir::OpAsmParser::OperandType, 4> operands;
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
      SairDialect::kMappingAttrName,
      ArrayAttr::get(mapping_attrs, parser.getBuilder().getContext()));

  // Parse the remaining part of the operation and build the domain shape. Note
  // that 'llvm::None' is passed as region arguments and types to indicate to
  // MLIR that the region is expected to have a named entry block that specifies
  // the names and types of those arguments.
  mlir::Region *body = result.addRegion();
  DomainShapeAttr domain_shape;
  mlir::FunctionType function_type;
  llvm::SMLoc type_loc;
  if (parser.parseRegion(*body, llvm::None, llvm::None) ||
      parser.parseColon() ||
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
    mlir::Type type = builder.getType<ValueType>(
        domain_shape.AccessedShape(mappings[i]), operand_element_types[i]);
    if (mlir::failed(
            parser.resolveOperand(operands[i], type, result.operands))) {
      return mlir::failure();
    }
  }

  // Construct result value types. Result types use only non-reduction domain.
  auto parallel_domain_shape = DomainShapeAttr::get(
      builder.getContext(),
      domain_shape.Dimensions().take_front(num_parallel_dimensions));
  for (int i = 0, e = function_type.getNumResults(); i < e; ++i) {
    mlir::Type type = builder.getType<ValueType>(parallel_domain_shape,
                                                 function_type.getResult(i));
    result.addTypes(type);
  }

  // Store the number of operands in each variadic segments as required by MLIR,
  // it expects specifically int64_t.
  std::array<int64_t, 4> segment_sizes(
      {num_parallel_dimensions,
       static_cast<int64_t>(domain.size()) - num_parallel_dimensions,
       num_reduction_init_operands,
       static_cast<int64_t>(operands.size()) - num_reduction_init_operands});
  result.addAttribute(SairMapOp::getOperandSegmentSizeAttr(),
                      builder.getI64VectorAttr(segment_sizes));

  return mlir::success();
}

// Prints a Sair MapReduce operation using the "printer" provided.
void Print(SairMapReduceOp op, mlir::OpAsmPrinter &printer) {
  // Print the parallel part of the domain.
  printer << SairMapReduceOp::getOperationName();
  PrintDomain(op.parallel_domain(), printer);
  printer << " ";
  int num_inits = op.inits().size();
  PrintValueAccessList(op.ValueOperands().take_front(num_inits), printer);

  // Print the reduction part of the domain.
  printer << " " << SairMapReduceOp::kReduceKeyword;
  PrintDomain(op.reduction_domain(), printer, op.parallel_domain().size());
  printer << " ";
  PrintValueAccessList(op.ValueOperands().drop_front(num_inits), printer);

  // Print the attributes and the body.
  printer.printOptionalAttrDictWithKeyword(
      op.getAttrs(),
      {SairMapOp::getOperandSegmentSizeAttr(), SairDialect::kShapeAttrName,
       SairDialect::kMappingAttrName});
  printer.printRegion(op.body());

  // Print the trailing type using operand and result element types as a single
  // functional type.
  printer << " : ";
  printer.printAttribute(op.shape());
  printer << ", ";

  llvm::SmallVector<mlir::Type, 4> input_types;
  llvm::SmallVector<mlir::Type, 4> init_types;
  ExtractElementTypes(op.inputs(), input_types);
  ExtractElementTypes(op.inits(), init_types);
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
mlir::LogicalResult Verify(SairMapReduceOp op) {
  // Check body region argument types.
  llvm::SmallVector<mlir::Type, 4> types;
  ExtractElementTypes(op.inits(), types);
  ExtractElementTypes(op.inputs(), types);
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
  llvm::SmallBitVector mask(domain().size());
  mask.set(parallel_domain().size(), domain().size());
  return mask;
}

// Returns an array containing the name of all loops used to implement sair
// operations in the given block.
static mlir::ArrayAttr GatherLoopNames(mlir::Block &block,
                                       mlir::Builder &builder) {
  llvm::SmallVector<mlir::Attribute, 8> loop_names;
  block.walk([&](ComputeOp op) {
    if (!op.loop_nest().hasValue()) return;
    for (mlir::Attribute attribute : op.LoopNestLoops()) {
      LoopAttr loop = attribute.dyn_cast<LoopAttr>();
      if (loop == nullptr) continue;
      loop_names.push_back(loop.name());
    }
  });
  return builder.getArrayAttr(loop_names);
}

// Parses a SairProgramOp using "parser" and populates the "result" with data
// sufficient for MLIR to construct the operation. The expected syntax is as
// follows.
//
// op ::= `sair.program` (`attributes` attr-dict)? region
mlir::ParseResult ParseProgramOp(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  mlir::Region *body = result.addRegion();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*body, /*arguments=*/llvm::None,
                         /*argTypes=*/llvm::None) ||
      parser.parseOptionalColonTypeList(result.types)) {
    return mlir::failure();
  }
  result.addAttribute(SairProgramOp::kLoopNameTable,
                      GatherLoopNames(body->front(), parser.getBuilder()));
  return mlir::success();
}

// Prints the given SairProgramOp using "printer".
void Print(SairProgramOp op, mlir::OpAsmPrinter &printer) {
  printer << SairProgramOp::getOperationName() << " ";
  printer.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                           {SairProgramOp::kLoopNameTable});
  printer.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  if (op.results().empty()) return;
  printer << " : ";
  llvm::interleaveComma(op.getResultTypes(), printer,
                        [&](mlir::Type type) { printer.printType(type); });
}

// Verifies the well-formedness of the given SairProgramOp, in particular that
// all its non-terminator ops are Sair ops.
mlir::LogicalResult Verify(SairProgramOp op) {
  mlir::Block *body = &op.body().front();
  for (mlir::Operation &nested_operation : *body) {
    if (!isa<SairOp>(nested_operation)) {
      return op.emitOpError("expected only Sair operations in the body")
                 .attachNote(nested_operation.getLoc())
             << "found";
    }
  }

  // Check that the terminator operands are coherent with the results.
  if (body->empty() || !llvm::isa<SairExitOp>(body->back())) {
    return op.emitError() << "expected a sair.exit terminator";
  }

  return VerifyLoopNests(op);
}

void SairProgramOp::build(mlir::OpBuilder &builder,
                          mlir::OperationState &result,
                          mlir::TypeRange result_types) {
  result.addTypes(result_types);
  result.addAttribute(kLoopNameTable, builder.getArrayAttr({}));
  result.addRegion()->push_back(new Block());
}

mlir::StringAttr SairProgramOp::GenLoopName(llvm::StringRef prefix) {
  mlir::StringAttr name = mlir::StringAttr::get(prefix, getContext());
  std::vector<mlir::Attribute> name_table(loop_name_table().begin(),
                                          loop_name_table().end());
  if (llvm::count(loop_name_table(), name) > 0) {
    llvm::SmallString<128> name_buffer(prefix);
    int original_size = name_buffer.size();
    int counter = 0;
    do {
      name_buffer.resize(original_size);
      name_buffer += '_';
      name_buffer += std::to_string(counter++);
      name = mlir::StringAttr::get(name_buffer, getContext());
    } while (llvm::count(name_table, name) > 0);
  }
  name_table.push_back(name);
  loop_name_tableAttr(mlir::ArrayAttr::get(name_table, getContext()));
  return name;
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
  auto mappings_attr = mlir::ArrayAttr::get(mappings, context);
  result.addAttribute(SairDialect::kMappingAttrName, mappings_attr);
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
  llvm::SmallBitVector mask(domain().size());
  if (sair_operand == 0) {
    mask.set(parallel_domain().size(), domain().size());
  }
  return mask;
}

llvm::SmallBitVector SairFbyOp::CarryingDimensions(int sair_operand) {
  llvm::SmallBitVector mask(domain().size());
  if (sair_operand == 1) {
    mask.set(parallel_domain().size(), domain().size());
  }
  return mask;
}

}  // namespace sair

#define GET_OP_CLASSES
#include "sair_ops.cc.inc"
