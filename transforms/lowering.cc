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

#include "transforms/lowering.h"

#include <memory>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "sair_dialect.h"
#include "transforms/default_lowering_attributes.h"
#include "transforms/lowering_pass_classes.h"

namespace sair {

namespace {
// Converts a SairUndefOp into LLVM's counterpart.
class LowerUndef : public ConvertOpToLLVMPattern<SairUndefOp> {
 public:
  using ConvertOpToLLVMPattern<SairUndefOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult matchAndRewrite(
      SairUndefOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type converted = typeConverter->convertType(op.getResult().getType());
    if (!converted) return failure();

    auto undef = rewriter.create<mlir::LLVM::UndefOp>(op.getLoc(), converted);
    for (mlir::NamedAttribute attr : op->getAttrs()) {
      undef->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(op, undef.getResult());
    return mlir::success();
  }
};

// A pass that converts Standard ops and SairUndefOp to the LLVM dialect.
class LowerToLLVMPass : public LowerToLLVMBase<LowerToLLVMPass> {
 public:
  void runOnOperation() override {
    auto module = getOperation();

    OwningRewritePatternList patterns(&getContext());
    LLVMTypeConverter converter(&getContext());
    populateMemRefToLLVMConversionPatterns(converter, patterns);
    populateStdToLLVMConversionPatterns(converter, patterns);
    arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
    patterns.insert<LowerUndef>(converter);

    LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();
    target.addIllegalDialect<SairDialect>();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}

void CreateSairToLoopConversionPipeline(mlir::OpPassManager *pm) {
  pm->addPass(CreateLowerMapReducePass());
  pm->addPass(CreateMaterializeBuffersPass());
  // Canonicalize removes non-compute operations for values converted to
  // buffers.
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(CreateNormalizeLoopsPass());
  pm->addPass(CreateLowerProjAnyPass());
  pm->addPass(CreateLowerToMapPass());
  pm->addPass(CreateIntroduceLoopsPass());
  pm->addPass(CreateInlineTrivialOpsPass());
}

void CreateSairToLLVMConversionPipeline(mlir::OpPassManager *pm) {
  CreateSairToLoopConversionPipeline(pm);
  pm->addPass(mlir::createLowerAffinePass());
  pm->addPass(mlir::createLowerToCFGPass());
  pm->addPass(CreateLowerToLLVMPass());
}

}  // namespace sair
