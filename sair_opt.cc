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

#include <cstdlib>
#include <memory>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MlirOptMain.h"
#include "sair_dialect.h"
#include "sair_registration.h"

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> input_filename(llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-"));
  llvm::cl::opt<std::string> output_filename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  llvm::cl::opt<bool> split_input_file(
      "split-input-file",
      llvm::cl::desc("Split the input file into pieces and process each chunk "
                     "independently"),
      llvm::cl::init(false));

  llvm::cl::opt<bool> verify_diagnostics(
      "verify-diagnostics",
      llvm::cl::desc("Check that emitted diagnostics match expected-* lines on "
                     "the corresponding line"),
      llvm::cl::init(false));

  llvm::cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      llvm::cl::desc("Allow operation with no registered dialects"),
      llvm::cl::init(false));

  llvm::InitLLVM init(argc, argv);

  // Register any pass manager command line options.
  mlir::registerAllDialects();
  mlir::registerAllPasses();
  sair::RegisterSairDialect();
  sair::RegisterSairPasses();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "MLIR passes to run");
  llvm::cl::ParseCommandLineOptions(argc, argv, "SAIR optimizer driver\n");

  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> inputFile =
      mlir::openInputFile(input_filename, &errorMessage);
  if (!inputFile) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(output_filename, &errorMessage);
  if (!outputFile) {
    llvm::errs() << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  mlir::DialectRegistry registry;
  mlir::getGlobalDialectRegistry().appendTo(registry);
  return mlir::failed(
      mlir::MlirOptMain(outputFile->os(), std::move(inputFile), passPipeline,
                        registry, split_input_file, verify_diagnostics,
                        /*verifyPasses=*/false, allowUnregisteredDialects,
                        /*preloadDialectsInContext=*/true));
}
