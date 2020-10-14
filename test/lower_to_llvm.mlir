// RUN: sair-opt --sair-lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: @undef
func @undef() {
  // = llvm.mlir.undef : !llvm.float
  %0 = sair.undef : f32
  return
}
