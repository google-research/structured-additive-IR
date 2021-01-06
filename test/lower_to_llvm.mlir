// RUN: sair-opt --sair-lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: @undef
func @undef() {
  // = llvm.mlir.undef : f32
  %0 = sair.undef : f32
  return
}
