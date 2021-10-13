// RUN: sair-opt %s | sair-opt | FileCheck %s

// CHECK-LABEL: @memory_space
func @memory_space() {
  %c1 = arith.constant 1.0 : f32
  sair.program {
    // CHECK: sair.map attributes {memory_space = [0, unit]}
    sair.map attributes {memory_space = [0, unit]} {
      ^bb0:
        %0 = arith.constant 1.0 : f32
        sair.return %0, %0 : f32, f32
    } : #sair.shape<()>, () -> (f32, f32)
    sair.exit
  }
  return
}
