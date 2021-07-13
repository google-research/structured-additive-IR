// RUN: sair-opt -sair-assign-default-expansion %s | FileCheck %s

func @main() {
  sair.program {
    // CHECK: sair.map
    // CHECK-SAME: expansion = "map"
    %0 = sair.map {
      ^bb0:
        %c0 = constant 1.0 : f32
        sair.return %c0 : f32
    } : #sair.shape<()>, () -> (f32)
    sair.exit
  }
  return
}
