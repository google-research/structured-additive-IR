// RUN: sair-opt -sair-assign-default-expansion %s | FileCheck %s

// CHECK-LABEL: @map
func.func @map() {
  sair.program {
    // CHECK: sair.map
    // CHECK-SAME: expansion = "map"
    %0 = sair.map attributes {instances = [{}]} {
      ^bb0:
        %c0 = arith.constant 1.0 : f32
        sair.return %c0 : f32
    } : #sair.shape<()>, () -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @copy
func.func @copy(%arg0: f32) {
  sair.program {
    // CHECK: sair.from_scalar
    %0 = sair.from_scalar %arg0 {
      // CHECK-SAME: expansion = "copy"
      copies = [[{}]]
    } : !sair.value<(), f32>
    sair.exit
  }
  return
}
