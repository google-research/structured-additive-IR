// RUN: sair-opt -inline-trivial-sair-ops %s | FileCheck %s

// CHECK-LABEL: @one_map
func.func @one_map() -> f32 {
  // CHECK: %[[v0:.*]] = arith.constant 1.0
  %0 = arith.constant 1.0 : f32
  // CHECK-NOT: sair.program
  %1 = sair.program {
    // CHECK-NOT: from_scalar
    %2 = sair.from_scalar %0 : !sair.value<(), f32>
    // CHECK-NOT: sair.map
    %3 = sair.map %2 {
    ^bb0(%arg0: f32):
      // CHECK: %[[v3:.*]] = arith.addf %[[v0]], %[[v0]]
      %4 = arith.addf %arg0, %arg0 : f32
      // CHECK-NOT: sair.return
      sair.return %4 : f32
    } : #sair.shape<()>, (f32) -> f32
    // CHECK-NOT: sair.exit
    sair.exit %3 : f32
  } : f32
  // CHECK: func.return %[[v3]] : f32
  func.return %1 : f32
}

// CHECK-LABEL: @sequence
func.func @sequence() -> f32 {
  // CHECK: %[[v0:.*]] = arith.constant 1.0
  %0 = arith.constant 1.0 : f32
  // CHECK-NOT: sair.program
  %6 = sair.program {
    // CHECK-NOT: from_scalar
    %1 = sair.from_scalar %0 : !sair.value<(), f32>
    // CHECK-NOT: sair.map
    %2 = sair.map %1 {
    ^bb0(%arg0: f32):
      // CHECK: %[[v3:.*]] = arith.addf %[[v0]], %[[v0]]
      %3 = arith.addf %arg0, %arg0 : f32
      // CHECK-NOT: sair.return
      sair.return %3 : f32
    } : #sair.shape<()>, (f32) -> f32
    // CHECK-NOT: sair.exit
    sair.exit %2 : f32
  } : f32
  // CHECK: func.return %[[v3]] : f32
  func.return %6 : f32
}

// CHECK-LABEL: @do_nothing
func.func @do_nothing() {
  %0 = arith.constant 1.0 : f32
  sair.program {
    %1 = sair.static_range : !sair.static_range<8>
    %2 = sair.from_scalar %0 : !sair.value<(), f32>
    %3 = sair.copy %2 : !sair.value<(), f32>
    // The "map" should not be removed unless we can find the scalar value used
    // to construct its operand, which is not available in this case.
    // CHECK: sair.map
    %4 = sair.map %3 {
    ^bb0(%arg1: f32):
      sair.return %arg1 : f32
    } : #sair.shape<()>, (f32) -> f32
    sair.exit
  }
  func.return
}
