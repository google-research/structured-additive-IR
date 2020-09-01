// RUN: sair-opt %s -sair-lower-tomemref | FileCheck %s

// CHECK-LABEL: @to_memref
// CHECK: %[[ARG0:.*]]: memref<?x?xf32>
func @to_memref(%arg0: memref<?x?xf32>) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: sair.map[d0:%[[V0]], d1:%[[V0]]] {
    %1 = sair.map[d0:%0, d1: %0] {
      // CHECK: ^{{.*}}(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index):
      ^bb0(%arg1: index, %arg2: index):
        // CHECK: %[[V1:.*]] = constant
        %2 = constant 1.0 : f32
         // CHECK: affine.store %[[V1]], %[[ARG0]][%[[ARG2]], %[[ARG1]]] : memref<?x?xf32>
         sair.return %2 : f32
    } : #sair.shape<d0:range x d1:range>, () -> f32
    // CHECK-NOT: sair.to_memref
    sair.to_memref[d0:%0, d1:%0] %1(d1, d0), %arg0
      : memref<?x?xf32>
    sair.exit
  }
  return
}
