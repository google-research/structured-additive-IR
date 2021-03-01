// RUN: sair-opt %s -sair-insert-copies | FileCheck %s

// CHECK-LABEL: @from_to_memref
func @from_to_memref(%arg0 : memref<?x?xf32>) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), memref<?x?xf32>>
    // CHECK: %[[V1:.*]] = sair.from_memref
    %2 = sair.from_memref %1 memref[d0:%0, d1:%0]
      : #sair.shape<d0:range x d1:range>, memref<?x?xf32>
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[V0]], d1:%[[V0]]] %[[V1]](d1, d0)
    // CHECK: : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.to_memref %{{.*}} memref[d0:%[[V0]], d1:%[[V0]]] %[[V2:.*]](d0, d1)
    sair.to_memref %1 memref[d0:%0, d1:%0] %2(d1, d0)
      : #sair.shape<d0:range x d1:range>, memref<?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @non_invertible_to_memref
// CHECK: %[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: f32
func @non_invertible_to_memref(%arg0: memref<?xf32>, %arg1: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.from_scalar %[[ARG1]]
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // CHECK: %[[V2:.*]] = sair.from_scalar %[[ARG0]]
    %2 = sair.from_scalar %arg0 : !sair.value<(), memref<?xf32>>
    // CHECK: %[[V3:.*]] = sair.copy[d0:%[[V0]]] %[[V1]]
    // CHECK:   : !sair.value<d0:range, f32>
    // CHECK: sair.to_memref %[[V2]] memref[d0:%[[V0]]] %[[V3]](d0)
    sair.to_memref %2 memref[d0:%0] %1 : #sair.shape<d0:range>, memref<?xf32>
    sair.exit
  }
  return
}
