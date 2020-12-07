// RUN: sair-opt %s -sair-lower-tomemref | FileCheck %s

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @to_memref
// CHECK: %[[ARG0:.*]]: memref<?x?xf32>
func @to_memref(%arg0: memref<?x?xf32>) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.from_scalar %[[ARG0]]
    %1 = sair.from_scalar %arg0 : !sair.value<(), memref<?x?xf32>>
    // CHECK: sair.map[d0:%[[V0]], d1:%[[V0]]] %[[V1]] {
    %2 = sair.map[d0:%0, d1: %0] {
      // CHECK: ^{{.*}}(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: memref<?x?xf32>):
      ^bb0(%arg1: index, %arg2: index):
        // CHECK: %[[V1:.*]] = constant
        %2 = constant 1.0 : f32
        // CHECK: %[[IDX0:.*]] = affine.apply #[[$map0]](%[[ARG1]], %[[ARG2]])
        // CHECK: %[[IDX1:.*]] = affine.apply #[[$map1]](%[[ARG1]], %[[ARG2]])
        // CHECK: store %[[V1]], %[[ARG3]][%[[IDX0]], %[[IDX1]]] : memref<?x?xf32>
        sair.return %2 : f32
    } : #sair.shape<d0:range x d1:range>, () -> f32
    // CHECK-NOT: sair.to_memref
    sair.to_memref %1 memref[d0:%0, d1:%0] %2(d1, d0)
      : #sair.shape<d0:range x d1:range>, memref<?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @to_memref_map_reduce
// CHECK: %[[ARG0:.*]]: memref<?xf32>,
func @to_memref_map_reduce(%arg0: memref<?xf32>, %arg1: f32) {
  sair.program {
    // CHECK: %[[RANGE:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    %2 = sair.copy[d0:%0] %1 : !sair.value<d0:range, f32>
    %3 = sair.copy[d0:%0, d1:%0] %1 : !sair.value<d0:range x d1:range, f32>

    // CHECK: %[[WRAPPED:.*]] = sair.from_scalar %[[ARG0:.*]] : !sair.value<(), memref<?xf32>>
    %6 = sair.from_scalar %arg0 : !sair.value<(), memref<?xf32>>
    // CHECK: sair.map_reduce[d0:%[[RANGE]]] %{{.*}}(d0) reduce[d1:%[[RANGE]]] %{{.*}}(d0, d1), %[[WRAPPED]] {
    %4 = sair.map_reduce[d0:%0] %2(d0) reduce[d1:%0] %3(d0, d1) {
    // CHECK: ^{{.*}}(%[[I0:.*]]: index, %[[I1:.*]]: index, %{{.*}}: f32, %{{.*}}: f32, %[[MEMREF:.*]]: memref<?xf32>):
    ^bb0(%arg2: index, %arg3: index, %arg4: f32, %arg5: f32):
      %5 = addf %arg5, %arg4 : f32
      // CHECK: %[[SUBSCRIPT:.*]] = affine.apply #[[$map2]](%[[I0]])
      // CHECK: store %{{.*}}, %[[MEMREF]][%[[SUBSCRIPT]]]
      sair.return %5 : f32
    } : #sair.shape<d0:range x d1:range>, (f32) -> f32
    // CHECK-NOT: sair.to_memref
    sair.to_memref %6 memref[d0:%0] %4(d0)
      : #sair.shape<d0:range>, memref<?xf32>
    sair.exit
  }
  return
}
