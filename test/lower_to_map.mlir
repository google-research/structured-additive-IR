// RUN: sair-opt %s -sair-lower-to-map | FileCheck %s
// RUN: sair-opt %s -sair-lower-to-map --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: @copy
func @copy(%arg0 : memref<?x?xf32>) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.from_memref
    %1 = sair.from_memref[d0:%0, d1:%0] %arg0
      : memref<?x?xf32> -> !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.map[d0:%[[V0]], d1:%[[V0]]] %[[V1]](d1, d0) {
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %[[ARG0:.*]]: f32):
    // CHECK: sair.return %[[ARG0]] : f32
    %2 = sair.copy[d0:%0, d1:%0] %1(d1, d0)
    // CHECK: } : #sair.shape<d0:range x d1:range>, (f32) -> f32
      : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @map_reduce
func @map_reduce(%r1: index, %r2: index, %in1: f32) {
  sair.program {
    %0 = sair.from_scalar %r1 : !sair.value<(), index>
    %1 = sair.from_scalar %r2 : !sair.value<(), index>
    // CHECK: %[[RANGE1:.*]] = sair.dyn_range
    %2 = sair.dyn_range %0 : !sair.range
    // CHECK: %[[RANGE2:.*]] = sair.dyn_range
    %3 = sair.dyn_range %1 : !sair.range

    %4 = sair.from_scalar %in1 : !sair.value<(), f32>
    %5 = sair.copy[d0:%2, d1:%3] %4 : !sair.value<d0:range x d1:range, f32>
    %6 = sair.copy[d0:%2] %4 : !sair.value<d0:range, f32>
    %7 = sair.copy[d0:%2] %4 : !sair.value<d0:range, f32>

    // CHECK: %[[FBY1:.*]] = sair.fby[d0:%[[RANGE1]]] %[[INIT1:.*]] then[d1:%[[RANGE2]]] %[[OUTPUT:.*]]#0(d0, d1)
    // CHECK: %[[FBY2:.*]] = sair.fby[d0:%[[RANGE1]]] %[[INIT2:.*]] then[d1:%[[RANGE2]]] %[[OUTPUT]]#1(d0, d1)
    // CHECK: %[[OUTPUT]]:2 = sair.map[d0:%[[RANGE1]], d1:%[[RANGE2]]] %[[FBY1]](d0, d1), %[[FBY2]](d0, d1), %{{.*}}
    %8:2 = sair.map_reduce[d0:%2] %6(d0), %7(d0) reduce[d1:%3] %5(d0, d1) {
    // CHECK: ^{{.*}}
    ^bb0(%arg0: index, %arg1: index, %arg2: f32, %arg3: f32, %arg4: f32):
      // CHECK: addf
      %9 = addf %arg2, %arg3 : f32
      // CHECK: mulf
      %10 = mulf %arg2, %arg3 : f32
      sair.return %9, %10 : f32, f32
    // CHECK: #sair.shape<d0:range x d1:range>, (f32, f32, f32) -> (f32, f32)
    } : #sair.shape<d0:range x d1:range>, (f32) -> (f32, f32)
    // Verify the result types have correct shapes.
    // GENERIC: "sair.map"
    // GENERIC:      (!sair.range, !sair.range, !sair.value<d0:range x d1:range, f32>,
    // GENERIC-SAME:  !sair.value<d0:range x d1:range, f32>, !sair.value<d0:range x d1:range, f32>) ->
    // GENERIC-SAME: (!sair.value<d0:range x d1:range, f32>, !sair.value<d0:range x d1:range, f32>)

    // CHECK: sair.proj_last[d0:%[[RANGE1]]] of[d1:%[[RANGE2]]] %[[OUTPUT]]#0(d0, d1)
    // CHECK: sair.proj_last[d0:%[[RANGE1]]] of[d1:%[[RANGE2]]] %[[OUTPUT]]#1(d0, d1)
    // GENERIC: "sair.proj_last"
    sair.exit
  }
  return
}
