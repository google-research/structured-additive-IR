// RUN: sair-opt %s -sair-copy-to-map | sair-opt | FileCheck %s

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
