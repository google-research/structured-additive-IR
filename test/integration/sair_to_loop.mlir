// RUN: sair-opt -sair-default-lowering-attributes -convert-sair-to-loop \
// RUN:   -canonicalize -mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: @empty_program
func @empty_program() {
  // CHECK-NOT: sair.program
  sair.program {
    // CHECK-NOT: sair.exit
    sair.exit
  }
  return
}

// CHECK-LABEL: @copy_to_memref
// CHECK: %[[ARG0:.*]]: memref<8xf32>, %[[ARG1:.*]]: memref<8xf32>
func @copy_to_memref(%arg0: memref<8xf32>, %arg1: memref<8xf32>) {
  // CHECK-NOT: sair.program
  sair.program {
    // CHECK-DAG: %[[C0:.*]] = constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = constant 1 : index
    // CHECK-DAG: %[[C8:.*]] = constant 8 : index
    // CHECK: scf.for %[[V0:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
    %0 = sair.static_range : !sair.static_range<8>
    %1 = sair.from_scalar %arg0 : !sair.value<(), memref<8xf32>>
    %4 = sair.from_scalar %arg1 : !sair.value<(), memref<8xf32>>
    // CHECK:   %[[V1:.*]] = memref.load %[[ARG0]][%[[V0]]] : memref<8xf32>
    %2 = sair.from_memref %1 memref[d0:%0] {
      buffer_name = "bufferA"
    } : #sair.shape<d0:static_range<8>>, memref<8xf32>
    %3 = sair.copy[d0:%0] %2(d0) : !sair.value<d0:static_range<8>, f32>
    // CHECK:   memref.store %[[V1]], %[[ARG1]][%[[V0]]] : memref<8xf32>
    sair.to_memref %4 memref[d0:%0] %3(d0) {
      buffer_name = "bufferB"
    }  : #sair.shape<d0:static_range<8>>, memref<8xf32>
    // CHECK: }
    // CHECK-NOT: sair.exit
    sair.exit
  }
  return
}

// CHECK-LABEL: @matmul
// CHECK: %[[A:.*]]: memref<8x8xf32>, %[[B:.*]]: memref<8x8xf32>, %[[C:.*]]:  memref<8x8xf32>
func @matmul(%arg0: memref<8x8xf32>,
             %arg1: memref<8x8xf32>,
             %arg2: memref<8x8xf32>) {
  // CHECK-DAG: %[[CF0:.*]] = constant 0.0
  // CHECK-DAG: %[[C0:.*]] = constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = constant 1 : index
  // CHECK-DAG: %[[C8:.*]] = constant 8 : index
  %C0 = constant 0.0 : f32
  // CHECK-NOT: sair.program
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), memref<8x8xf32>>
    %1 = sair.from_scalar %arg1 : !sair.value<(), memref<8x8xf32>>
    %2 = sair.from_scalar %arg2 : !sair.value<(), memref<8x8xf32>>
    %3 = sair.static_range : !sair.static_range<8>

    %4 = sair.from_memref %0 memref[d0:%3, d1: %3] {buffer_name = "A"}
      : #sair.shape<d0:static_range<8> x d1:static_range<8>>, memref<8x8xf32>
    %5 = sair.from_memref %1 memref[d0:%3, d1: %3] {buffer_name = "B"}
      : #sair.shape<d0:static_range<8> x d1:static_range<8>>, memref<8x8xf32>

    %6 = sair.from_scalar %C0 : !sair.value<(), f32>
    // CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
    // CHECK:   scf.for %[[J:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
    // CHECK:     memref.store %[[CF0]], %[[C]][%[[I]], %[[J]]] : memref<8x8xf32>
    %7 = sair.copy[d0:%3, d1:%3] %6 {
      loop_nest = [
        {name = "loopI", iter = #sair.mapping_expr<d0>},
        {name = "loopJ", iter = #sair.mapping_expr<d1>}
      ],
      storage = [{
        space = "memory", name = "C",
        layout = #sair.named_mapping<[d0:"loopI", d1:"loopJ"] -> (d0, d1)>
      }]
    } : !sair.value<d0:static_range<8> x d1:static_range<8>, f32>

    %8 = sair.fby[d0:%3, d1:%3] %7(d0, d1) then[d2:%3] %9(d0, d1, d2)
      : !sair.value<d0:static_range<8> x d1:static_range<8> x d2:static_range<8>, f32>
    // CHECK:     scf.for %[[K:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
    // CHECK-DAG:   %[[V0:.*]] = memref.load %[[C]][%[[I]], %[[J]]] : memref<8x8xf32>
    // CHECK-DAG:   %[[V1:.*]] = memref.load %[[A]][%[[I]], %[[K]]] : memref<8x8xf32>
    // CHECK-DAG:   %[[V2:.*]] = memref.load %[[B]][%[[J]], %[[K]]] : memref<8x8xf32>
    // CHECK:       %[[V3:.*]] = mulf %[[V1]], %[[V2]] : f32
    // CHECK:       %[[V4:.*]] = addf %[[V0]], %[[V3]] : f32
    // CHECK:       memref.store %[[V4]], %[[C]][%[[I]], %[[J]]] : memref<8x8xf32>
    %9 = sair.map[d0:%3, d1:%3, d2:%3] %8(d0, d1, d2), %4(d0, d2), %5(d1, d2) attributes {
      loop_nest = [
        {name = "loopI", iter = #sair.mapping_expr<d0>},
        {name = "loopJ", iter = #sair.mapping_expr<d1>},
        {name = "loopK", iter = #sair.mapping_expr<d2>}
      ],
      storage = [{
        space = "memory", name = "C",
        layout = #sair.named_mapping<[d0:"loopI", d1:"loopJ"] -> (d0, d1)>
      }]
    } {
      ^bb0(%i: index, %j: index, %k : index, %c0: f32, %a: f32, %b: f32):
        %c1 = mulf %a, %b : f32
        %c2 = addf %c0, %c1 : f32
        sair.return %c2 : f32
    } : #sair.shape<d0:static_range<8> x d1:static_range<8> x d2:static_range<8>>,
        (f32, f32, f32) -> (f32)
    // CHECK:     }
    // CHECK:   }
    // CHECK: }

    %10 = sair.proj_last[d0:%3, d1:%3] of[d2:%3] %9(d0, d1, d2)
      : #sair.shape<d0:static_range<8> x d1:static_range<8> x d2:static_range<8>>, f32

    sair.to_memref %2 memref[d0:%3, d1: %3] %10(d0, d1) {buffer_name = "C"}
      : #sair.shape<d0:static_range<8> x d1:static_range<8>>, memref<8x8xf32>
    sair.exit
  }
  // CHECK: return
  return
}
