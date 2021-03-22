// RUN: sair-opt -sair-default-lowering-attributes -convert-sair-to-loop %s | FileCheck %s

// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0) -> (d0)>

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
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), memref<8xf32>>
    %4 = sair.from_scalar %arg1 : !sair.value<(), memref<8xf32>>
    // CHECK:   %[[V1:.*]] = affine.apply #[[$map0]](%[[V0]])
    // CHECK:   %[[V2:.*]] = memref.load %[[ARG0]][%[[V1]]] : memref<8xf32>
    %2 = sair.from_memref %1 memref[d0:%0] {
      buffer_name = "bufferA"
    } : #sair.shape<d0:range>, memref<8xf32>
    %3 = sair.copy[d0:%0] %2(d0) : !sair.value<d0:range, f32>
    // CHECK:   %[[V3:.*]] = affine.apply #[[$map0]](%[[V0]])
    // CHECK:   memref.store %[[V2]], %[[ARG1]][%[[V3]]] : memref<8xf32>
    sair.to_memref %4 memref[d0:%0] %3(d0) {
      buffer_name = "bufferB"
    }  : #sair.shape<d0:range>, memref<8xf32>
    // CHECK: }
    // CHECK-NOT: sair.exit
    sair.exit
  }
  return
}
