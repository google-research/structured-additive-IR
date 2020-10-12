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
    // CHECK:   %[[V1:.*]] = affine.load %[[ARG0]][%[[V0]]] : memref<8xf32>
    %1 = sair.from_memref[d0:%0] %arg0 : memref<8xf32> -> !sair.value<d0:range, f32>
    // CHECK:   %[[V2:.*]] = affine.apply #[[$map0]](%[[V0]])
    // CHECK:   store %[[V1]], %[[ARG1]][%[[V2]]] : memref<8xf32>
    sair.to_memref[d0:%0] %1(d0), %arg1 : memref<8xf32>
    // CHECK: }
    // CHECK-NOT: sair.exit
    sair.exit
  }
  return
}
