// RUN: sair-opt %s -sair-introduce-loops | FileCheck %s

// CHECK-LABEL: @map
// CHECK: %[[ARG0:.*]]: index
func @map(%arg0: index) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %[[ARG0]]
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.range %0 : !sair.range
    %2 = sair.static_range 8 : !sair.range
    // CHECK: sair.map %[[V0]] {
    sair.map[d0: %1, d1: %2] {
      // CHECK: ^{{.*}}(%[[ARG1:.*]]: index):
      ^bb0(%arg1: index, %arg2: index):
        // CHECK-DAG: %[[C0:.*]] = constant 0 : index
        // CHECK-DAG: %[[C1:.*]] = constant 1 : index
        // CHECK: scf.for %[[V1:.*]] = %[[C0]] to %[[ARG1]] step %[[C1]] {
        // CHECK-DAG:   %[[C2:.*]] = constant 0 : index
        // CHECK-DAG:   %[[C3:.*]] = constant 1 : index
        // CHECK-DAG:   %[[C4:.*]] = constant 8 : index
        // CHECK:   scf.for %[[V2:.*]] = %[[C2]] to %[[C4]] step %[[C3]] {
        // CHECK:     %{{.*}} = addi %[[V1]], %[[V2]] : index
                      %3 = addi %arg1, %arg2 : index
        // CHECK:   }
        // CHECK: }
        // CHECK: sair.return
        sair.return
    // CHECK: } : #sair.shape<()>, (index) -> ()
    } : #sair.shape<d0:range x d1:range>, () -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @map_reduce
// CHECK: %[[ARG0:.*]]: index, %[[ARG1:.*]]: f32
func @map_reduce(%arg0: index, %arg1: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %[[ARG0]]
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.range %0 : !sair.range
    %2 = sair.static_range 8 : !sair.range

    // CHECK: %[[V1:.*]] = sair.from_scalar %[[ARG1]]
    %3 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // CHECK: %{{.*}} = sair.map %[[V1]], %[[V0]] {
    %4 = sair.map_reduce %3 reduce[d0: %1, d1: %2] {
      // CHECK: ^{{.*}}(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: index):
      ^bb0(%5: index, %6: index, %7: f32):
        // CHECK-DAG: %[[C0:.*]] = constant 0 : index
        // CHECK-DAG: %[[C1:.*]] = constant 1 : index
        // CHECK: %[[V2:.*]] = scf.for %[[V3:.*]] = %[[C0]] to %[[ARG3]] step %[[C1]]
        // CHECK:     iter_args(%[[V4:.*]] = %[[ARG2]]) -> (f32) {
        // CHECK-DAG:   %[[C2:.*]] = constant 0 : index
        // CHECK-DAG:   %[[C3:.*]] = constant 1 : index
        // CHECK-DAG:   %[[C4:.*]] = constant 8 : index
        // CHECK:   %[[V5:.*]] = scf.for %[[V6:.*]] = %[[C2]] to %[[C4]] step %[[C3]]
        // CHECK:       iter_args(%[[V7:.*]] = %[[V4]]) -> (f32) {
        // CHECK:     %{{.*}} = addi %[[V3]], %[[V6]] : index
                      %8 = addi %5, %6 : index
        // CHECK:     %[[V8:.*]] = addf %[[V7]], %[[V7]]
                      %9 = addf %7, %7 : f32
        // CHECK:     scf.yield %[[V8]] : f32
        // CHECK:   }
        // CHECK:   scf.yield %[[V5]] : f32
        // CHECK: }
        // CHECK: sair.return %[[V2]] : f32
        sair.return %9 : f32
    // CHECK: } : #sair.shape<()>, (f32, index) -> f32
    } : #sair.shape<d0:range x d1:range>, () -> f32
    sair.exit
  }
  return
}
