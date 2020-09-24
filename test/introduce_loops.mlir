// RUN: sair-opt %s -sair-introduce-loops | FileCheck %s

func @foo(%arg0: index, %arg1: index) { return }

// CHECK-LABEL: @map
// CHECK: %[[ARG0:.*]]: index
func @map(%arg0: index) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %[[ARG0]]
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.range %0 : !sair.range
    %2 = sair.static_range 8 : !sair.range
    // CHECK: sair.map %[[V0]] attributes {loop_nest = []} {
    sair.map[d0: %1, d1: %2] attributes {
      loop_nest = [
        {name = "A", iter = #sair.iter<d1>},
        {name = "B", iter = #sair.iter<d0>}
      ]
    } {
      // CHECK: ^{{.*}}(%[[ARG1:.*]]: index):
      ^bb0(%arg1: index, %arg2: index):
        // CHECK-DAG: %[[C0:.*]] = constant 0 : index
        // CHECK-DAG: %[[C1:.*]] = constant 1 : index
        // CHECK-DAG: %[[C8:.*]] = constant 8 : index
        // CHECK: scf.for %[[V1:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
        // CHECK-DAG:   %[[C2:.*]] = constant 0 : index
        // CHECK-DAG:   %[[C3:.*]] = constant 1 : index
        // CHECK:   scf.for %[[V2:.*]] = %[[C2]] to %[[ARG1]] step %[[C3]] {
        // CHECK:     call @foo(%[[V2]], %[[V1]]) : (index, index) -> ()
                      call @foo(%arg1, %arg2) : (index, index) -> ()
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

// CHECK-LABEL: @proj_last
func @proj_last(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V0:.*]] = sair.map %{{.*}}
    %2 = sair.map[d0:%0] %1 attributes {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } {
      ^bb0(%arg1: index, %arg2: f32):
        sair.return %arg2: f32
    } : #sair.shape<d0:range>, (f32) -> f32
    // CHECK-NOT: sair.proj_last
    %3 = sair.proj_last of[d0:%0] %2(d0) : #sair.shape<d0:range>, f32
    // CHECK: sair.exit %[[V0]]
    sair.exit %3 : f32
  } : f32
  return
}

func @bar(%arg0: f32) -> f32 { return %arg0 : f32 }

// CHECK-LABEL: @fby
func @fby(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK-NOT: sair.fby
    %2 = sair.fby %1 then[d0:%0] %3(d0) : !sair.value<d0:range, f32>
    // CHECK: %[[V1:.*]] = sair.map %[[V0]] attributes
    %3 = sair.map[d0: %0] %2(d0) attributes {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } {
    // CHECK: ^bb0(%[[V2:.*]]: f32):
      ^bb0(%arg1: index, %5: f32):
        %6 = call @bar(%5) : (f32) -> f32
        sair.return %6 : f32
    } : #sair.shape<d0:range>, (f32) -> (f32)
    %4 = sair.proj_last of[d0:%0] %3(d0) : #sair.shape<d0:range>, f32
    sair.exit %4 : f32
  } : f32
  return
}
