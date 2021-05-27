// RUN: sair-opt %s -sair-introduce-loops | FileCheck %s

func @foo(%arg0: index, %arg1: index) { return }

// CHECK-LABEL: @map
// CHECK: %[[ARG0:.*]]: index
func @map(%arg0: index) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %[[ARG0]]
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.dyn_range
    %2 = sair.static_range : !sair.static_range<8>
    // CHECK: sair.map %[[V0]] attributes {loop_nest = []} {
    sair.map[d0: %1, d1: %2] attributes {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<d1>},
        {name = "B", iter = #sair.mapping_expr<d0>}
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
    } : #sair.shape<d0:dyn_range x d1:static_range<8>>, () -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @proj_last
func @proj_last(%arg0: f32) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<8>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V0:.*]] = sair.map %{{.*}}
    %2 = sair.map[d0:%0] %1 attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } {
      ^bb0(%arg1: index, %arg2: f32):
        sair.return %arg2: f32
    } : #sair.shape<d0:static_range<8>>, (f32) -> f32
    // CHECK-NOT: sair.proj_last
    %3 = sair.proj_last of[d0:%0] %2(d0) : #sair.shape<d0:static_range<8>>, f32
    // CHECK: sair.exit %[[V0]]
    sair.exit %3 : f32
  } : f32
  return
}

func @bar(%arg0: f32) -> f32 { return %arg0 : f32 }

// CHECK-LABEL: @fby
func @fby(%arg0: f32) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<8>
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK-NOT: sair.fby
    %2 = sair.fby %1 then[d0:%0] %3(d0) : !sair.value<d0:static_range<8>, f32>
    // CHECK: %[[V1:.*]] = sair.map %[[V0]] attributes
    %3 = sair.map[d0: %0] %2(d0) attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
      storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
    } {
    // CHECK: ^bb0(%[[V2:.*]]: f32):
      ^bb0(%arg1: index, %5: f32):
        %6 = call @bar(%5) : (f32) -> f32
        sair.return %6 : f32
    } : #sair.shape<d0:static_range<8>>, (f32) -> (f32)
    %4 = sair.proj_last of[d0:%0] %3(d0) : #sair.shape<d0:static_range<8>>, f32
    sair.exit %4 : f32
  } : f32
  return
}

// CHECK-LABEL: @fuse
func @fuse(%arg0: f32) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<4>
    %1 = sair.static_range : !sair.static_range<8>
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.map %[[V0]] attributes
    %3 = sair.map[d0:%0, d1:%1] attributes {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<d0>},
        {name = "B", iter = #sair.mapping_expr<d1>}
      ]
    } {
    // CHECK: ^{{.*}}(%[[ARG0:.*]]: f32):
      ^bb0(%arg1: index, %arg2: index):
        // CHECK: scf.for %[[I0:.*]] = %{{.*}} to %{{.*}}
        // CHECK: scf.for %[[I1:.*]] = %{{.*}} to %{{.*}}
        // CHECK: call @foo(%[[I0]], %[[I1]])
        call @foo(%arg1, %arg2) : (index, index) -> ()
        // CHECK: %[[V1:.*]] = constant
        %4 = constant 1.0 : f32
        sair.return %4 : f32
    } : #sair.shape<d0:static_range<4> x d1:static_range<8>>, () -> (f32)
    // CHECK-NOT: sair.map
    sair.map[d0:%1, d1:%0] %2, %3(d1, d0) attributes {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<d1>},
        {name = "B", iter = #sair.mapping_expr<d0>}
      ]
    } {
      ^bb0(%arg1:index, %arg2: index, %arg3: f32, %arg4: f32):
        // CHECK: call @foo(%[[I1]], %[[I0]])
        call @foo(%arg1, %arg2) : (index, index) -> ()
        // CHECK: call @bar(%[[ARG0]])
        call @bar(%arg3) : (f32) -> f32
        // CHECK: call @bar(%[[V1]])
        call @bar(%arg4) : (f32) -> f32
        sair.return
    } : #sair.shape<d0:static_range<8> x d1:static_range<4>>, (f32, f32) -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @dependent_dims
func @dependent_dims() {
  sair.program {
    // CHECK: sair.map
      // CHECK-DAG: %[[V0:.*]] = constant 0
      // CHECK-DAG: %[[V1:.*]] = constant 64
      // CHECK-DAG: %[[V2:.*]] = constant 8
      // CHECK: scf.for %[[V3:.*]] = %[[V0]] to %[[V1]] step %[[V2]] {
    %0 = sair.static_range : !sair.static_range<64, 8>
    %1, %2 = sair.map[d0:%0] attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
      storage = [
        {space = "register", layout = #sair.named_mapping<[] -> ()>},
        {space = "register", layout = #sair.named_mapping<[] -> ()>}
      ]
    } {
      ^bb0(%arg0: index):
        // CHECK: %[[V4:.*]] = constant 8
        %4 = constant 8 : index
        // CHECK: %[[V5:.*]] = addi %[[V3]], %[[V4]]
        %5 = addi %arg0, %4 : index
        sair.return %arg0, %5 : index, index
    } : #sair.shape<d0:static_range<64, 8>>, () -> (index, index)
        // CHECK: %[[V6:.*]] = constant 1
    %3 = sair.dyn_range[d0:%0] %1(d0), %2(d0) : !sair.dyn_range<d0:static_range<64, 8>>
        // CHECK: scf.for %[[V7:.*]] = %[[V3]] to %[[V5]] step %[[V6]] {
    sair.map[d0:%0, d1:%3] attributes {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<d0>},
        {name = "B", iter = #sair.mapping_expr<d1>}
      ]
    } {
      ^bb0(%arg0: index, %arg1: index):
          // CHECK: call @foo(%[[V3]], %[[V7]])
        call @foo(%arg0, %arg1) : (index, index) -> ()
        sair.return
    } : #sair.shape<d0:static_range<64, 8> x d1:dyn_range(d0)>, () -> ()
        // CHECK: }
      // CHECK: }
      // CHECK: sair.return
    // CHECK: } : #sair.shape<()>, () -> ()
    // CHECK: sair.exit
    sair.exit
  }
  return
}
