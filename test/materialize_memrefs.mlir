// RUN: sair-opt %s -sair-materialize-memrefs | FileCheck %s

// CHECK-DAG: #[[$map2d1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$map2d0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[$map1d0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @from_memref
// CHECK: %[[ARG0:.*]]: index, %[[ARG1:.*]]: memref<?x8xf32>
func @from_memref(%arg0: index, %arg1: memref<?x8xf32>) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.range
    %2 = sair.static_range 8 : !sair.range

    // CHECK: %[[V0:.*]] = sair.from_scalar %[[ARG1]]
    %5 = sair.from_scalar %arg1 : !sair.value<(), memref<?x8xf32>>
    // CHECK: : !sair.value<(), memref<?x8xf32>
    %3 = sair.from_memref %5 memref[d0:%1, d1:%2]
      : #sair.shape<d0:range x d1:range>, memref<?x8xf32>
    // CHECK: sair.map[d0:{{.*}}, d1:{{.*}}] %[[V0]] {
    sair.map[d0:%2, d1:%1] %3(d1, d0) {
      // CHECK: ^{{.*}}(%[[ARG2:.*]]: index, %[[ARG3:.*]]: index,
      // CHECK-SAME: %[[ARG4:.*]]: memref<?x8xf32>):
      ^bb0(%arg2: index, %arg3: index, %arg4: f32):
        // CHECK: %[[IDX0:.*]] = affine.apply #[[$map2d1]](%[[ARG2]], %[[ARG3]])
        // CHECK: %[[IDX1:.*]] = affine.apply #[[$map2d0]](%[[ARG2]], %[[ARG3]])
        // CHECK: %[[V1:.*]] = memref.load %[[ARG4]][%[[IDX0]], %[[IDX1]]]
        // CHECK: %{{.*}} = addf %[[V1]], %[[V1]] : f32
        %4 = addf %arg4, %arg4 : f32
        sair.return
    } : #sair.shape<d0:range x d1:range>, (f32) -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @map
// CHECK: %[[ARG0:.*]]: index
func @map(%arg0: index) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %[[ARG0]]
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[V1:.*]] = sair.static_range 8
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[V2:.*]] = sair.dyn_range %[[V0]]
    %2 = sair.dyn_range %0 : !sair.range

    // CHECK: %[[V3:.*]] = sair.map %[[V0]] attributes {
    // CHECK:   storage = [{layout = #sair.named_mapping<[] -> ()>, space = "register"}]
    // CHECK: } {
    // CHECK: ^{{.*}}(%[[ARG1:.*]]: index):
    // CHECK: %[[V4:.*]] = memref.alloc(%[[ARG1]]) : memref<8x?xf32>
    // CHECK: sair.return %[[V4]] : memref<8x?xf32>
    // CHECK: } : #sair.shape<()>, (index) -> memref<8x?xf32>

    // CHECK: sair.map[d0:%[[V1]], d1:%[[V2]]] %[[V3]] attributes {
    // CHECK:   storage = []
    // CHECK: } {
    %3 = sair.map[d0:%1, d1:%2] attributes {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        {name = "loopB", iter = #sair.mapping_expr<d1>}
      ],
      storage = [{
        space = "memory", name = "A",
        layout = #sair.named_mapping<[d0:"loopA", d1:"loopB"] -> (d0, d1)>
      }]
    } {
      // CHECK: ^{{.*}}(%[[ARG2:.*]]: index, %[[ARG3:.*]]: index,
      // CHECK-SAME: %[[ARG4:.*]]: memref<8x?xf32>):
      ^bb0(%arg1: index, %arg2: index):
        // CHECK: %[[V5:.*]] = constant
        %4 = constant 1.0 : f32
        // CHECK: memref.store %[[V5]], %[[ARG4]][%[[ARG2]], %[[ARG3]]] : memref<8x?xf32>
        // CHECK-NOT: sair.return %{{.*}}
        // CHECK: sair.return
        sair.return %4 : f32
    // CHECK } : #sair.shape<d0:range x d1:range>, (memref<8x?xf32>) -> ()
    } : #sair.shape<d0:range x d1:range>, () -> f32

    // CHECK: sair.map[{{.*}}] %[[V3]] {
    sair.map[d0:%1, d1:%2] %3(d0, d1) {
      ^bb0(%arg3: index, %arg4: index, %arg5: f32):
        %5 = addf %arg5, %arg5 : f32
        sair.return
    // CHECK: } : #sair.shape<d0:range x d1:range>, (memref<8x?xf32>) -> ()
    } : #sair.shape<d0:range x d1:range>, (f32) -> ()

    // CHECK: sair.map %[[V3]] attributes {
    // CHECK:   storage = []
    // CHECK: } {
    // CHECK: ^{{.*}}(%[[ARG5:.*]]: memref<8x?xf32>):
    // CHECK: memref.dealloc %[[ARG5]]
    // CHECK: sair.return
    // CHECK: } : #sair.shape<()>, (memref<8x?xf32>) -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @loop_nest
func @loop_nest(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // CHECK: sair.copy
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "A"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.map
    // CHECK: loop_nest = []
    // CHECK: memref.alloc

    // CHECK: sair.copy
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "B"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "B", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.map
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "B"}]
    %2 = sair.map[d0:%1] attributes {
      loop_nest = [{name = "B", iter = #sair.mapping_expr<d0>}],
      storage = [{
        space = "memory", name = "bufferA",
        layout = #sair.named_mapping<[d0:"B"] -> (d0)>
      }]
    } {
      ^bb0(%arg1: index):
        %c0 = constant 1.0 : f32
        sair.return %c0 : f32
    } : #sair.shape<d0:range>, () -> f32

    // CHECK: sair.map
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "C"}]
    sair.map[d0:%1] %2(d0) attributes {
      loop_nest = [{name = "C", iter = #sair.mapping_expr<d0>}]
    } {
      ^bb0(%arg1: index, %arg2: f32):
        sair.return
    } : #sair.shape<d0:range>, (f32) -> ()

    // CHECK: sair.copy
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "C"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "C", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.map
    // CHECK: loop_nest = []
    // CHECK: memref.dealloc

    // CHECK: sair.copy
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "D"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "D", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}
