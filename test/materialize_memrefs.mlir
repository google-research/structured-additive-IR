// RUN: sair-opt %s -sair-materialize-memrefs | FileCheck %s

// CHECK-LABEL: @from_memref
// CHECK: %[[ARG0:.*]]: index, %[[ARG1:.*]]: memref<?x8xf32>
func @from_memref(%arg0: index, %arg1: memref<?x8xf32>) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.range %0 : !sair.range
    %2 = sair.static_range 8 : !sair.range

    // CHECK: %[[V0:.*]] = sair.from_scalar %[[ARG1]]
    // CHECK: : !sair.value<(), memref<?x8xf32>
    %3 = sair.from_memref[d0:%1, d1:%2] %arg1
      : memref<?x8xf32> -> !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.map[d0:{{.*}}, d1:{{.*}}] %[[V0]] {
    sair.map[d0:%2, d1:%1] %3(d1, d0) {
      // CHECK: ^{{.*}}(%[[ARG2:.*]]: index, %[[ARG3:.*]]: index,
      // CHECK-SAME: %[[ARG4:.*]]: memref<?x8xf32>):
      ^bb0(%arg2: index, %arg3: index, %arg4: f32):
        // CHECK: %[[V1:.*]] = affine.load %[[ARG4]][%[[ARG3]], %[[ARG2]]]
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
    // CHECK: %[[V2:.*]] = sair.range %[[V0]]
    %2 = sair.range %0 : !sair.range

    // CHECK: %[[V3:.*]] = sair.map %[[V0]] attributes {memory_space = [0]} {
    // CHECK: ^{{.*}}(%[[ARG1:.*]]: index):
    // CHECK: %[[V4:.*]] = alloc(%[[ARG1]]) : memref<8x?xf32>
    // CHECK: sair.return %[[V4]] : memref<8x?xf32>
    // CHECK: } : #sair.shape<()>, (index) -> memref<8x?xf32>

    // CHECK: sair.map[d0:%[[V1]], d1:%[[V2]]] %[[V3]] attributes {memory_space = []} {
    %3 = sair.map[d0:%1, d1:%2] attributes {memory_space=[1]} {
      // CHECK: ^{{.*}}(%[[ARG2:.*]]: index, %[[ARG3:.*]]: index,
      // CHECK-SAME: %[[ARG4:.*]]: memref<8x?xf32>):
      ^bb0(%arg1: index, %arg2: index):
        // CHECK: %[[V5:.*]] = constant
        %4 = constant 1.0 : f32
        // CHECK: store %[[V5]], %[[ARG4]][%[[ARG2]], %[[ARG3]]] : memref<8x?xf32>
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

    // CHECK: sair.map %[[V3]] attributes {memory_space = []} {
    // CHECK: ^{{.*}}(%[[ARG5:.*]]: memref<8x?xf32>):
    // CHECK: dealloc %[[ARG5]]
    // CHECK: sair.return
    // CHECK: } : #sair.shape<()>, (memref<8x?xf32>) -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @map_reduce_input
func @map_reduce_input() {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.map[d0:%0] attributes {memory_space=[1]} {
      ^bb0(%arg0: index):
        %c0 = constant 1.0 : f32
        sair.return %c0 : f32
    } : #sair.shape<d0:range>, () -> (f32)
    sair.map_reduce reduce[d0: %0] %1(d0) {
      // CHECK: ^{{.*}}(%[[ARG0:.*]]: index, %[[ARG1:.*]]: memref<8xf32>):
      ^bb0(%arg0: index, %arg1: f32):
        // CHECK: %[[V0:.*]] = affine.load %{{.*}}[%[[ARG0]]]
        // CHECK: %{{.*}} = addf %[[V0]], %[[V0]] : f32
        %2 = addf %arg1, %arg1 : f32
        // CHECK-NOT: store
        // CHECK: sair.return
        sair.return
    // CHECK: #sair.shape<d0:range>, (memref<8xf32>) -> ()
    } : #sair.shape<d0:range>, (f32) -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @map_reduce_init()
func @map_reduce_init() {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.map attributes {memory_space = [0]} {
      // CHECK: alloc
    // CHECK: } : #sair.shape<()>, () -> memref<8x8xf32>
    // CHECK: sair.map[d0:%[[V0]], d1:%[[V0]]] %[[V1]] attributes {memory_space = []} {
    %1 = sair.map[d0:%0, d1:%0] attributes {memory_space=[1]} {
      ^bb0(%arg0: index, %arg1: index):
        %c0 = constant 1.0 : f32
        sair.return %c0 : f32
    // CHECK: } : #sair.shape<d0:range x d1:range>, (memref<8x8xf32>) -> ()
    } : #sair.shape<d0:range x d1:range>, () -> (f32)

    // CHECK: sair.map_reduce[d0:%[[V0]], d1:%[[V0]]] reduce %[[V1]] attributes {memory_space = []} {
    %2 = sair.map_reduce[d0:%0, d1:%0] %1(d1, d0) reduce attributes {
      memory_space=[1]
    } {
      // CHECK: ^{{.*}}(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<8x8xf32>):
      ^bb0(%arg0: index, %arg1: index, %arg2: f32):
        // CHECK: %[[V2:.*]] = affine.load %[[ARG2]][%[[ARG1]], %[[ARG0]]] : memref<8x8xf32>
        // CHECK: %[[V3:.*]] = addf %[[V2]], %[[V2]] : f32
        %3 = addf %arg2, %arg2 : f32
        // CHECK: affine.store %[[V3]], %[[ARG2]][%[[ARG1]], %[[ARG0]]] : memref<8x8xf32>
        // CHECK-NOT: sair.return %{{.*}}
        // CHECK: sair.return
        sair.return %3 : f32
    // CHECK: } : #sair.shape<d0:range x d1:range>, (memref<8x8xf32>) -> ()
    } : #sair.shape<d0:range x d1:range>, () -> (f32)

    // CHECK: sair.map[d0:%[[V0]], d1:%[[V0]]] %[[V1]] attributes {memory_space = []} {
    sair.map[d0:%0, d1:%0] %2(d0, d1) attributes {memory_space = []} {
      // CHECK: ^{{.*}}(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<8x8xf32>):
      ^bb0(%arg0: index, %arg1: index, %arg2: f32):
        // CHECK: %[[V2:.*]] = affine.load %[[ARG2]][%[[ARG1]], %[[ARG0]]] : memref<8x8xf32>
        // CHECK: %{{.*}} = addf %[[V2]], %[[V2]] : f32
        %4 = addf %arg2, %arg2 : f32
        sair.return
    // CHECK: } : #sair.shape<d0:range x d1:range>, (memref<8x8xf32>) -> ()
    } : #sair.shape<d0:range x d1:range>, (f32) -> ()

    // CHECK: sair.map %[[V1]] attributes {memory_space = []} {
      // CHECK: dealloc
    // CHECK: } : #sair.shape<()>, (memref<8x8xf32>) -> ()
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
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "A"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.map
    // CHECK: loop_nest = []
    // CHECK: alloc

    // CHECK: sair.copy
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "B"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "B", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.map
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "B"}]
    %2 = sair.map[d0:%1] attributes {
      loop_nest = [{name = "B", iter = #sair.iter<d0>}],
      memory_space = [1]
    } {
      ^bb0(%arg1: index):
        %c0 = constant 1.0 : f32
        sair.return %c0 : f32
    } : #sair.shape<d0:range>, () -> f32

    // CHECK: sair.map
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "C"}]
    sair.map[d0:%1] %2(d0) attributes {
      loop_nest = [{name = "C", iter = #sair.iter<d0>}]
    } {
      ^bb0(%arg1: index, %arg2: f32):
        sair.return
    } : #sair.shape<d0:range>, (f32) -> ()

    // CHECK: sair.copy
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "C"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "C", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.map
    // CHECK: loop_nest = []
    // CHECK: dealloc

    // CHECK: sair.copy
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "D"}]
    sair.copy[d0:%1] %0 {
      loop_nest = [{name = "D", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}
