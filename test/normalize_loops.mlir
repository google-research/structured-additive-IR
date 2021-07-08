// RUN: sair-opt %s -sair-normalize-loops -cse -mlir-print-local-scope | FileCheck %s
// RUN: sair-opt %s -sair-normalize-loops -cse -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC


// CHECK-LABEL: @identity
func @identity(%arg0: index, %arg1: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // CHECK: %[[D0:.*]] = sair.static_range
    %2 = sair.static_range : !sair.static_range<8>
    // CHECK: %[[V1:.*]] = sair.map %[[V0]]
    // CHECK: %[[D1:.*]] = sair.dyn_range %[[V1]]
    %3 = sair.dyn_range %0 : !sair.dyn_range
    // CHECK: %[[V2:.*]] = sair.fby[d0:%[[D0]]] %{{.*}} then[d1:%[[D1]]] %[[V3:.*]](d0, d1)
    %4 = sair.fby[d0:%2] %1 then[d1:%3] %5(d0, d1)
      : !sair.value<d0:static_range<8> x d1:dyn_range, f32>
    // CHECK: %[[V3]] = sair.copy[d0:%[[D0]], d1:%[[D1]]]
    %5 = sair.copy[d0:%2, d1:%3] %4(d0, d1) {
      decisions = {
        loop_nest = [
          // CHECK: iter = #sair.mapping_expr<d0>, name = "loopA"
          {name = "loopA", iter = #sair.mapping_expr<d0>},
          // CHECK: iter = #sair.mapping_expr<d1>, name = "loopB"
          {name = "loopB", iter = #sair.mapping_expr<d1>}
        ],
        storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
      }
    } : !sair.value<d0:static_range<8> x d1:dyn_range, f32>
    // CHECK: %[[V4:.*]] = sair.proj_last of[d0:%[[D0]], d1:%[[D1]]] %[[V3]](d0, d1)
    %6 = sair.proj_last of[d0:%2, d1:%3] %5(d0, d1)
      : #sair.shape<d0:static_range<8> x d1:dyn_range>, f32
    // CHECK: sair.exit %[[V4]]
    sair.exit %6 : f32
  } : f32
  return
}

// CHECK-LABEL: @stripe
func @stripe() {
  sair.program {
    %0 = sair.static_range : !sair.static_range<62>
    // CHECK: %[[D0:.*]] = sair.static_range : !sair.static_range<62, 4>

    // CHECK: %[[V0:.*]]:2 = sair.map[d0:%[[D0]]]
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "loopA"}]
    // CHECK:   ^bb0(%[[ARG0:.*]]: index):
    // CHECK:     %[[V1:.*]] = affine.apply affine_map<(d0) -> (d0)>(%arg0)
    // CHECK:     %[[V2:.*]] = constant 4 : index
    // CHECK:     %[[V3:.*]] = addi %[[V1]], %[[V2]] : index
    // CHECK:     %[[V4:.*]] = constant 62 : index
    // CHECK:     %[[V5:.*]] = cmpi ult, %[[V4]], %[[V3]] : index
    // CHECK:     %[[V6:.*]] = select %[[V5]], %[[V4]], %[[V3]] : index
    // CHECK:     sair.return %[[V1]], %[[V6]] : index, index

    // CHECK: %[[D1:.*]] = sair.dyn_range[d0:%[[D0]]] %[[V0]]#0(d0), %[[V0]]#1(d0)
    // CHECK-SAME: !sair.dyn_range<d0:static_range<62, 4>>

    // CHECK: %[[V7:.*]] = sair.map[d0:%[[D0]], d1:%[[D1]]]
    %1 = sair.map[d0: %0] attributes {
      decisions = {
        loop_nest = [
          // CHECK: iter = #sair.mapping_expr<d0>, name = "loopA"
          {name = "loopA", iter = #sair.mapping_expr<stripe(d0, [4])>},
          // CHECK: iter = #sair.mapping_expr<d1>, name = "loopB"
          {name = "loopB", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ]
      }
    } {
      // CHECK: ^bb0(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index):
      ^bb0(%arg0: index):
        // CHECK: %[[V8:.*]] = affine.apply affine_map<(d0, d1) -> (d1)>(%[[ARG0]], %[[ARG1]])
        // CHECK: sair.return %[[V8]] : index
        sair.return %arg0 : index
    } : #sair.shape<d0:static_range<62>>, () -> (index)
    // CHECK: %[[V9:.*]] = sair.proj_any of[d0:%[[D0]], d1:%[[D1]]] %[[V7]](d0, d1)
    // CHECK: #sair.shape<d0:static_range<62, 4> x d1:dyn_range(d0)>, index
    %2 = sair.proj_any of[d0:%0] %1(d0) : #sair.shape<d0:static_range<62>>, index
    // CHECK: sair.exit %[[V9]]
    sair.exit %2 : index
  } : index
  return
}

// CHECK-LABEL: @unstripe
func @unstripe(%arg0: f32) {
  %c4 = constant 4 : index
  sair.program {
    %sc4 = sair.from_scalar %c4 : !sair.value<(), index>
    // CHECK-DAG: %[[D0:.*]] = sair.static_range : !sair.static_range<4>
    %0 = sair.static_range : !sair.static_range<4, 4>
    %1 = sair.dyn_range[d0:%0] %sc4 : !sair.dyn_range<d0:static_range<4, 4>>

    // CHECK-DAG: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.map_reduce %[[V0]] reduce[d0:%[[D0]]] attributes
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "loopA"}]
    %3 = sair.map_reduce %2 reduce[d0:%0, d1:%1] attributes {
      decisions = {
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>}],
        storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
      }
    } {
      // CHECK: ^bb0(%[[V2:.*]]: index, %[[V3:.*]]: f32):
      ^bb0(%arg1: index, %arg2: index, %arg3: f32):
        // CHECK: sair.return %[[V3]]
        sair.return %arg3: f32
    } : #sair.shape<d0:static_range<4, 4> x d1:dyn_range(d0)>, () -> (f32)
    // CHECK: sair.exit %[[V1]]
    sair.exit %3 : f32
  } : f32
  return
}

// CHECK-LABEL: @load_store_memref
func @load_store_memref(%arg0: index) {
  sair.program {
    // CHECK: %[[SIZE:.*]] = sair.from_scalar
    %size = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.static_range : !sair.static_range<4, 4>
    %0 = sair.static_range : !sair.static_range<4>
    // CHECK: %[[MAPPED:.*]] = sair.map %[[SIZE]]
    // CHECK: %[[D2:.*]] = sair.dyn_range %[[MAPPED]]
    %1 = sair.dyn_range %size : !sair.dyn_range
    // CHECK: %[[DYNMAPPED:.*]]:2 = sair.map[d0:{{.*}}]
    // CHECK: %[[D1:.*]] = sair.dyn_range[d0:{{.*}}] %[[DYNMAPPED]]#0(d0), %[[DYNMAPPED]]#1(d0)
    // CHECK: sair.alloc[d0:%[[D0]], d1:%[[D1]]]
    // CHECK: loop_nest = [
    // CHECK:   {iter = #sair.mapping_expr<d0>, name = "A"}
    // CHECK:   {iter = #sair.mapping_expr<d1>, name = "B"}
    %memref = sair.alloc[d0:%0] %size {
      decisions = {
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ]
      }
    } : !sair.value<d0:static_range<4>, memref<?xf32>>
    // CHECK: sair.load_from_memref[d0:%[[D0]], d1:%[[D1]], d2:%[[D2]]] %{{.*}}(d0, d1)
    // CHECK: loop_nest = [
    // CHECK:   {iter = #sair.mapping_expr<d0>, name = "A"}
    // CHECK:   {iter = #sair.mapping_expr<d1>, name = "B"}
    // CHECK:   {iter = #sair.mapping_expr<d2>, name = "C"}
    // CHECK: layout = #sair.mapping<3 : d2>
    // CHECK: memref<?xf32> -> !sair.value<d0:static_range<4, 4> x d1:dyn_range(d0) x d2:dyn_range, f32>
    %2 = sair.load_from_memref[d0:%0, d1:%1] %memref(d0) {
      layout = #sair.mapping<2 : d1>,
      decisions = {
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>},
          {name = "C", iter = #sair.mapping_expr<d1>}
        ]
      }
    } : memref<?xf32> -> !sair.value<d0:static_range<4> x d1:dyn_range, f32>
    // CHECK: sair.store_to_memref[d0:%[[D0]], d1:%[[D1]], d2:%[[D2]]] %{{.*}}(d0, d1), %{{.*}}(d0, d1, d2)
    // CHECK: loop_nest = [
    // CHECK:   {iter = #sair.mapping_expr<d0>, name = "A"}
    // CHECK:   {iter = #sair.mapping_expr<d1>, name = "B"}
    // CHECK:   {iter = #sair.mapping_expr<d2>, name = "C"}
    // CHECK: layout = #sair.mapping<3 : d2>
    // CHECK: #sair.shape<d0:static_range<4, 4> x d1:dyn_range(d0) x d2:dyn_range>
    sair.store_to_memref[d0:%0, d1:%1] %memref(d0), %2(d0, d1) {
      layout = #sair.mapping<2 : d1>,
      decisions = {
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>},
          {name = "C", iter = #sair.mapping_expr<d1>}
        ]
      }
    } : #sair.shape<d0:static_range<4> x d1:dyn_range>, memref<?xf32>
    sair.free[d0:%0] %memref(d0) {
      decisions = {
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ]
      }
    } : !sair.value<d0:static_range<4>, memref<?xf32>>
    sair.exit
  }
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "load_store_memref"

// CHECK-LABEL: @remat
func @remat(%arg0: f32) {
  sair.program {
    // CHECK: %[[INIT:.*]] = sair.from_scalar
    // GENERIC: %[[INIT:.*]] = "sair.from_scalar"
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[RANGE:.*]] = sair.static_range
    // GENERIC: %[[RANGE:.*]] = "sair.static_range"
    %1 = sair.static_range : !sair.static_range<8>
    // CHECK: %[[RESULT:.*]] = sair.copy[d0:%[[RANGE]]] %[[INIT]]
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "A"}]
    // CHECK: !sair.value<d0:static_range<8>, f32>
    //
    // CHECK: %[[P0:.*]] = sair.placeholder : !sair.static_range<8>
    // CHECK: %[[VALUE:.*]] = sair.proj_any of[d0:%[[P0]]] %[[RESULT]](d0)
    // CHECK: #sair.shape<d0:static_range<8>>
    //
    // Ensure that the mapping has the expected use domain, and that the
    // resulting value has the right type.
    // GENERIC: "sair.copy"(%[[RANGE]], %[[INIT]])
    // GENERIC-SAME: mapping_array = [#sair.mapping<1>]
    // GENERIC-SAME: (!sair.static_range<8>, !sair.value<(), f32>) -> !sair.value<d0:static_range<8>, f32>
    %2 = sair.copy %0 {
      decisions = {
        loop_nest = [{name = "A", iter = #sair.mapping_expr<none>}]
      }
    } : !sair.value<(), f32>
    // CHECK: sair.copy[d0:%[[RANGE]]] %[[VALUE]]
    %3 = sair.copy[d0:%1] %2 {
      decisions = {
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
      }
    } : !sair.value<d0:static_range<8>, f32>
    %4 = sair.proj_last of[d0:%1] %3(d0) : #sair.shape<d0:static_range<8>>, f32
    sair.exit %4 : f32
  } : f32
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "remat"


func private @foo(index, f32)

// CHECK-LABEL: @sequence_attr
func @sequence_attr(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[STATIC:.*]] = sair.static_range : !sair.static_range<16, 4>
    %1 = sair.static_range : !sair.static_range<16>

    // CHECK: %[[RANGE:.*]]:2 = sair.map
    // CHECK-SAME: sequence = 0
    // CHECK: ^{{.*}}(%[[ARG:.*]]: index):
    // CHECK:   %[[V1:.*]] = affine.apply affine_map<(d0) -> (d0)>(%[[ARG]])
    // CHECK:   %[[C4:.*]] = constant 4
    // CHECK:   %[[V2:.*]] = addi %[[V1]], %[[C4]]
    // CHECK:   %[[C16:.*]] = constant 16
    // CHECK:   %[[V3:.*]] = cmpi ult, %[[C16]], %[[V2]]
    // CHECK:   %[[V4:.*]] = select %[[V3]], %[[C16]], %[[V2]]
    // CHECK:   sair.return %[[V1]], %[[V4]]
    // CHECK: %[[DYN:.*]] = sair.dyn_range[d0:%[[STATIC]]] %[[RANGE]]#0(d0), %[[RANGE]]#1(d0)

    // CHECK: sair.map[d0:%[[STATIC]], d1:%[[DYN]]]
    // CHECK-SAME: sequence = 1
    sair.map[d0:%1] %0 attributes {
      decisions = {
        loop_nest = [
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ],
        sequence = 1
      }
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()

    // CHECK: %[[OTHER_STATIC:.*]] = sair.static_range : !sair.static_range<16>
    // CHECK: sair.map[d0:%[[OTHER_STATIC]]]
    // CHECK-SAME: sequence = 3
    sair.map[d0:%1] %0 attributes {
      decisions = {
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
        sequence = 3
      }
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()

    // CHECK: sair.map[d0:%[[STATIC]], d1:%[[DYN]]]
    // CHECK-SAME: sequence = 2
    sair.map[d0:%1] %0 attributes {
      decisions = {
        loop_nest = [
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ],
        sequence = 2
      }
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()
    sair.exit
  }
  return
}

// Loop bound computation code will be inserted before the operation with the
// lowest sequence number that participates in the loop. This is fine because
// Sair doesn't enforce use-def order but relies on sequences instead.
// CHECK-LABEL: @sequence_attr_inversion
func @sequence_attr_inversion(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range : !sair.static_range<16>

    // CHECK: sair.map[d0:%[[STATIC:.*]], d1:%[[DYN:.*]]] %{{.*}} attributes
    // CHECK-SAME: sequence = 2
    sair.map[d0:%1] %0 attributes {
      decisions = {
        loop_nest = [
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ],
        sequence = 2
      }
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()

    // CHECK: %[[OTHER_STATIC:.*]] = sair.static_range : !sair.static_range<16>
    // CHECK: sair.map[d0:%[[OTHER_STATIC]]]
    // CHECK-SAME: sequence = 3
    sair.map[d0:%1] %0 attributes {
      decisions = {
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
        sequence = 3
      }
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()

    // CHECK: %[[STATIC]] = sair.static_range : !sair.static_range<16, 4>
    // CHECK: %[[RANGE:.*]]:2 = sair.map
    // CHECK-SAME: sequence = 0
    // CHECK: ^{{.*}}(%[[ARG:.*]]: index):
    // CHECK:   %[[V1:.*]] = affine.apply affine_map<(d0) -> (d0)>(%[[ARG]])
    // CHECK:   %[[C4:.*]] = constant 4
    // CHECK:   %[[V2:.*]] = addi %[[V1]], %[[C4]]
    // CHECK:   %[[C16:.*]] = constant 16
    // CHECK:   %[[V3:.*]] = cmpi ult, %[[C16]], %[[V2]]
    // CHECK:   %[[V4:.*]] = select %[[V3]], %[[C16]], %[[V2]]
    // CHECK:   sair.return %[[V1]], %[[V4]]
    // CHECK: %[[DYN]] = sair.dyn_range[d0:%[[STATIC]]] %[[RANGE]]#0(d0), %[[RANGE]]#1(d0)

    // CHECK: sair.map[d0:%[[STATIC]], d1:%[[DYN]]]
    // CHECK-SAME: sequence = 1
    sair.map[d0:%1] %0 attributes {
      decisions = {
        loop_nest = [
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ],
        sequence = 1
      }
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @unroll_preserved
func @unroll_preserved(%arg0: index, %arg1: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    %2 = sair.static_range : !sair.static_range<8>
    %3 = sair.dyn_range %0 : !sair.dyn_range
    %4 = sair.fby[d0:%2] %1 then[d1:%3] %5(d0, d1)
      : !sair.value<d0:static_range<8> x d1:dyn_range, f32>
    %5 = sair.copy[d0:%2, d1:%3] %4(d0, d1) {
      decisions = {
        loop_nest = [
          // CHECK: name = "loopA"
          // CHECK-SAME: unroll = 42
          {name = "loopA", iter = #sair.mapping_expr<d0>, unroll = 42},
          // CHECK: name = "loopB"
          // CHECK-SAME: unroll = 10
          {name = "loopB", iter = #sair.mapping_expr<d1>, unroll = 10}
        ],
        storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
      }
    } : !sair.value<d0:static_range<8> x d1:dyn_range, f32>
    %6 = sair.proj_last of[d0:%2, d1:%3] %5(d0, d1)
      : #sair.shape<d0:static_range<8> x d1:dyn_range>, f32
    sair.exit %6 : f32
  } : f32
  return
}

// CHECK-LABEL: @unroll_propagated
func @unroll_propagated() {
  sair.program {
    %0 = sair.static_range : !sair.static_range<62>
    // CHECK: %[[D0:.*]] = sair.static_range : !sair.static_range<62, 4>

    // CHECK: sair.map[d0:%[[D0]]]
    // CHECK-SAME: name = "loopA"
    // CHECK-SAME: unroll = 10

    // CHECK: sair.map[d0:%[[D0]], d1:%{{.*}}] attributes
    %1 = sair.map[d0: %0] attributes {
      decisions = {
        loop_nest = [
          // CHECK: name = "loopA"
          // CHECK-SAME: unroll = 10
          {name = "loopA", iter = #sair.mapping_expr<stripe(d0, [4])>, unroll = 10},
          // CHECK: name = "loopB"
          // CHECK-NOT: unroll
          {name = "loopB", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ]
      }
    } {
      ^bb0(%arg0: index):
        sair.return %arg0 : index
    } : #sair.shape<d0:static_range<62>>, () -> (index)
    %2 = sair.proj_any of[d0:%0] %1(d0) : #sair.shape<d0:static_range<62>>, index
    sair.exit %2 : index
  } : index
  return
}
