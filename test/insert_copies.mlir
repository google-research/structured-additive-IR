// RUN: sair-opt %s -sair-insert-copies | FileCheck %s
// RUN: sair-opt %s -sair-insert-copies -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: @from_to_memref
// CHECK: %[[ARG0:.*]]: memref<?x?xf32>
func @from_to_memref(%arg0 : memref<?x?xf32>) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.from_memref
    %1 = sair.from_memref[d0:%0, d1:%0] %arg0
      : memref<?x?xf32> -> !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[V0]], d1:%[[V0]]] %[[V1]](d1, d0)
    // CHECK: : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.to_memref[d0:%[[V0]], d1:%[[V0]]] %[[V2:.*]](d0, d1), %[[ARG0]]
    // CHECK: : memref<?x?xf32>
    sair.to_memref[d0:%0, d1:%0] %1(d1, d0), %arg0 : memref<?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @non_invertible_to_memref
// CHECK: %[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: f32
func @non_invertible_to_memref(%arg0: memref<?xf32>, %arg1: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[V0]]] %[[V1]]
    // CHECK:   : !sair.value<d0:range, f32>
    // CHECK: sair.to_memref[d0:%[[V0]]] %[[V2]](d0), %[[ARG0]] : memref<?xf32>
    sair.to_memref[d0:%0] %1, %arg0 : memref<?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @reduce_not_injective
func @reduce_not_injective(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.static_range
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[V1]]] %[[V0]]
    // CHECK: %{{.*}} = sair.map_reduce[d0:%[[V1]]] %[[V2]](d0) reduce {
    %2 = sair.map_reduce[d0:%1] %0 reduce {
      ^bb0(%arg1: index, %3: f32):
        sair.return %3 : f32
    } : #sair.shape<d0:range>, () -> f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @reduce_not_last_use
func @reduce_not_last_use(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[V0]]] %[[V1]]
    %2 = sair.copy[d0:%0] %1 : !sair.value<d0:range, f32>

    // Must copy %2 as it is used by %4.
    // CHECK: %[[V3:.*]] = sair.copy[d0:%[[V0]]] %[[V2]](d0)
    // CHECK: %{{.*}} = sair.map_reduce[d0:%[[V0]]] %[[V3]](d0) reduce {
    %3 = sair.map_reduce[d0:%0] %2(d0) reduce {
      ^bb0(%arg1: index, %3: f32):
        sair.return %3 : f32
    } : #sair.shape<d0:range>, () -> f32
    // CHECK: %{{.*}} = sair.copy[d0:%[[V0]]] %[[V2]](d0)
    %4 = sair.copy[d0:%0] %2(d0) : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @reduce_no_copies_needed
func @reduce_no_copies_needed(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[V0]]] %[[V1]]
    %2 = sair.copy[d0:%0] %1 : !sair.value<d0:range, f32>

    // CHECK: %{{.*}} = sair.map_reduce[d0:%[[V0]]] %[[V2]](d0) reduce {
    %3 = sair.map_reduce[d0:%0] %2(d0) reduce {
      ^bb0(%arg1: index, %3: f32):
        sair.return %3 : f32
    } : #sair.shape<d0:range>, () -> f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @reduce_memory_space_mismatch
func @reduce_memory_space_mismatch(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.copy %[[V0]] {memory_space = [1]}
    // CHECK: %{{.*}} = sair.map_reduce %[[V1]] reduce
    // CHECK: attributes {memory_space = [1]} {
    %1 = sair.map_reduce %0 reduce attributes {memory_space = [1]} {
      ^bb0(%2: f32):
        sair.return %2 : f32
    } : #sair.shape<()>, () -> f32
    sair.exit
  }
  return
}
// In the generic syntax, the function name is an attribute that is printed
// after the function. Check for it anyway to delimit the test.
// GENERIC-LABEL: sym_name = "reduce_memory_space_mismatch"

// CHECK-LABEL: @reduce_loop_nest
func @reduce_loop_nest(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.static_range
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[V1]]] %[[V0]]
    // CHECK:   loop_nest = [{iter = #sair.pattern_expr<d0>, name = "A"}]
    // GENERIC:      "sair.copy"
    // GENERIC-SAME: access_pattern_array = [#sair.pattern<1>]

    // CHECK: sair.copy[d0:%[[V1]], d1:%[[V1]]] %[[V0]]
    // GENERIC:      "sair.copy"
    // GENERIC-SAME: access_pattern_array = [#sair.pattern<2>]
    sair.copy[d0:%1, d1:%1] %0 {
      loop_nest = [
        {name = "A", iter = #sair.pattern_expr<d0>},
        {name = "B", iter = #sair.pattern_expr<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.map_reduce[d0:%[[V1]]] %[[V2]](d0) reduce[d1:%[[V1]]]
    // GENERIC: "sair.map_reduce"
    // GENERIC: access_pattern_array = [#sair.pattern<2 : d0>]
    sair.map_reduce[d0:%1] %0 reduce[d1:%1] attributes {
      loop_nest = [
        {name = "A", iter = #sair.pattern_expr<d0>},
        {name = "B", iter = #sair.pattern_expr<d1>}
      ]
    } {
      ^bb0(%arg1: index, %arg2: index, %arg3: f32):
         sair.return %arg3 : f32
    } : #sair.shape<d0:range x d1:range>, () -> f32
    sair.exit
  }
  return
}
// In the generic syntax, the function name is an attribute that is printed
// after the function. Check for it anyway to delimit the test.
// GENERIC-LABEL: sym_name = "reduce_loop_nest"
