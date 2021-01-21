// RUN: sair-opt %s -sair-normalize-loops -cse -mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: @identity
func @identity(%arg0: index, %arg1: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // CHECK: %[[D0:.*]] = sair.static_range
    %2 = sair.static_range 8 : !sair.range
    // CHECK: %[[V1:.*]] = sair.map %[[V0]]
    // CHECK: %[[D1:.*]] = sair.dyn_range %[[V1]]
    %3 = sair.dyn_range %0 : !sair.range
    // CHECK: %[[V2:.*]] = sair.fby[d0:%[[D0]]] %{{.*}} then[d1:%[[D1]]] %[[V3:.*]](d0, d1)
    %4 = sair.fby[d0:%2] %1 then[d1:%3] %5(d0, d1)
      : !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[V3]] = sair.copy[d0:%[[D0]], d1:%[[D1]]]
    %5 = sair.copy[d0:%2, d1:%3] %4(d0, d1) {
      loop_nest = [
        // CHECK: iter = #sair.mapping_expr<d0>, name = "loopA"
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        // CHECK: iter = #sair.mapping_expr<d1>, name = "loopB"
        {name = "loopB", iter = #sair.mapping_expr<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[V4:.*]] = sair.proj_last of[d0:%[[D0]], d1:%[[D1]]] %[[V3]](d0, d1)
    %6 = sair.proj_last of[d0:%2, d1:%3] %5(d0, d1)
      : #sair.shape<d0:range x d1:range>, f32
    // CHECK: sair.exit %[[V4]]
    sair.exit %6 : f32
  } : f32
  return
}

// CHECK-LABEL: @stripe
func @stripe() {
  sair.program {
    %0 = sair.static_range 62 : !sair.range
    // CHECK: %[[D0:.*]] = sair.static_range 62 step 4

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
    // CHECK-SAME: !sair.range<d0:range>

    // CHECK: %[[V7:.*]] = sair.map[d0:%[[D0]], d1:%[[D1]]]
    %1 = sair.map[d0: %0] attributes {
      loop_nest = [
        // CHECK: iter = #sair.mapping_expr<d0>, name = "loopA"
        {name = "loopA", iter = #sair.mapping_expr<stripe(d0, 4)>},
        // CHECK: iter = #sair.mapping_expr<d1>, name = "loopB"
        {name = "loopB", iter = #sair.mapping_expr<stripe(d0, 1 size 4)>}
      ]
    } {
      // CHECK: ^bb0(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index):
      ^bb0(%arg0: index):
        // CHECK: %[[V8:.*]] = affine.apply affine_map<(d0, d1) -> (d1)>(%[[ARG0]], %[[ARG1]])
        // CHECK: sair.return %[[V8]] : index
        sair.return %arg0 : index
    } : #sair.shape<d0:range>, () -> (index)
    // CHECK: %[[V9:.*]] = sair.proj_any of[d0:%[[D0]], d1:%[[D1]]] %[[V7]](d0, d1)
    // CHECK: #sair.shape<d0:range x d1:range(d0)>, index
    %2 = sair.proj_any of[d0:%0] %1(d0) : #sair.shape<d0:range>, index
    // CHECK: sair.exit %[[V9]]
    sair.exit %2 : index
  } : index
  return
}

// CHECK-LABEL: @unstripe
func @unstripe(%arg0: f32) {
  sair.program {
    // CHECK-DAG: %[[D0:.*]] = sair.static_range 4 : !sair.range
    %0 = sair.static_range 4 step 4 : !sair.range
    %1 = sair.static_range 4 : !sair.range
    // CHECK-DAG: %[[V0:.*]] = sair.from_scalar
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.map_reduce %[[V0]] reduce[d0:%[[D0]]] attributes
    // CHECK: loop_nest = [{iter = #sair.mapping_expr<d0>, name = "loopA"}]
    %3 = sair.map_reduce %2 reduce[d0:%0, d1:%0] attributes {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<unstripe(d0, d1, [4])>}]
    } {
      // CHECK: ^bb0(%[[V2:.*]]: index, %[[V3:.*]]: f32):
      ^bb0(%arg1: index, %arg2: index, %arg3: f32):
        // CHECK: sair.return %[[V3]]
        sair.return %arg3: f32
    } : #sair.shape<d0:range x d1:range>, () -> (f32)
    // CHECK: sair.exit %[[V1]]
    sair.exit %3 : f32
  } : f32
  return
}

// CHECK-LABEL: @from_to_memref
func @from_to_memref(%arg0: index) {
  sair.program {
    // CHECK: %[[SIZE:.*]] = sair.from_scalar
    %size = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.static_range 4
    %0 = sair.static_range 4 : !sair.range
    // CHECK: %[[D1:.*]] = sair.dyn_range %{{.*}}
    // CHECK: %[[D2:.*]] = sair.dyn_range %{{.*}}
    %1 = sair.dyn_range %size : !sair.range
    // CHECK: %[[MEMREF:.*]] = sair.alloc[d0:%[[D0]]] %[[SIZE]]
    %memref = sair.alloc[d0:%0] %size {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, memref<?xf32>>
    // CHECK: %[[V0:.*]] = sair.from_memref[d0:%[[D0]]] %[[MEMREF]](d0) memref[d1:%[[D1]]]
    %2 = sair.from_memref[d0:%0] %memref(d0) memref[d1:%1]
      : #sair.shape<d0:range x d1:range>, memref<?xf32>
    // CHECK: sair.to_memref[d0:%[[D0]]] %[[MEMREF]](d0) memref[d1:%[[D2]]] %[[V0]](d0, d1)
    sair.to_memref[d0:%0] %memref(d0) memref[d1:%1] %2(d0, d1)
      : #sair.shape<d0:range x d1:range>, memref<?xf32>
    sair.free[d0:%0] %memref(d0) {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, memref<?xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @load_store_memref
func @load_store_memref(%arg0: index) {
  sair.program {
    // CHECK: %[[SIZE:.*]] = sair.from_scalar
    %size = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.static_range 4 step 4
    %0 = sair.static_range 4 : !sair.range
    // CHECK: %[[MAPPED:.*]] = sair.map %[[SIZE]]
    // CHECK: %[[D2:.*]] = sair.dyn_range %[[MAPPED]]
    %1 = sair.dyn_range %size : !sair.range
    // CHECK: %[[DYNMAPPED:.*]]:2 = sair.map[d0:{{.*}}]
    // CHECK: %[[D1:.*]] = sair.dyn_range[d0:{{.*}}] %[[DYNMAPPED]]#0(d0), %[[DYNMAPPED]]#1(d0)
    // CHECK: sair.alloc[d0:%[[D0]], d1:%[[D1]]]
    // CHECK: loop_nest = [
    // CHECK:   {iter = #sair.mapping_expr<d0>, name = "A"}
    // CHECK:   {iter = #sair.mapping_expr<d1>, name = "B"}
    %memref = sair.alloc[d0:%0] %size {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<stripe(d0, 4)>},
        {name = "B", iter = #sair.mapping_expr<stripe(d0, 1 size 4)>}
      ]
    } : !sair.value<d0:range, memref<?xf32>>
    // CHECK: sair.load_from_memref[d0:%[[D0]], d1:%[[D1]]] %{{.*}}(d0, d1) memref[d2:%[[D2]]]
    // CHECK: loop_nest = [
    // CHECK:   {iter = #sair.mapping_expr<d0>, name = "A"}
    // CHECK:   {iter = #sair.mapping_expr<d1>, name = "B"}
    // CHECK:   {iter = #sair.mapping_expr<d2>, name = "C"}
    // CHECK: #sair.shape<d0:range x d1:range(d0) x d2:range>
    %2 = sair.load_from_memref[d0:%0] %memref(d0) memref[d1:%1]
      { loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, 4)>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, 1 size 4)>},
          {name = "C", iter = #sair.mapping_expr<d1>}
        ]
      } : #sair.shape<d0:range x d1:range>, memref<?xf32>
    // CHECK: sair.store_to_memref[d0:%[[D0]], d1:%[[D1]]] %{{.*}}(d0, d1) memref[d2:%[[D2]]] %{{.*}}(d0, d1, d2)
    // CHECK: loop_nest = [
    // CHECK:   {iter = #sair.mapping_expr<d0>, name = "A"}
    // CHECK:   {iter = #sair.mapping_expr<d1>, name = "B"}
    // CHECK:   {iter = #sair.mapping_expr<d2>, name = "C"}
    // CHECK: #sair.shape<d0:range x d1:range(d0) x d2:range>
    sair.store_to_memref[d0:%0] %memref(d0) memref[d1:%1] %2(d0, d1)
      { loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, 4)>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, 1 size 4)>},
          {name = "C", iter = #sair.mapping_expr<d1>}
        ]
      } : #sair.shape<d0:range x d1:range>, memref<?xf32>
    sair.free[d0:%0] %memref(d0) {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<stripe(d0, 4)>},
        {name = "B", iter = #sair.mapping_expr<stripe(d0, 1 size 4)>}
      ]
    } : !sair.value<d0:range, memref<?xf32>>
    sair.exit
  }
  return
}
