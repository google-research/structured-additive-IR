// RUN: sair-opt %s -canonicalize | FileCheck %s

func @use(%arg0: f32) {
  return
}

// CHECK-LABEL: @deduplicate_map_input
func @deduplicate_map_input(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.map %[[V0]] attributes {storage = []} {
    sair.map %0, %0 {
      // CHECK: ^bb0(%[[V1:.*]]: f32):
      ^bb0(%arg1: f32, %arg2: f32):
        // CHECK: addf %[[V1]], %[[V1]] : f32
        %1 = addf %arg1, %arg2 : f32
        call @use(%1) : (f32) -> ()
        sair.return
    // CHECK: } : #sair.shape<()>, (f32) -> ()
    } : #sair.shape<()>, (f32, f32) -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @deduplicate_map_output
func @deduplicate_map_output() {
  %3, %4 = sair.program {
    // CHECK: %[[V0:.*]] = sair.map
    %0, %1 = sair.map {
      ^bb0:
        // CHECK: %[[V1:.*]] = constant
        %2 = constant 1.0 : f32
        // CHECK: sair.return %[[V1]] : f32
        sair.return %2, %2 : f32, f32
    // CHECK: #sair.shape<()>, () -> f32
    } : #sair.shape<()>, () -> (f32, f32)
    // CHECK: sair.exit %[[V0]], %[[V0]] : f32, f32
    sair.exit %0, %1 : f32, f32
  } : f32, f32
  return
}

// CHECK-LABEL: @fold_empty_proj
func @fold_empty_proj(%arg0: f32) {
  %0 = sair.program {
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.copy
    %3 = sair.copy[d0:%2] %1 : !sair.value<d0:range, f32>
    %4 = sair.proj_last[d0:%2] of %3(d0) : #sair.shape<d0:range>, f32
    // CHECK: %[[V1:.*]] = sair.proj_last of[d0:%{{.*}}] %[[V0]](d0)
    %5 = sair.proj_last of[d0:%2] %4(d0) : #sair.shape<d0:range>, f32
    // CHECK: sair.exit %[[V1]] : f32
    sair.exit %5 : f32
  } : f32
  sair.return
}

// CHECK-LABEL: @fold_empty_fby
func @fold_empty_fby(%arg0: f32) {
  %0 = sair.program {
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.copy
    %3 = sair.copy[d0:%2] %1 : !sair.value<d0:range, f32>
    %4 = sair.fby[d0:%2] %3(d0) then %3(d0) : !sair.value<d0:range, f32>
    // CHECK: %[[V1:.*]] = sair.proj_last of[d0:%{{.*}}] %[[V0]](d0)
    %5 = sair.proj_last of[d0:%2] %4(d0) : #sair.shape<d0:range>, f32
    // CHECK: sair.exit %[[V1]] : f32
    sair.exit %5 : f32
  } : f32
  sair.return
}

// CHECK-LABEL: @merge_proj
func @merge_proj(%arg0: f32) {
  %0 = sair.program {
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.copy
    %3 = sair.copy[d0:%2, d1: %2] %1 : !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[V1:.*]] = sair.proj_last of[d0:%{{.*}}, d1:%{{.*}}] %[[V0]](d0, d1)
    // CHECK: #sair.shape<d0:range x d1:range>, f32
    %4 = sair.proj_last[d0:%2] of[d1:%2] %3(d0, d1)
      : #sair.shape<d0:range x d1:range>, f32
    %5 = sair.proj_last of[d0:%2] %4(d0) : #sair.shape<d0:range>, f32
    // CHECK: sair.exit %[[V1]]
    sair.exit %5 : f32
  } : f32
  return
}

// CHECK-LABEL: @remove_cyclic_fby
func @remove_cyclic_fby(%arg0: f32, %arg1: memref<?x?x?xf32>) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %4 = sair.from_scalar %arg1 : !sair.value<(), memref<?x?x?xf32>>
    // CHECK: %[[INIT:.*]] = sair.copy
    %2 = sair.copy[d0:%1, d1:%1, d2:%1] %0 : !sair.value<d0:range x d1:range x d2:range, f32>
    // CHECK-NOT: sair.fby
    %3 = sair.fby[d0:%1, d1:%1, d2:%1] %2(d2, d1, d0) then[d3:%1] %3(d0, d1, d2, d3) : !sair.value<d0:range x d1:range x d2:range x d3:range, f32>
    // CHECK: sair.to_memref[{{.*}}] %{{.*}} memref[{{.*}}] %[[INIT]](d1, d3, d2)
    sair.to_memref[d0:%1] %4 memref[d1:%1, d2:%1, d3:%1] %3(d2, d3, d1, d0) {
      buffer_name = "bufferA"
    }  : #sair.shape<d0:range x d1:range x d2:range x d3:range>, memref<?x?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @remove_useless_dims_fby
func @remove_useless_dims_fby(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[R:.*]] = sair.static_range
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, f32>
    %3 = sair.copy[d0:%1, d1:%1] %0 : !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[FBY:.*]] = sair.fby[d0:%[[R]], d1:%[[R]]] %{{.*}}(d1)
    // CHECK:                   then[d2:%[[R]]] %{{.*}}(d0, d2)
    // CHECK:                   {test.foo = "bar"}
    // CHECK:                   !sair.value<d0:range x d1:range x d2:range, f32>
    %4 = sair.fby[d0:%1, d1:%1, d2:%1] %2(d2) then[d3:%1, d4:%1] %3(d0, d4) {test.foo="bar"} : !sair.value<d0:range x d1:range x d2:range x d3:range x d4:range, f32>
    // CHECK: sair.proj_last of[d0:%[[R]], d1:%[[R]], d2:%[[R]]] %[[FBY]](d2, d1, d0)
    %5 = sair.proj_last of[d0:%1, d1:%1, d2:%1, d3:%1, d4:%1] %4(d4, d3, d2, d1, d0)
      : #sair.shape<d0:range x d1:range x d2:range x d3:range x d4:range>, f32
    sair.exit %5 : f32
  } : f32
  return
}

// CHECK-LABEL: @remove_useless_dims_proj
func @remove_useless_dims_proj(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[R:.*]] = sair.static_range
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1, d1:%1, d2:%1] %0 : !sair.value<d0:range x d1:range x d2:range, f32>
    // CHECK: %[[PROJ:.*]] = sair.proj_any[d0:%[[R]], d1:%[[R]]]
    // CHECK:                           of[d2:%[[R]]] %{{.*}}(d0, d2, d1)
    // CHECK:                #sair.shape<d0:range x d1:range x d2:range>, f32
    %3 = sair.proj_any[d0:%1, d1:%1, d2:%1] of[d3:%1, d4:%1] %2(d1, d4, d2)
      : #sair.shape<d0:range x d1:range x d2:range x d3:range x d4:range>, f32
    // CHECK: sair.proj_last of[d0:%[[R]], d1:%[[R]]] %[[PROJ]](d0, d1)
    %4 = sair.proj_last of[d0:%1, d1:%1, d2:%1] %3(d0, d1, d2)
      : #sair.shape<d0:range x d1:range x d2:range>, f32
    sair.exit %4 : f32
  } : f32
  return
}

// CHECK-LABEL: @remove_useless_dims_proj_dependent
func @remove_useless_dims_proj_dependent(%arg0: f32, %arg1: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[R:.*]] = sair.static_range
    %1 = sair.static_range 8 : !sair.range
    %4 = sair.from_scalar %arg1 : !sair.value<(), index>
    // CHECK: %[[DR:.*]] = sair.dyn_range
    %5 = sair.dyn_range %4 : !sair.range
    %6 = sair.copy[d0:%5] %4 : !sair.value<d0:range, index>
    // CHECK: %[[DRD:.*]] = sair.dyn_range
    %7 = sair.dyn_range[d0:%5] %6(d0) : !sair.range<d0:range>
    %8 = sair.copy[d0:%5, d1:%7, d2:%1] %0 : !sair.value<d0:range x d1:range(d0) x d2:range, f32>
    // CHECK: %[[PROJ:.*]] = sair.proj_any[d0:%[[DR]], d1:%[[R]]]
    // CHECK:                           of[d2:%[[DRD]]] %{{.*}}(d0, d2, d1)
    // CHECK:                #sair.shape<d0:range x d1:range x d2:range(d0)>, f32
    %9 = sair.proj_any[d0:%1, d1:%5, d2:%1] of[d3:%1, d4:%7] %8(d1, d4, d2)
      : #sair.shape<d0:range x d1:range x d2:range x d3:range x d4:range(d1)>, f32
    // CHECK: sair.proj_last of[d0:%[[DR]], d1:%[[R]]] %[[PROJ]](d0, d1)
    %10 = sair.proj_last of[d0:%1, d1:%5, d2:%1] %9(d0, d1, d2)
      : #sair.shape<d0:range x d1:range x d2:range>, f32
    sair.exit %10 : f32
  } : f32
  return
}

// CHECK-LABEL: @mappings
func @mappings(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.copy
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, f32>
    // CHECK: sair.copy[d0:%{{.*}}] %[[V0]](d0)
    %3 = sair.copy[d0:%1] %2(unstripe(stripe(d0, [4]), stripe(d0, [4, 1]), [4, 1]))
      : !sair.value<d0:range, f32>
    %4 = sair.proj_last of[d0:%1] %3(d0) : #sair.shape<d0:range>, f32
    sair.exit %4 : f32
  } : f32
  return
}
