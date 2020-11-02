// RUN: sair-opt %s -canonicalize | FileCheck %s

func @use(%arg0: f32) {
  return
}

// CHECK-LABEL: @deduplicate_map_input
func @deduplicate_map_input(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.map %[[V0]] attributes {memory_space = []} {
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
func @remove_cyclic_fby(%arg0: f32, %arg1: memref<?x?x?x?xf32>) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[INIT:.*]] = sair.copy
    %2 = sair.copy[d0:%1, d1:%1, d2:%1] %0 : !sair.value<d0:range x d1:range x d2:range, f32>
    // CHECK-NOT: sair.fby
    %3 = sair.fby[d0:%1, d1:%1, d2:%1] %2(d2, d1, d0) then[d3:%1] %3(d0, d1, d2, d3) : !sair.value<d0:range x d1:range x d2:range x d3:range, f32>
    // CHECK: sair.to_memref[{{.*}}] %[[INIT]](d1, d3, d0), %{{.*}}
    sair.to_memref[d0:%1, d1:%1, d2:%1, d3:%1] %3(d0, d3, d1, d2), %arg1 : memref<?x?x?x?xf32>
    sair.exit
  }
  return
}
