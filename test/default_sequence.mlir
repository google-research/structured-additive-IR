// RUN: sair-opt -sair-assign-default-sequence %s | FileCheck %s

// CHECK-LABEL: @empty
// Shouldn't fail here.
func @empty() {
  sair.program {
    sair.exit
  }
  return
}

// CHECK-LABEL: @simple_use_def
func @simple_use_def() {
  sair.program {
    // CHECK: sair.alloc
    // CHECK-SAME: sequence = 0
    %0 = sair.alloc : !sair.value<(), memref<f32>>
    // CHECK: sair.free
    // CHECK-SAME: sequence = 1
    sair.free %0 : !sair.value<(), memref<f32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @simple_use_def_chain
func @simple_use_def_chain(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %1 = sair.copy %0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %2 = sair.copy %1 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    %3 = sair.copy %2 : !sair.value<(), f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 3
    %4 = sair.map %3 {
    ^bb0(%arg1: f32):
      sair.return %arg1 : f32
    } : #sair.shape<()>, (f32) -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @use_def_graph
// Check that we deterministically prefer the original order of operations in
// absence of other criteria.
func @use_def_graph(%arg0: f32, %arg1: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // CHECK: %[[V2:.*]] = sair.copy %[[V0]]
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %0 : !sair.value<(), f32>
    // CHECK: %[[V3:.*]] = sair.copy %[[V1]]
    // CHECK-SAME: sequence = 1
    %3 = sair.copy %1 : !sair.value<(), f32>
    // CHECK: sair.copy %[[V3]]
    // CHECK-SAME: sequence = 2
    %4 = sair.copy %3 : !sair.value<(), f32>
    // CHECK: sair.copy %[[V2]]
    // CHECK-SAME: sequence = 3
    %5 = sair.copy %2 : !sair.value<(), f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 4
    sair.map %3, %4, %5 {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      sair.return %arg2, %arg3, %arg4 : f32, f32, f32
    } : #sair.shape<()>, (f32, f32, f32) -> (f32, f32, f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @use_def_graph_partial
// Check that pre-existent relative order is preserved.
func @use_def_graph_partial(%arg0: f32, %arg1: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // CHECK: %[[V2:.*]] = sair.copy %[[V0]]
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %0 : !sair.value<(), f32>
    // CHECK: %[[V3:.*]] = sair.copy %[[V1]]
    // CHECK-SAME: sequence = 2
    %3 = sair.copy %1 { sequence = 5 } : !sair.value<(), f32>
    // CHECK: sair.copy %[[V3]]
    // CHECK-SAME: sequence = 3
    %4 = sair.copy %3 : !sair.value<(), f32>
    // CHECK: sair.copy %[[V2]]
    // CHECK-SAME: sequence = 1
    %5 = sair.copy %2 { sequence = 2 } : !sair.value<(), f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 4
    sair.map %3, %4, %5 {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
      sair.return %arg2, %arg3, %arg4 : f32, f32, f32
    } : #sair.shape<()>, (f32, f32, f32) -> (f32, f32, f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @fby
// Check that the pass is not confused by fby self-dependency.
func @fby(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 42 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    %3 = sair.fby %2 then[d0:%0] %4(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 1
    %4 = sair.map[d0:%0] %3(d0) {
    ^bb0(%arg1: index, %arg2: f32):
      sair.return %arg2 : f32
    } : #sair.shape<d0:range>, (f32) -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @fby_many_compute
func @fby_many_compute(%arg0: f32) -> f32 {
  %out = sair.program {
    %0 = sair.static_range 42 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    %3 = sair.fby %2 then[d0:%0] %7(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %4 = sair.copy[d0:%0] %3(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    %5 = sair.copy[d0:%0] %4(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 3
    %6 = sair.copy[d0:%0] %4(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 4
    %7 = sair.copy[d0:%0] %5(d0) : !sair.value<d0:range, f32>
    %8 = sair.proj_last of[d0:%0] %3(d0) : #sair.shape<d0:range>, f32
    sair.exit %8 : f32
  } : f32
  return %out : f32
}

// CHECK-LABEL: @fby_two_cycles
func @fby_two_cycles(%arg0: f32) -> f32 {
  %out = sair.program {
    %0 = sair.static_range 42 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    %3 = sair.fby %2 then[d0:%0] %8(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %4 = sair.copy[d0:%0] %3(d0) : !sair.value<d0:range, f32>
    %5 = sair.fby[d0:%0] %4(d0) then[d1:%0] %6(d0, d1) : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    %6 = sair.copy[d0:%0, d1:%0] %5(d0, d1) : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 3
    %7 = sair.copy[d0:%0, d1:%0] %6(d0, d1) : !sair.value<d0:range x d1:range, f32>
    %8 = sair.proj_any[d0:%0] of[d1:%0] %7(d0, d1) : #sair.shape<d0:range x d1:range>, f32
    %9 = sair.proj_last of[d0:%0, d1:%0] %5(d0, d1) : #sair.shape<d0:range x d1:range>, f32
    sair.exit %9 : f32
  } : f32
  return %out : f32
}

// CHECK-LABEL: @fby_then_different_source
func @fby_then_different_source(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 42 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %3 = sair.copy[d0:%0] %1 : !sair.value<d0:range, f32>
    %4 = sair.fby %2 then[d0:%0] %3(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 2
    sair.map[d0:%0] %4(d0) {
    ^bb0(%arg1: index, %arg2: f32):
      sair.return %arg2 : f32
    } : #sair.shape<d0:range>, (f32) -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @sequence_domain
func @sequence_domain(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.range
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, index>

    %3 = sair.dyn_range[d0:%1] %2(d0) : !sair.range<d0:range>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    sair.copy[d0:%1, d1:%3] %0 : !sair.value<d0:range x d1:range(d0), index>

    sair.exit
  }
  return
}

// CHECK-LABEL: @sequence_implicit_domain
func @sequence_implicit_domain(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.range
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, index>

    %3 = sair.dyn_range[d0:%1] %2(d0) : !sair.range<d0:range>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %4 = sair.copy[d0:%1, d1:%3] %0 : !sair.value<d0:range x d1:range(d0), index>

    %5 = sair.proj_any[d0:%1] of[d1:%3] %4(d0, d1) : #sair.shape<d0:range x d1:range(d0)>, index
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    sair.copy[d0:%1] %5(d0) : !sair.value<d0:range, index>

    sair.exit
  }
  return
}

// CHECK-LABEL: @sequence_implicit_domain_partial
func @sequence_implicit_domain_partial(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.range
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %2 = sair.copy[d0:%1] %0 { sequence = 2 } : !sair.value<d0:range, index>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %3 = sair.copy[d0:%1] %0 { sequence = 1 }: !sair.value<d0:range, index>
    %4 = sair.dyn_range[d0:%1] %2(d0) : !sair.range<d0:range>
    %5 = sair.dyn_range[d0:%1] %3(d0) : !sair.range<d0:range>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    %6 = sair.copy[d0:%1, d1:%4, d2:%5] %0 : !sair.value<d0:range x d1:range(d0) x d2:range(d0), index>
    %7 = sair.proj_any[d0:%1] of[d1:%4, d2:%5] %6(d0, d1, d2) : #sair.shape<d0:range x d1:range(d0) x d2:range(d0)>, index
    sair.copy[d0:%1] %7(d0) : !sair.value<d0:range, index>

    sair.exit
  }
  return
}
