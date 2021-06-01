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
    %0 = sair.static_range : !sair.static_range<42>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    %3 = sair.fby %2 then[d0:%0] %4(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 1
    %4 = sair.map[d0:%0] %3(d0) {
    ^bb0(%arg1: index, %arg2: f32):
      sair.return %arg2 : f32
    } : #sair.shape<d0:static_range<42>>, (f32) -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @fby_many_compute
func @fby_many_compute(%arg0: f32) -> f32 {
  %out = sair.program {
    %0 = sair.static_range : !sair.static_range<42>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    %3 = sair.fby %2 then[d0:%0] %7(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %4 = sair.copy[d0:%0] %3(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    %5 = sair.copy[d0:%0] %4(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 3
    %6 = sair.copy[d0:%0] %4(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 4
    %7 = sair.copy[d0:%0] %5(d0) : !sair.value<d0:static_range<42>, f32>
    %8 = sair.proj_last of[d0:%0] %3(d0) : #sair.shape<d0:static_range<42>>, f32
    sair.exit %8 : f32
  } : f32
  return %out : f32
}

// CHECK-LABEL: @fby_two_cycles
func @fby_two_cycles(%arg0: f32) -> f32 {
  %out = sair.program {
    %0 = sair.static_range : !sair.static_range<42>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    %3 = sair.fby %2 then[d0:%0] %8(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %4 = sair.copy[d0:%0] %3(d0) : !sair.value<d0:static_range<42>, f32>
    %5 = sair.fby[d0:%0] %4(d0) then[d1:%0] %6(d0, d1)
      : !sair.value<d0:static_range<42> x d1:static_range<42>, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    %6 = sair.copy[d0:%0, d1:%0] %5(d0, d1)
      : !sair.value<d0:static_range<42> x d1:static_range<42>, f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 3
    %7 = sair.copy[d0:%0, d1:%0] %6(d0, d1)
      : !sair.value<d0:static_range<42> x d1:static_range<42>, f32>
    %8 = sair.proj_any[d0:%0] of[d1:%0] %7(d0, d1)
      : #sair.shape<d0:static_range<42> x d1:static_range<42>>, f32
    %9 = sair.proj_last of[d0:%0, d1:%0] %5(d0, d1)
      : #sair.shape<d0:static_range<42> x d1:static_range<42>>, f32
    sair.exit %9 : f32
  } : f32
  return %out : f32
}

// CHECK-LABEL: @fby_then_different_source
func @fby_then_different_source(%arg0: f32) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<42>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %1 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %3 = sair.copy[d0:%0] %1 : !sair.value<d0:static_range<42>, f32>
    %4 = sair.fby %2 then[d0:%0] %3(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 2
    sair.map[d0:%0] %4(d0) {
    ^bb0(%arg1: index, %arg2: f32):
      sair.return %arg2 : f32
    } : #sair.shape<d0:static_range<42>>, (f32) -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @sequence_domain
func @sequence_domain(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.dyn_range
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:dyn_range, index>

    %3 = sair.dyn_range[d0:%1] %2(d0) : !sair.dyn_range<d0:dyn_range>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    sair.copy[d0:%1, d1:%3] %0 : !sair.value<d0:dyn_range x d1:dyn_range(d0), index>

    sair.exit
  }
  return
}

// CHECK-LABEL: @sequence_implicit_domain
func @sequence_implicit_domain(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.dyn_range
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:dyn_range, index>

    %3 = sair.dyn_range[d0:%1] %2(d0) : !sair.dyn_range<d0:dyn_range>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %4 = sair.copy[d0:%1, d1:%3] %0 : !sair.value<d0:dyn_range x d1:dyn_range(d0), index>

    %5 = sair.proj_any[d0:%1] of[d1:%3] %4(d0, d1) : #sair.shape<d0:dyn_range x d1:dyn_range(d0)>, index
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    sair.copy[d0:%1] %5(d0) : !sair.value<d0:dyn_range, index>

    sair.exit
  }
  return
}

// CHECK-LABEL: @sequence_implicit_domain_partial
func @sequence_implicit_domain_partial(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.dyn_range
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %2 = sair.copy[d0:%1] %0 { sequence = 2 } : !sair.value<d0:dyn_range, index>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %3 = sair.copy[d0:%1] %0 { sequence = 1 }: !sair.value<d0:dyn_range, index>
    %4 = sair.dyn_range[d0:%1] %2(d0) : !sair.dyn_range<d0:dyn_range>
    %5 = sair.dyn_range[d0:%1] %3(d0) : !sair.dyn_range<d0:dyn_range>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 2
    %6 = sair.copy[d0:%1, d1:%4, d2:%5] %0 : !sair.value<d0:dyn_range x d1:dyn_range(d0) x d2:dyn_range(d0), index>
    %7 = sair.proj_any[d0:%1] of[d1:%4, d2:%5] %6(d0, d1, d2) : #sair.shape<d0:dyn_range x d1:dyn_range(d0) x d2:dyn_range(d0)>, index
    sair.copy[d0:%1] %7(d0) : !sair.value<d0:dyn_range, index>

    sair.exit
  }
  return
}

// CHECK-LABEL: @reordered_remat
// It shouldn't be a problem to have a dynamic range for a rematerialized
// dimension to be defined after its used as long as there is no circular
// dependency introduced.
func @reordered_remat(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 0
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<none>}]
    } : !sair.value<(), f32>
    %1 = sair.static_range : !sair.static_range<8>
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %3 = sair.copy[d0:%1] %2 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:static_range<8>, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @buffer_def_explicit_seq
// Given explicit sequence attributes, we should take them into account in
// buffer use-after-defined verification. In particular, even if the definition
// of the buffer happens textually later, it is sequenced before in this case.
func @buffer_def_explicit_seq(%arg0: f32, %arg1: memref<f32>) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.copy %0 {
      loop_nest = [],
      storage = [{name = "bufferA", space = "memory", layout = #sair.named_mapping<[] -> ()>}],
      sequence = 2
    } : !sair.value<(), f32>

    %2 = sair.from_scalar %arg1 : !sair.value<(), memref<f32>>
    %copy = sair.copy %2 { sequence = 1 } : !sair.value<(), memref<f32>>
    %3 = sair.from_memref %copy memref {
      buffer_name = "bufferA"
    } : #sair.shape<()>, memref<f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @buffer_def_implicit_seq
// Implicit sequencing preserves textual order so we shouldn't complain about
// buffer being used before it is defined.
func @buffer_def_implicit_seq(%arg0: f32, %arg1: memref<f32>) {
  sair.program {
    %2 = sair.from_scalar %arg1 : !sair.value<(), memref<f32>>
    %copy = sair.copy %2 : !sair.value<(), memref<f32>>
    %3 = sair.from_memref %copy memref {
      buffer_name = "bufferA"
    } : #sair.shape<()>, memref<f32>

    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.copy %0 {
      loop_nest = [],
      storage = [{name = "bufferA", space = "memory", layout = #sair.named_mapping<[] -> ()>}]
    } : !sair.value<(), f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @buffer_dimension_def_seq
// Explicit sequencing makes this code verify - buffer dimension computation
// (copy) is sequenced explicitly before the buffer is being first written into
// - despite the inverted order of operations in the block.
func @buffer_dimension_def_seq(%arg0: f32, %arg1: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>

    %2 = sair.static_range : !sair.static_range<8>
    // expected-error @+1 {{buffer "bufferA" is used before one of its dimensions is defined}}
    %3 = sair.copy[d0:%2] %0 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}],
      storage = [{
        space = "memory", name = "bufferA",
        layout = #sair.named_mapping<[d0:"loopA"] -> (d0, none)>
      }],
      sequence = 2
    } : !sair.value<d0:static_range<8>, f32>

    %dim = sair.from_scalar %arg1 : !sair.value<(), index>
    %copy = sair.copy %dim { sequence = 1 } : !sair.value<(), index>
    // expected-note @+1 {{dimension defined here}}
    %4 = sair.dyn_range %copy : !sair.dyn_range
    %5 = sair.copy[d0:%4] %0 {
      loop_nest = [
        {name = "loopB", iter = #sair.mapping_expr<d0>}
      ],
      storage = [{
        space = "memory", name = "bufferA",
        layout = #sair.named_mapping<[d0:"loopB"] -> (none, d0)>
      }]
    } : !sair.value<d0:dyn_range, f32>
    sair.exit
  }
  return
}
