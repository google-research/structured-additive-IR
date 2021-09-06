// RUN: sair-opt -sair-assign-default-storage -split-input-file -verify-diagnostics %s

func @expected_loop_nest(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{expected a loop-nest attribute}}
    %1 = sair.copy %0 : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @index_to_memory(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.dyn_range %0 : !sair.dyn_range
    // expected-error @+1 {{cannot generate default storage for multi-dimensional index values}}
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, index>
    %3 = sair.copy[d0:%1] %2(d0) {
      instances = [{
        loop_nest = [{name = "loopB", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, index>
    sair.exit
  }
  return
}

// -----

func @non_rectangular_shape(%arg0: f32, %arg1: index) {
  // expected-error @+1 {{unable to generate storage attributes}}
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.from_scalar %arg1 : !sair.value<(), index>
    %2 = sair.dyn_range %1 : !sair.dyn_range
    %3 = sair.dyn_range[d0:%2] %1 : !sair.dyn_range<d0:dyn_range>
    // expected-error @+1 {{in buffer "buffer_0": layout depends on loops it cannot be nested in}}
    %4 = sair.copy[d0:%2, d1:%3] %0 {
      instances = [{
        loop_nest = [
          {name = "loopA", iter = #sair.mapping_expr<d0>},
          {name = "loopB", iter = #sair.mapping_expr<d1>}
        ]
      }]
    } : !sair.value<d0:dyn_range x d1:dyn_range(d0), f32>
    %5 = sair.copy[d0:%2, d1:%3] %4(d0, d1) {
      instances = [{
        loop_nest = [
          {name = "loopC", iter = #sair.mapping_expr<d0>},
          {name = "loopD", iter = #sair.mapping_expr<d1>}
        ]
      }]
    } : !sair.value<d0:dyn_range x d1:dyn_range(d0), f32>
    sair.exit
  }
  return
}

// -----

func @incomplete_loop_nest(%arg0: memref<4xf32>, %arg1: index) {
  %c = constant 42.0 : f32
  sair.program {
    %n = sair.from_scalar %arg1 : !sair.value<(), index>
    %r = sair.dyn_range %n : !sair.dyn_range
    %mem = sair.from_scalar %arg0 : !sair.value<(), memref<4xf32>>
    %0 = sair.from_scalar %c : !sair.value<(), f32>
    // expected-error@below {{expected a loop-nest attribute}}
    %1 = sair.copy[d0:%r] %0 : !sair.value<d0:dyn_range, f32>
    sair.to_memref %mem memref[d0:%r] %1(d0) { buffer_name = "buffer" }
      : #sair.shape<d0:dyn_range>, memref<4xf32>
    sair.exit
  }
  return
}

// -----

func @increase_external_buffer_rank(%arg0: memref<f32>, %arg1: index) {
  sair.program {
    %n = sair.from_scalar %arg1 : !sair.value<(), index>
    %r = sair.dyn_range %n : !sair.dyn_range
    %mem = sair.from_scalar %arg0 : !sair.value<(), memref<f32>>
    %0 = sair.from_memref %mem memref { buffer_name = "buffer" }
      : #sair.shape<()>, memref<f32>
    %1 = sair.copy[d0:%r] %0 {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    // expected-error@below {{specifying value layout would require to increase the rank of an external buffer}}
    %2 = sair.copy[d0:%r] %1(d0) {
      instances = [{
        loop_nest = [{name = "B", iter = #sair.mapping_expr<d0>}],
        storage = [{space = "memory", name = "buffer"}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    %3 = sair.copy[d0:%r] %2(d0) {
      instances = [{
        loop_nest = [{name = "C", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    sair.exit
  }
  return
}

