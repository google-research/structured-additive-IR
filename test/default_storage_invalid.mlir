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
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{cannot generate default storage for multi-dimensional index values}}
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, index>
    sair.exit
  }
  return
}

// -----

func @non_rectangular_shape(%arg0: f32, %arg1: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.from_scalar %arg1 : !sair.value<(), index>
    %2 = sair.static_range 8 : !sair.range
    %3 = sair.dyn_range[d0:%2] %1 : !sair.range<d0:range>
    // expected-error @+1 {{cannot generate default storage for non-rectangular shapes}}
    %4 = sair.copy[d0:%2, d1:%3] %0 {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        {name = "loopB", iter = #sair.mapping_expr<d1>}
      ]
    } : !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  return
}
