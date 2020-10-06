// RUN: sair-opt -sair-rematerialize %s -verify-diagnostics -split-input-file

func @dependent_dimension(%arg0: f32, %arg1: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.from_scalar %arg1 : !sair.value<(), index>
    %2 = sair.static_range 8 : !sair.range
    %3 = sair.copy[d0:%2] %1 : !sair.value<d0:range, index>
    %4 = sair.range[d0:%2] %3(d0) : !sair.range<d0:range>

    // expected-error @+1 {{rematerialization is not supported for dependent dimensions}}
    %5 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>},
                   {name = "B", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>
    %6 = sair.copy[d0:%2, d1:%4] %5 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>},
                   {name = "B", iter = #sair.iter<d1>}]
    } : !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  return
}

// -----

func @wrong_order(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-note @+1 {{to be used here}}
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>
    // expected-error @+1 {{range value definition would not precede its use after rematerialization}}
    %1 = sair.static_range 8 : !sair.range
    %3 = sair.copy[d0:%1] %2 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

