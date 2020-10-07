// RUN: sair-opt -sair-rematerialize %s -verify-diagnostics -split-input-file

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

