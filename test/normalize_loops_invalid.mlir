// RUN: sair-opt -split-input-file -sair-normalize-loops -verify-diagnostics %s

func @must_be_fully_specified(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{loop normalization called on a partially specified loop nest}}
    %2 = sair.copy[d0:%0] %1 {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        {name = "loopB", iter = #sair.mapping_expr<none>}
      ]
    } : !sair.value<d0:range, f32>
    %3 = sair.copy[d0:%0, d1:%0] %2(d0) {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        {name = "loopB", iter = #sair.mapping_expr<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}
