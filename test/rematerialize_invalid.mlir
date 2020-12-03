// RUN: sair-opt -split-input-file -sair-rematerialize -verify-diagnostics %s

func @partial_rematerialization(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<stripe(d0, 2)>},
        {name = "B", iter = #sair.mapping_expr<stripe(d0, 1 size 2)>}
      ]
    } : !sair.value<d0:range, f32>
    // expected-error @+1 {{rematerialization only supports plain loops}}
    %3 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<none>}]
    } : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----
