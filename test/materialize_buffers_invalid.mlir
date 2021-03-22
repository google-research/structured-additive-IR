// RUN: sair-opt -split-input-file -verify-diagnostics -sair-materialize-buffers %s

func @partial_layout(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{partial layouts are not yet supported}}
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}],
      storage = [{name = "bufferA", space = "memory",
                  layout = #sair.named_mapping<[d0:"loopA"] -> (d0, none)>}]
    } : !sair.value<d0:range, f32>
    %3 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "loopB", iter = #sair.mapping_expr<d0>}],
      storage = [{name = "bufferA", space = "memory",
                  layout = #sair.named_mapping<[d0:"loopB"] -> (none, d0)>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @missing_information(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{missing storage information}}
    %1 = sair.copy %0 : !sair.value<(), f32>
    sair.exit
  }
  return
}
