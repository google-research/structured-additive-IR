// RUN: sair-opt -split-input-file -verify-diagnostics -sair-materialize-buffers %s

func.func @partial_layout(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    %1 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    // expected-error @+1 {{partial layouts are not yet supported}}
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}],
        storage = [{name = "bufferA", space = "memory",
                    layout = #sair.named_mapping<[d0:"loopA"] -> (d0, none)>}]
      }]
    } : !sair.value<d0:static_range<8>, f32>
    %3 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "loopB", iter = #sair.mapping_expr<d0>}],
        storage = [{name = "bufferA", space = "memory",
                    layout = #sair.named_mapping<[d0:"loopB"] -> (none, d0)>}]
      }]
    } : !sair.value<d0:static_range<8>, f32>
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @missing_memory_space(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    // expected-error @+1 {{missing memory space}}
    %1 = sair.copy %0 {
      instances = [{}]
    } : !sair.value<(), f32>
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @missing_layout(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    // expected-error @+1 {{missing layout}}
    %1 = sair.copy %0 {
      instances = [{
        loop_nest = [],
        storage = [{space = "memory", name = "A"}]
      }]
    } : !sair.value<(), f32>
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @copies(%arg0: f32) {
  sair.program {
    // expected-error @+1 {{operations must have exactly one instance when materializing buffers}}
    sair.from_scalar %arg0 {
      copies = [[{sequence = 0}]]
    } : !sair.value<(), f32>
    sair.exit { instances = [{}] }
  }
  return
}
