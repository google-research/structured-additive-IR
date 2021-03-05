// RUN: sair-opt -split-input-file -sair-materialize-memrefs -verify-diagnostics %s

func @dependent_dimensions() {
  %c0 = constant 4 : index
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %c0 : !sair.value<(), index>
    %2 = sair.dyn_range[d0:%0] %1 : !sair.range<d0:range>
    // expected-error @+1 {{can only materialize hyper-rectangular Sair values}}
    %3 = sair.map[d0:%0, d1:%2] attributes {memory_space=[1]} {
      ^bb0(%arg0: index, %arg1: index):
        %4 = constant 1.0 : f32
        sair.return %4 : f32
    } : #sair.shape<d0:range x d1:range(d0)>, () -> f32
    sair.exit
  }
  return
}

// -----

func @non_zero_based_range(%arg0: index, %arg1: index) {
  %c1 = constant 1 : index
  %c4 = constant 4 : index
  sair.program {
    %0 = sair.from_scalar %c1 : !sair.value<(), index>
    %1 = sair.from_scalar %c4 : !sair.value<(), index>
    %2 = sair.dyn_range %0, %1 : !sair.range
    // expected-error @+1 {{only 0-based ranges are supported for memrefs}}
    %3 = sair.map[d0:%2] attributes {memory_space=[1]} {
      ^bb0(%arg2: index):
        %4 = constant 1.0 : f32
        sair.return %4 : f32
    } : #sair.shape<d0:range>, () -> f32
    sair.exit
  }
  return
}

// -----

func @invalid_consumer() {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // expected-note @+1 {{while trying to materialize a value produced here}}
    %1 = sair.map[d0:%0] attributes {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>}
      ],
      storage = [{
        name = "bufferA", space = "memory",
        layout = #sair.named_mapping<[d0:"loopA"] -> (d0)>
      }]
    } {
      ^bb0(%arg0: index):
        %2 = constant 1.0 : f32
        sair.return %2 : f32
    } : #sair.shape<d0:range>, () -> f32
    // expected-error @+1 {{can only materialize operands of sair.map operations}}
    %3 = sair.copy[d0:%0] %1(d0) {memory_space=[1]} : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @unsupported_from_memref(%arg0: memref<f32>) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), memref<f32>>
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{operation not supported by memref materialization}}
    %2 = sair.from_memref[d0:%1] %0 memref : #sair.shape<d0:range>, memref<f32>
    sair.exit
  }
  return
}
