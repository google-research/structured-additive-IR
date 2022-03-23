// RUN: sair-opt %s -split-input-file

// These shouldn't fail because Sair doesn't enforce use-def order.

func.func @dimension_use_before_def(%arg0 : f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.copy[d0:%2] %0 : !sair.value<d0:static_range<8>, f32>
    %2 = sair.static_range : !sair.static_range<8>
    sair.exit
  }
  return
}

// -----

func.func @operand_use_before_def(%arg0 : f32) {
  sair.program {
    %0 = sair.copy %1 : !sair.value<(), f32>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

// It shouldn't be a problem to have a dynamic range for a rematerialized
// dimension to be defined after its used as long as there is no circular
// dependency introduced.
func.func @reordered_remat(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<none>}]
    } : !sair.value<(), f32>
    %1 = sair.static_range : !sair.static_range<8>
    %3 = sair.copy[d0:%1] %2 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:static_range<8>, f32>
    sair.exit
  }
  return
}

// -----

// Given explicit sequence attributes, we should take them into account in
// buffer use-after-defined verification. In particular, even if the definition
// of the buffer happens textually later, it is sequenced before in this case.
func.func @buffer_def_explicit_seq(%arg0: f32, %arg1: memref<f32>) {
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

// -----

// Implicit sequencing preserves textual order so we shouldn't complain about
// buffer being used before it is defined.
func.func @buffer_def_implicit_seq(%arg0: f32, %arg1: memref<f32>) {
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

// -----

// Explicit sequencing makes this code verify - buffer dimension computation
// (copy) is sequenced explicitly before the buffer is being first written into
// - despite the inverted order of operations in the block.
func.func @buffer_dimension_def_seq(%arg0: f32, %arg1: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>

    %2 = sair.static_range : !sair.static_range<8>
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
