// RUN: sair-opt %s | FileCheck %s

// Check that the result of from_memref has the inferred storage that places
// it in memory. It is sufficient that we don't fail here due to mismatch in
// map_reduce.
// CHECK-LABEL: @from_memref_in_memory
func @from_memref_in_memory(%arg0: memref<?xf32>) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), memref<?xf32>>
    %1 = sair.static_range 42 : !sair.range
    %2 = sair.from_memref %0 memref[d0:%1] { buffer_name = "buffer" } : #sair.shape<d0:range>, memref<?xf32>
    sair.map_reduce[d0:%1] %2(d0) reduce %2(d0) attributes {
      loop_nest = [
        { name = "loop0", iter = #sair.mapping_expr<d0> }
      ],
      // Make sure the result (and the init that is expected to have the same
      // storage) are stored in memory.
      storage = [
        { name = "buffer", space = "memory",
          layout = #sair.named_mapping<[d0:"loop0"] -> (d0)> }
      ]
    } {
    ^bb0(%arg1: index, %arg2: f32, %arg3: f32):
      sair.return %arg2 : f32
    } : #sair.shape<d0:range>, (f32) -> (f32)
    sair.exit
  }
  return
}

