// RUN: sair-opt %s | FileCheck %s

// Check that the result of from_memref has the inferred storage that places
// it in memory. It is sufficient that we don't fail here due to mismatch in
// map_reduce.
// CHECK-LABEL: @from_memref_in_memory
func @from_memref_in_memory(%arg0: memref<?xf32>, %arg1: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), memref<?xf32>>
    %1 = sair.dyn_range %sn : !sair.range
    %2 = sair.from_memref %0 memref[d0:%1] { buffer_name = "buffer" } : #sair.shape<d0:range>, memref<?xf32>
    %3 = sair.from_scalar %arg1 : !sair.value<(), f32>
    sair.map_reduce[d0:%1] %2(d0) reduce %3 attributes {
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
    ^bb0(%arg2: index, %arg3: f32, %arg4: f32):
      sair.return %arg3 : f32
    } : #sair.shape<d0:range>, (f32) -> (f32)
    sair.exit
  }
  return
}

