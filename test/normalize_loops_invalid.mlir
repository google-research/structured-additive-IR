// RUN: sair-opt -split-input-file -sair-normalize-loops -verify-diagnostics %s

// CHECK-LABEL: @from_memref
func @from_memref(%arg0: index) {
  sair.program {
    %size = sair.from_scalar %arg0 : !sair.value<(), index>
    %0 = sair.static_range : !sair.static_range<4>
    %1 = sair.dyn_range %size : !sair.dyn_range
    %memref = sair.alloc[d0:%0] %size {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:static_range<4>, memref<?xf32>>
    // expected-error @+1 {{sair.from_memref and sair.to_memref must be eliminated before loop normalization}}
    %2 = sair.from_memref[d0:%0] %memref(d0) memref[d1:%1] {
      buffer_name = "bufferA"
    }  : #sair.shape<d0:static_range<4> x d1:dyn_range>, memref<?xf32>
    sair.exit
  }
  return
}

// -----

func @memrefs_must_be_introduced(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range : !sair.static_range<8>
    // expected-error @+1 {{operation with an incomplete iteration space}}
    %2 = sair.proj_last of[d0:%1] %0 : #sair.shape<d0:static_range<8>>, f32
    sair.exit
  }
  return
}

// -----

func private @foo(index, f32)

// Loop normalization will currently result in invalid IR because the loop
// bound computation will be inserted before the compute operation with the
// lowest sequence number, i.e. the third copy, and will be used by the
// operation preceding it. This will be resolved by relaxing use-def order
// later.
func @sequence_attr(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range : !sair.static_range<16>

    // expected-error @below {{dimension used before its definition}}
    // expected-note @below {{definition here}}
    sair.map[d0:%1] %0 attributes {
      loop_nest = [
        {name = "B", iter = #sair.mapping_expr<stripe(d0, [4])>},
        {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
      ],
      sequence = 2
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()

    sair.map[d0:%1] %0 attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
      sequence = 3
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()

    sair.map[d0:%1] %0 attributes {
      loop_nest = [
        {name = "B", iter = #sair.mapping_expr<stripe(d0, [4])>},
        {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
      ],
      sequence = 1
    } {
    ^bb0(%arg1: index, %arg2: f32):
      call @foo(%arg1, %arg2) : (index, f32) -> ()
      sair.return
    } : #sair.shape<d0:static_range<16>>, (f32) -> ()
    sair.exit
  }
  return
}
