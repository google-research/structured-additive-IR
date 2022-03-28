// RUN: sair-opt -split-input-file -sair-normalize-loops -verify-diagnostics %s

// CHECK-LABEL: @from_memref
func.func @from_memref(%arg0: index) {
  sair.program {
    %size = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), index>
    %0 = sair.static_range { instances = [{}] } : !sair.static_range<4>
    %1 = sair.dyn_range %size { instances = [{}] } : !sair.dyn_range
    %memref = sair.alloc[d0:%0] %size {
      instances = [{loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]}]
    } : !sair.value<d0:static_range<4>, memref<?xf32>>
    // expected-error @+1 {{sair.from_memref and sair.to_memref must be eliminated before loop normalization}}
    %2 = sair.from_memref[d0:%0] %memref(d0) memref[d1:%1] {
      instances = [{}],
      buffer_name = "bufferA"
    }  : #sair.shape<d0:static_range<4> x d1:dyn_range>, memref<?xf32>
    sair.exit { instances = [{}] }
  }
  func.return
}

// -----

func.func @memrefs_must_be_introduced(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    %1 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    // expected-error @+1 {{operation with an incomplete iteration space}}
    %2 = sair.proj_last of[d0:%1] %0 { instances = [{}] } : #sair.shape<d0:static_range<8>>, f32
    sair.exit { instances = [{}] }
  }
  func.return
}

// -----

func.func @copies(%arg0: f32) {
  sair.program {
    // expected-error @+1 {{operations must have exactly one instance when normalizing loop nests}}
    sair.from_scalar %arg0 {
      copies = [[{sequence = 0}]]
    } : !sair.value<(), f32>
    sair.exit { instances = [{}] }
  }
  func.return
}
