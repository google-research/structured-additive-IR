// RUN: sair-opt --convert-linalg-to-sair -split-input-file -verify-diagnostics %s

#reductions_trait = {
  indexing_maps = [
    affine_map<(i, j, k, l, m) -> (i, j, k, l, m)>,
    // Accessing a reduction dimension in the output.
    affine_map<(i, j, k, l, m) -> (i, k, l)>
  ],
  iterator_types = ["parallel", "reduction", "parallel", "reduction",
                    "parallel"]
}

func.func @reductions(%arg0: memref<2x3x4x5x6xf32>, %arg1: memref<2x4x6xf32>) {
  // expected-error @+1 {{unexpected output tensor expression in indexing map #0 a.k.a 'd3' is function of reduction iterator 'd3'}}
  linalg.generic #reductions_trait
    ins(%arg0 : memref<2x3x4x5x6xf32>)
   outs(%arg1 : memref<2x4x6xf32>) {
  ^bb0(%a0: f32, %a1: f32):
    %0 = arith.addf %a1, %a0 : f32
    linalg.yield %0 : f32
  }

  return
}

// -----

#pointwise_trait = {
  indexing_maps = [
    affine_map<(i) -> (i)>,
    affine_map<(i) -> (i)>
  ],
  iterator_types = ["parallel"]
}

func.func @shape_mismatch(%arg0: memref<?xf32>, %arg1: memref<2xf32>) {
  // expected-error @+1 {{Linalg op is not compatible with Sair}}
  linalg.generic #pointwise_trait
    ins(%arg0 : memref<?xf32>)
   outs(%arg1 : memref<2xf32>) {
  ^bb(%a0: f32, %a1: f32):
    %1 = arith.addf %a0, %a1 : f32
    linalg.yield %1 : f32
  }
  return
}
