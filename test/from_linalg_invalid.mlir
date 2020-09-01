// RUN: sair-opt --convert-linalg-to-sair -split-input-file -verify-diagnostics %s

#reductions_trait = {
  args_in = 1,
  args_out = 1,
  indexing_maps = [
    affine_map<(i, j, k, l, m) -> (i, j, k, l, m)>,
    // Accessing a reduction dimension in the output.
    affine_map<(i, j, k, l, m) -> (i, k, l)>
  ],
  iterator_types = ["parallel", "reduction", "parallel", "reduction",
                    "parallel"]
}

func @reductions(%arg0: memref<2x3x4x5x6xf32>, %arg1: memref<2x4x6xf32>) {
  // expected-error @+1 {{Linalg op is not compatible with Sair}}
  linalg.generic #reductions_trait %arg0, %arg1 {
  ^bb0(%a0: f32, %a1: f32):
    %0 = addf %a1, %a0 : f32
    linalg.yield %0 : f32
  } : memref<2x3x4x5x6xf32>, memref<2x4x6xf32>

  return
}
