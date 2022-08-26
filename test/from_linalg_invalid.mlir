// RUN: sair-opt --convert-linalg-to-sair -split-input-file -verify-diagnostics %s

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
  func.return
}
