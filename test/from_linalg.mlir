// RUN: sair-opt --convert-linalg-to-sair %s | FileCheck %s

#pointwise_trait = {
  indexing_maps = [
    affine_map<(i, j, k) -> (i, k, j)>,
    affine_map<(i, j, k) -> (k, j, i)>
  ],
  iterator_types = ["parallel", "parallel", "parallel"]
}

// CHECK-LABEL: @pointwise
func @pointwise(%arg0: memref<1x2x3xf32>, %arg1: memref<2x3x1xf32>) {
  // CHECK: %[[s0:.*]] = sair.static_range 1 : !sair.range
  // CHECK: %[[s1:.*]] = sair.static_range 2 : !sair.range
  // CHECK: %[[s2:.*]] = sair.static_range 3 : !sair.range
  // CHECK: %[[v0:.*]] = sair.from_memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %{{.*}} : memref<1x2x3xf32> -> !sair.value<d0:range x d1:range x d2:range, f32>

  // CHECK: %[[s0:.*]] = sair.static_range 2 : !sair.range
  // CHECK: %[[s1:.*]] = sair.static_range 3 : !sair.range
  // CHECK: %[[s2:.*]] = sair.static_range 1 : !sair.range
  // CHECK: %[[v1:.*]] = sair.from_memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %{{.*}} : memref<2x3x1xf32> -> !sair.value<d0:range x d1:range x d2:range, f32>

  // CHECK: %[[d0:.*]] = sair.static_range 1 : !sair.range
  // CHECK: %[[d1:.*]] = sair.static_range 3 : !sair.range
  // CHECK: %[[d2:.*]] = sair.static_range 2 : !sair.range
  // CHECK: %[[v2:.*]] = sair.map[d0:%[[d0]], d1:%[[d1]], d2:%[[d2]]] %[[v0]](d0, d2, d1), %[[v1]](d2, d1, d0)
  linalg.generic #pointwise_trait
    ins(%arg0 : memref<1x2x3xf32>)
   outs(%arg1 : memref<2x3x1xf32>) {
  // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: f32, %{{.*}}: f32):
  ^bb(%a0: f32, %a1: f32):
    %1 = addf %a0, %a1 : f32
    // CHECK: sair.return %{{.*}} f32
    linalg.yield %1 : f32
  // CHECK: #sair.shape<d0:range x d1:range x d2:range>, (f32, f32) -> f32
  }

  // CHECK: sair.to_memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v2]](d2, d1, d0), %{{.*}} : memref<2x3x1xf32>
  return
}

// CHECK-LABEL: @dynamic
// CHECK: (%[[ARG0:.*]]: memref<?x2x?xf32>, %[[ARG1:.*]]: memref<?x3x?xf32>)
func @dynamic(%arg0: memref<?x2x?xf32>, %arg1: memref<?x3x?xf32>) {
  // CHECK: %[[DIM0_0:.*]] = dim %[[ARG0]], %c0
  // CHECK: %[[DIM0_2:.*]] = dim %[[ARG0]], %c2
  // CHECK: %[[DIM1_0:.*]] = dim %[[ARG1]], %c0
  // CHECK: %[[DIM1_2:.*]] = dim %[[ARG1]], %c2
  // CHECK: %[[DIM2_0:.*]] = dim %[[ARG0]], %c0
  // CHECK: %[[DIM2_2:.*]] = dim %[[ARG0]], %c2

  // CHECK: sair.program

  // CHECK: %[[DIM0_0_VAL:.*]] = sair.from_scalar %[[DIM0_0]] : !sair.value<(), index>
  // CHECK: %[[s0:.*]] = sair.range %[[DIM0_0_VAL]]
  // CHECK: %[[s1:.*]] = sair.static_range 2
  // CHECK: %[[DIM0_2_VAL:.*]] = sair.from_scalar %[[DIM0_2]] : !sair.value<(), index>
  // CHECK: %[[s2:.*]] = sair.range %[[DIM0_2_VAL]]
  // CHECK: %[[v0:.*]] = sair.from_memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %{{.*}} : memref<?x2x?xf32> -> !sair.value<d0:range x d1:range x d2:range, f32>

  // CHECK: %[[DIM1_0_VAL:.*]] = sair.from_scalar %[[DIM1_0]] : !sair.value<(), index>
  // CHECK: %[[s0:.*]] = sair.range %[[DIM1_0_VAL]]
  // CHECK: %[[s1:.*]] = sair.static_range 3
  // CHECK: %[[DIM1_2_VAL:.*]] = sair.from_scalar %[[DIM1_2]] : !sair.value<(), index>
  // CHECK: %[[s2:.*]] = sair.range %[[DIM1_2_VAL]]
  // CHECK: %[[v1:.*]] = sair.from_memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %{{.*}} : memref<?x3x?xf32> -> !sair.value<d0:range x d1:range x d2:range, f32>

  // CHECK: %[[DIM2_0_VAL:.*]] = sair.from_scalar %[[DIM2_0]] : !sair.value<(), index>
  // CHECK: %[[d0:.*]] = sair.range %[[DIM2_0_VAL]]
  // CHECK: %[[DIM2_2_VAL:.*]] = sair.from_scalar %[[DIM2_2]] : !sair.value<(), index>
  // CHECK: %[[d1:.*]] = sair.range %[[DIM2_2_VAL]]
  // CHECK: %[[d2:.*]] = sair.static_range 2
  // CHECK: %[[v2:.*]] = sair.map[d0:%[[d0]], d1:%[[d1]], d2:%[[d2]]] %[[v0]](d0, d2, d1), %[[v1]](d2, d1, d0)
  linalg.generic #pointwise_trait
    ins(%arg0 : memref<?x2x?xf32>)
   outs(%arg1 : memref<?x3x?xf32>) {
  ^bb(%a0: f32, %a1: f32):
    %1 = addf %a0, %a1 : f32
    linalg.yield %1 : f32
  }

  // CHECK: sair.to_memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v2]](d2, d1, d0), %{{.*}} : memref<?x3x?xf32>
  return
}


#reductions_trait = {
  indexing_maps = [
    affine_map<(i, j, k, l, m) -> (i, j, k, l, m)>,
    affine_map<(i, j, k, l, m) -> (i, k, m)>
  ],
  iterator_types = ["parallel", "reduction", "parallel", "reduction",
                    "parallel"]
}

// CHECK-LABEL: @reductions
func @reductions(%arg0: memref<2x3x4x5x6xf32>, %arg1: memref<2x4x6xf32>) {
  // CHECK: %[[s0:.*]] = sair.static_range 2 : !sair.range
  // CHECK: %[[s1:.*]] = sair.static_range 3 : !sair.range
  // CHECK: %[[s2:.*]] = sair.static_range 4 : !sair.range
  // CHECK: %[[s3:.*]] = sair.static_range 5 : !sair.range
  // CHECK: %[[s4:.*]] = sair.static_range 6 : !sair.range
  // CHECK: %[[INPUT:.*]] = sair.from_memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]], d3:%[[s3]], d4:%[[s4]]] {{.*}} : memref<2x3x4x5x6xf32> -> !sair.value

  // CHECK: %[[r0:.*]] = sair.static_range 2 : !sair.range
  // CHECK: %[[r1:.*]] = sair.static_range 4 : !sair.range
  // CHECK: %[[r2:.*]] = sair.static_range 6 : !sair.range
  // CHECK: %[[INIT:.*]] = sair.from_memref[d0:%[[r0]], d1:%[[r1]], d2:%[[r2]]] {{.*}} : memref<2x4x6xf32> -> !sair.value

  // CHECK: %[[d0:.*]] = sair.static_range 2 : !sair.range
  // CHECK: %[[d1:.*]] = sair.static_range 4 : !sair.range
  // CHECK: %[[d2:.*]] = sair.static_range 6 : !sair.range
  // CHECK: %[[d3:.*]] = sair.static_range 3 : !sair.range
  // CHECK: %[[d4:.*]] = sair.static_range 5 : !sair.range
  // CHECK: %[[RES:.*]] = sair.map_reduce[d0:%[[d0]], d1:%[[d1]], d2:%[[d2]]] %[[INIT]](d0, d1, d2)
  // CHECK:                        reduce[d3:%[[d3]], d4:%[[d4]]] %[[INPUT]](d0, d3, d1, d4, d2) {
  linalg.generic #reductions_trait
    ins(%arg0 : memref<2x3x4x5x6xf32>)
   outs(%arg1 : memref<2x4x6xf32>) {
  // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %[[reduce:.*]]: f32, %[[arg:.*]]: f32):
  ^bb0(%a0: f32, %a1: f32):
    // Expecting the operands to be swapped because of block argument
    // reordering that places the partial reduction first to comply with Sair
    // conventions.
    // CHECK: %[[v0:.*]] = addf %[[reduce]], %[[arg]]
    %0 = addf %a1, %a0 : f32
    // CHECK: sair.return %[[v0]]
    linalg.yield %0 : f32
  // CHECK: } : #sair.shape<d0:range x d1:range x d2:range x d3:range x d4:range>, (f32) -> f32
  }

  // CHECK: sair.to_memref[d0:%[[r0]], d1:%[[r1]], d2:%[[r2]]] %[[RES]](d0, d1, d2), %{{.*}} : memref<2x4x6xf32>
  return
}

// CHECK-LABEL: @indexed
func @indexed(%arg0: memref<2x3x4x5x6xf64>, %arg1: memref<2x4x6xf64>) {
  linalg.indexed_generic #reductions_trait
    ins(%arg0 : memref<2x3x4x5x6xf64>)
   outs(%arg1 : memref<2x4x6xf64>) {
  // Sair puts reduction dimensions as innermost, so we expect the corresponding
  // indices (j and l) to be permuted accordingly in the block signature. Value
  // arguments are also permuted, similarly to the regular reduction.
  // CHECK: ^{{.*}}(%[[I:.*]]: index, %[[K:.*]]: index, %[[M:.*]]: index, %[[J:.*]]: index, %[[L:.*]]: index, %[[reduce:.*]]: f64, %[[arg:.*]]: f64):
  ^bb0(%i: index, %j: index, %k: index, %l: index, %m: index, %a0: f64, %a1: f64):
    // CHECK: addf %[[reduce]], %[[arg]]
    %0 = addf %a1, %a0 : f64
    // CHECK: affine.apply {{.*}}(%[[I]], %[[J]], %[[K]], %[[L]], %[[M]])
    %1 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 + d1 + d2 + d3 + d4)>(%i, %j, %k, %l, %m)
    %2 = index_cast %1 : index to i64
    %3 = sitofp %2 : i64 to f64
    %4 = addf %0, %3 : f64
    linalg.yield %4 : f64
  }
  return
}
