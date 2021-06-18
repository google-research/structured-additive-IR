// RUN: sair-opt --convert-linalg-to-sair %s | FileCheck %s
// RUN: sair-opt --convert-linalg-to-sair --mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC

#pointwise_trait = {
  indexing_maps = [
    affine_map<(i, j, k) -> (i, k, j)>,
    affine_map<(i, j, k) -> (k, j, i)>
  ],
  iterator_types = ["parallel", "parallel", "parallel"]
}

// CHECK-LABEL: @pointwise
func @pointwise(%arg0: memref<1x2x3xf32>, %arg1: memref<2x3x1xf32>) {
  // CHECK: %[[s0:.*]] = sair.static_range : !sair.static_range<1>
  // CHECK: %[[s1:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[s2:.*]] = sair.static_range : !sair.static_range<3>
  // CHECK: %[[v0_:.*]] = sair.from_memref %{{.*}} memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]]
  // CHECK: : #sair.shape<d0:static_range<1> x d1:static_range<2> x d2:static_range<3>>, memref<1x2x3xf32>
  // CHECK: %[[v0:.*]] = sair.copy[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v0_]](d0, d1, d2)
  // GENERIC: %[[s0:.*]] = "sair.static_range"() : () -> !sair.static_range<1>
  // GENERIC: %[[s1:.*]] = "sair.static_range"() : () -> !sair.static_range<2>
  // GENERIC: %[[s2:.*]] = "sair.static_range"() : () -> !sair.static_range<3>
  // GENERIC: "sair.from_memref"(%[[s0]], %[[s1]], %[[s2]],

  // CHECK: %[[s0:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[s1:.*]] = sair.static_range : !sair.static_range<3>
  // CHECK: %[[s2:.*]] = sair.static_range : !sair.static_range<1>
  // CHECK: %[[v1_:.*]] = sair.from_memref %{{.*}} memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]]
  // CHECK: : #sair.shape<d0:static_range<2> x d1:static_range<3> x d2:static_range<1>>, memref<2x3x1xf32>
  // CHECK: %[[v1:.*]] = sair.copy[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v1_]](d0, d1, d2)

  // CHECK: %[[d0:.*]] = sair.static_range : !sair.static_range<1>
  // CHECK: %[[d1:.*]] = sair.static_range : !sair.static_range<3>
  // CHECK: %[[d2:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[v2:.*]] = sair.map[d0:%[[d0]], d1:%[[d1]], d2:%[[d2]]] %[[v0]](d0, d2, d1), %[[v1]](d2, d1, d0)
  linalg.generic #pointwise_trait
    ins(%arg0 : memref<1x2x3xf32>)
   outs(%arg1 : memref<2x3x1xf32>) {
  // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: f32, %{{.*}}: f32):
  ^bb(%a0: f32, %a1: f32):
    %1 = addf %a0, %a1 : f32
    // CHECK: sair.return %{{.*}} f32
    linalg.yield %1 : f32
  // CHECK: #sair.shape<d0:static_range<1> x d1:static_range<3> x d2:static_range<2>>, (f32, f32) -> f32
  }

  // CHECK: sair.to_memref %{{.*}} memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v2]](d2, d1, d0)
  // CHECK:   : #sair.shape<d0:static_range<2> x d1:static_range<3> x d2:static_range<1>>, memref<2x3x1xf32>
  return
}

// CHECK-LABEL: @dynamic
// CHECK: (%[[ARG0:.*]]: memref<?x2x?xf32>, %[[ARG1:.*]]: memref<2x?x?xf32>)
func @dynamic(%arg0: memref<?x2x?xf32>, %arg1: memref<2x?x?xf32>) {
  // CHECK: %[[DIM0_0:.*]] = memref.dim %[[ARG0]], %c0
  // CHECK: %[[DIM0_2:.*]] = memref.dim %[[ARG0]], %c2
  // CHECK: %[[DIM1_1:.*]] = memref.dim %[[ARG1]], %c1
  // CHECK: %[[DIM1_2:.*]] = memref.dim %[[ARG1]], %c2
  // CHECK: %[[DIM2_0:.*]] = memref.dim %[[ARG0]], %c0
  // CHECK: %[[DIM2_2:.*]] = memref.dim %[[ARG0]], %c2

  // CHECK: sair.program

  // CHECK: %[[DIM0_0_VAL:.*]] = sair.from_scalar %[[DIM0_0]] : !sair.value<(), index>
  // CHECK: %[[s0:.*]] = sair.dyn_range %[[DIM0_0_VAL]]
  // CHECK: %[[s1:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[DIM0_2_VAL:.*]] = sair.from_scalar %[[DIM0_2]] : !sair.value<(), index>
  // CHECK: %[[s2:.*]] = sair.dyn_range %[[DIM0_2_VAL]]
  // CHECK: %[[v0_:.*]] = sair.from_memref %{{.*}} memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]]
  // CHECK:   : #sair.shape<d0:dyn_range x d1:static_range<2> x d2:dyn_range>, memref<?x2x?xf32>
  // CHECK: %[[v0:.*]] = sair.copy[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v0_]](d0, d1, d2)

  // CHECK: %[[s0:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[DIM1_1_VAL:.*]] = sair.from_scalar %[[DIM1_1]] : !sair.value<(), index>
  // CHECK: %[[s1:.*]] = sair.dyn_range %[[DIM1_1_VAL]]
  // CHECK: %[[DIM1_2_VAL:.*]] = sair.from_scalar %[[DIM1_2]] : !sair.value<(), index>
  // CHECK: %[[s2:.*]] = sair.dyn_range %[[DIM1_2_VAL]]
  // CHECK: %[[v1_:.*]] = sair.from_memref %{{.*}} memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]]
  // CHECK:   : #sair.shape<d0:static_range<2> x d1:dyn_range x d2:dyn_range>, memref<2x?x?xf32>
  // CHECK: %[[v1:.*]] = sair.copy[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v1_]](d0, d1, d2)

  // CHECK: %[[DIM2_0_VAL:.*]] = sair.from_scalar %[[DIM2_0]] : !sair.value<(), index>
  // CHECK: %[[d0:.*]] = sair.dyn_range %[[DIM2_0_VAL]]
  // CHECK: %[[DIM2_2_VAL:.*]] = sair.from_scalar %[[DIM2_2]] : !sair.value<(), index>
  // CHECK: %[[d1:.*]] = sair.dyn_range %[[DIM2_2_VAL]]
  // CHECK: %[[d2:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[v2:.*]] = sair.map[d0:%[[d0]], d1:%[[d1]], d2:%[[d2]]] %[[v0]](d0, d2, d1), %[[v1]](d2, d1, d0)
  linalg.generic #pointwise_trait
    ins(%arg0 : memref<?x2x?xf32>)
   outs(%arg1 : memref<2x?x?xf32>) {
  ^bb(%a0: f32, %a1: f32):
    %1 = addf %a0, %a1 : f32
    linalg.yield %1 : f32
  }

  // CHECK: sair.to_memref %{{.*}} memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]]] %[[v2]](d2, d1, d0)
  // CHECK:   : #sair.shape<d0:static_range<2> x d1:dyn_range x d2:dyn_range>, memref<2x?x?xf32>
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
  // CHECK: %[[s0:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[s1:.*]] = sair.static_range : !sair.static_range<3>
  // CHECK: %[[s2:.*]] = sair.static_range : !sair.static_range<4>
  // CHECK: %[[s3:.*]] = sair.static_range : !sair.static_range<5>
  // CHECK: %[[s4:.*]] = sair.static_range : !sair.static_range<6>
  // CHECK: %[[INPUT_:.*]] = sair.from_memref %{{.}} memref[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]], d3:%[[s3]], d4:%[[s4]]]
  // CHECK: %[[INPUT:.*]] = sair.copy[d0:%[[s0]], d1:%[[s1]], d2:%[[s2]], d3:%[[s3]], d4:%[[s4]]]  %[[INPUT_]](d0, d1, d2, d3, d4)

  // CHECK: %[[r0:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[r1:.*]] = sair.static_range : !sair.static_range<4>
  // CHECK: %[[r2:.*]] = sair.static_range : !sair.static_range<6>
  // CHECK: %[[INIT_:.*]] = sair.from_memref %{{.*}} memref[d0:%[[r0]], d1:%[[r1]], d2:%[[r2]]]
  // CHECK: %[[INIT:.*]] = sair.copy[d0:%[[r0]], d1:%[[r1]], d2:%[[r2]]] %[[INIT_]](d0, d1, d2)

  // CHECK: %[[d0:.*]] = sair.static_range : !sair.static_range<2>
  // CHECK: %[[d1:.*]] = sair.static_range : !sair.static_range<4>
  // CHECK: %[[d2:.*]] = sair.static_range : !sair.static_range<6>
  // CHECK: %[[d3:.*]] = sair.static_range : !sair.static_range<3>
  // CHECK: %[[d4:.*]] = sair.static_range : !sair.static_range<5>
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
  // CHECK: } : #sair.shape<d0:static_range<2> x d1:static_range<4> x
  // CHECK:       d2:static_range<6> x d3:static_range<3> x d4:static_range<5>>, (f32) -> f32
  }

  // CHECK: sair.to_memref %{{.*}}[d0:%[[r0]], d1:%[[r1]], d2:%[[r2]]] %[[RES]](d0, d1, d2)
  // CHECK:   : #sair.shape<d0:static_range<2> x d1:static_range<4> x d2:static_range<6>>, memref<2x4x6xf32>
  return
}

