// RUN: sair-opt %s -sair-lower-to-map --mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: @copy
func @copy(%arg0 : memref<?x?xf32>) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[V0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.dyn_range
    %1 = sair.from_scalar %arg0 : !sair.value<(), memref<?x?xf32>>
    // CHECK: %[[V1:.*]] = sair.from_memref
    %2 = sair.from_memref %1 memref[d0:%0, d1:%0] {
      buffer_name = "bufferA"
    } : #sair.shape<d0:dyn_range x d1:dyn_range>, memref<?x?xf32>
    // CHECK: sair.map[d0:%[[V0]], d1:%[[V0]]] %[[V1]](d1, d0)
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %[[ARG0:.*]]: f32):
    // CHECK: sair.return %[[ARG0]] : f32
    %3 = sair.copy[d0:%0, d1:%0] %2(d1, d0) {
      instances = [{expansion = "copy"}]
    // CHECK: } : #sair.shape<d0:dyn_range x d1:dyn_range>, (f32) -> f32
    }  : !sair.value<d0:dyn_range x d1:dyn_range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @alloc
func @alloc(%arg0: index) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.dyn_range
    // CHECK: %[[D1:.*]] = sair.dyn_range
    %1 = sair.dyn_range %sn : !sair.dyn_range
    %idx = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[SZ0:.*]] = sair.map
    %2 = sair.copy[d0:%0] %idx {
      instances = [{expansion = "copy"}]
    } : !sair.value<d0:dyn_range, index>
    // CHECK: %[[SZ1:.*]] = sair.map
    %3 = sair.copy[d0:%1] %idx {
      instances = [{expansion = "copy"}]
    } : !sair.value<d0:dyn_range, index>
    // CHECK: sair.map[d0:%[[D0]], d1:%[[D1]]] %[[SZ0]](d0), %[[SZ1]](d1)
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index):
    // CHECK:   %[[ALLOC:.*]] = memref.alloc(%[[ARG2]], %[[ARG3]]) : memref<?x?xf32>
    // CHECK:   sair.return %[[ALLOC]]
    // CHECK: } : #sair.shape<d0:dyn_range x d1:dyn_range>, (index, index) -> memref<?x?xf32>
    sair.alloc[d0:%0, d1:%1] %2(d0), %3(d1) {
      instances = [{expansion = "alloc"}]
    } : !sair.value<d0:dyn_range x d1:dyn_range, memref<?x?xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @sair_free
func @sair_free(%arg0: index) {
  sair.program {
    %idx = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %idx : !sair.dyn_range
    // CHECK: %[[D1:.*]] = sair.dyn_range
    %1 = sair.dyn_range %idx : !sair.dyn_range
    // CHECK: sair.map
    %2 = sair.copy[d0:%0] %idx {
      instances = [{expansion = "copy"}]
    } : !sair.value<d0:dyn_range, index>
    // CHECK: sair.map
    %3 = sair.copy[d0:%1] %idx {
      instances = [{expansion = "copy"}]
    } : !sair.value<d0:dyn_range, index>
    // CHECK: %[[ALLOC:.*]] = sair.map
    %4 = sair.alloc[d0:%0, d1:%1] %2(d0), %3(d1) {
      instances = [{expansion = "alloc"}]
    } : !sair.value<d0:dyn_range x d1:dyn_range, memref<?x?xf32>>
    // CHECK: sair.map[d0:%[[D1]], d1:%[[D0]]] %[[ALLOC]](d1, d0)
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %[[ARG2:.*]]: memref<?x?xf32>):
    // CHECK:   memref.dealloc %[[ARG2]] : memref<?x?xf32>
    // CHECK:   sair.return
    // CHECK: } : #sair.shape<d0:dyn_range x d1:dyn_range>, (memref<?x?xf32>) -> ()
    sair.free[d0:%1, d1:%0] %4(d1, d0) {
      instances = [{expansion = "free"}]
    }: !sair.value<d0:dyn_range x d1:dyn_range, memref<?x?xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @load_from_memref
func @load_from_memref(%arg0 : memref<?x?xf32>) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<8, 2>
    %1, %2 = sair.map[d0:%0] attributes {
      instances = [{expansion = "map"}]
    } {
      ^bb0(%arg1: index):
        %c4 = constant 4 : index
        %3 = addi %arg1, %c4 : index
        sair.return %arg1, %3 : index, index
    } : #sair.shape<d0:static_range<8, 2>>, () -> (index, index)
    %3 = sair.dyn_range[d0:%0] %1(d0), %2(d0) : !sair.dyn_range<d0:static_range<8,2>>
    %4 = sair.from_scalar %arg0 : !sair.value<(), memref<?x?xf32>>
    // CHECK: = sair.map[d0:%{{.*}}, d1:%{{.*}}, d2:%{{.*}}] %{{.*}}, %{{.*}}#0(d0), %{{.*}}#1(d0)
    // CHECK: ^{{.*}}(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[MEMREF:.*]]: memref<?x?xf32>, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index):
    // CHECK:   %[[I0:.*]] = affine.apply affine_map<(d0, d1, d2)[s0] -> (d2 - s0)>(%[[ARG1]], %[[ARG2]], %[[ARG3]])[%[[ARG4]]]
    // CHECK:   %[[C0:.*]] = constant 0
    // CHECK:   %[[I1:.*]] = affine.apply affine_map<(d0, d1, d2)[s0] -> ((d1 - s0) floordiv 2)>(%[[ARG1]], %[[ARG2]], %[[ARG3]])[%[[C0]]]
    // CHECK:   %[[VALUE:.*]] = memref.load %[[MEMREF]][%[[I0]], %[[I1]]] : memref<?x?xf32>
    // CHECK:   sair.return %[[VALUE]] : f32
    // CHECK: } : #sair.shape<d0:static_range<8, 2> x d1:static_range<8, 2> x d2:dyn_range(d0)>, (memref<?x?xf32>, index, index) -> f32
    %5 = sair.load_from_memref[d0:%0, d1:%0, d2:%3] %4 {
      layout = #sair.mapping<3 : d2, d1>,
      instances = [{expansion = "load"}]
    } : memref<?x?xf32> -> !sair.value<d0:static_range<8, 2> x d1:static_range<8, 2> x d2:dyn_range(d0), f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @store_to_memref
func @store_to_memref(%arg0 : f32, %arg1 : memref<?x?xf32>) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<8>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.from_scalar %arg1 : !sair.value<(), memref<?x?xf32>>
    // CHECK: sair.map
    %3 = sair.copy[d0:%0, d1:%0, d2:%0] %1 {
      instances = [{expansion = "copy"}]
    } : !sair.value<d0:static_range<8> x d1:static_range<8> x d2:static_range<8>, f32>

    // CHECK: sair.map[d0:%{{.*}}, d1:%{{.*}}, d2:%{{.*}}] %{{.*}}, %{{.*}}(d0, d1, d2)
    // CHECK: ^{{.*}}(%[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[MEMREF:.*]]: memref<?x?xf32>, %[[VALUE:.*]]: f32):
    // CHECK:   %[[C0_0:.*]] = constant 0
    // CHECK:   %[[I0:.*]] = affine.apply affine_map<(d0, d1, d2)[s0] -> (d2 - s0)>(%[[ARG1]], %[[ARG2]], %[[ARG3]])[%[[C0_0]]]
    // CHECK:   %[[C0_1:.*]] = constant 0
    // CHECK:   %[[I1:.*]] = affine.apply affine_map<(d0, d1, d2)[s0] -> (d1 - s0)>(%[[ARG1]], %[[ARG2]], %[[ARG3]])[%[[C0_1]]]
    // CHECK:   memref.store %[[VALUE]], %[[MEMREF]][%[[I0]], %[[I1]]]
    // CHECK:   sair.return
    // CHECK: } : #sair.shape<d0:static_range<8> x d1:static_range<8> x d2:static_range<8>>, (memref<?x?xf32>, f32) -> ()
    sair.store_to_memref[d0:%0, d1:%0, d2:%0] %2, %3(d0, d1, d2) {
      layout = #sair.mapping<3 : d2, d1>,
        instances = [{expansion = "store"}]
    } : #sair.shape<d0:static_range<8> x d1:static_range<8> x d2:static_range<8>>, memref<?x?xf32>
    sair.exit
  }
  return
}
