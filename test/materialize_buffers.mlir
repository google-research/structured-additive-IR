// RUN: sair-opt %s -sair-materialize-buffers -mlir-print-local-scope | FileCheck %s

// CHECK-LABEL: @from_to_memref
func @from_to_memref(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n { instances = [{}] } : !sair.value<(), index>
    // CHECK: %[[M0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), memref<?xf32>>
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), memref<?xf32>>
    // CHECK: %[[M1:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), memref<?xf32>>
    %1 = sair.from_scalar %arg1 { instances = [{}] } : !sair.value<(), memref<?xf32>>
    %2 = sair.dyn_range %sn { instances = [{}] } : !sair.dyn_range
    %3 = sair.from_memref %0 memref[d0:%2] {
      instances = [{}],
      buffer_name = "ARG0"
    } : #sair.shape<d0:dyn_range>, memref<?xf32>
    // CHECK: %[[V0:.*]] = sair.load_from_memref[d0:%{{.*}}] %[[M0]]
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "loopA", unroll = 42 : i64}]
    // CHECK:   layout = #sair.mapping<1 : d0>
    // CHECK: %[[V1:.*]] = sair.copy[d0:%{{.*}}] %[[V0]](d0)
    %4 = sair.copy[d0:%2] %3(d0) {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>, unroll = 42}],
        storage = [{name = "ARG1", space = "memory",
                    layout = #sair.named_mapping<[d0:"loopA"] -> (d0)>}]
       }]
    } : !sair.value<d0:dyn_range, f32>
    // CHECK: sair.store_to_memref[d0:%{{.*}}] %[[M1]], %[[V1]]
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "loopA", unroll = 42 : i64}]
    // CHECK:   layout = #sair.mapping<1 : d0>
    sair.to_memref %1 memref[d0:%2] %4(d0) {
      instances = [{}],
      buffer_name = "ARG1"
    } : #sair.shape<d0:dyn_range>, memref<?xf32>
    sair.exit { instances = [{}] }
  }
  return
}

// CHECK-LABEL: @static_shape
func @static_shape(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    %1 = sair.static_range { instances = [{}] } : !sair.static_range<16, 2>
    // CHECK: %[[V0:.*]] = sair.alloc
    // CHECK-SAME: expansion = "alloc", loop_nest = [], operands = [], sequence = 0
    // CHECK-SAME: storage = [{layout = #sair.named_mapping<[] -> ()>, space = "register"}]
    // CHECK-SAME: : !sair.value<(), memref<8xf32>>
    // CHECK: sair.free %[[V0]] {
    // CHECK-SAME:   loop_nest = [], operands = [#sair.instance<0>], sequence = 5
    // CHECK-SAME: } : !sair.value<(), memref<8xf32>>
    // CHECK: %[[V1:.*]] = sair.copy[d0:%{{.*}}]
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
        storage = [{
          name = "B", space = "memory",
          layout = #sair.named_mapping<[d0:"A"] -> (d0)>
        }]
      }]
    } : !sair.value<d0:static_range<16, 2>, f32>
    // CHECK: sair.store_to_memref[d0:%{{.*}}] %[[V0]], %[[V1]](d0)
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "A"}]
    // CHECK:   layout = #sair.mapping<1 : d0>
    // CHECK:   : #sair.shape<d0:static_range<16, 2>>, memref<8xf32>

    // CHECK: %[[V2:.*]] = sair.load_from_memref[d0:%{{.*}}] %[[V0]]
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "B"}]
    // CHECK:   layout = #sair.mapping<1 : d0>
    // CHECK:   : memref<8xf32> -> !sair.value<d0:static_range<16, 2>, f32>
    // CHECK: sair.copy[d0:%{{.*}}] %[[V2]](d0)
    %3 = sair.copy[d0:%1] %2(d0) {
      instances = [{
        loop_nest = [{name = "B", iter = #sair.mapping_expr<d0>}],
        storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
      }]
    } : !sair.value<d0:static_range<16, 2>, f32>
    %4 = sair.proj_last of[d0:%1] %3(d0) { instances = [{}] } : #sair.shape<d0:static_range<16, 2>>, f32
    sair.exit %4 { instances = [{}] } : f32
  } : f32
  return
}

// CHECK-LABEL: @dynamic_shape
func @dynamic_shape(%arg0: f32, %arg1: index, %arg2: index) {
  sair.program {
    // CHECK: %[[V3:.*]] = sair.map %[[V1:.*]], %[[V2:.*]] attributes {
    // CHECK:   loop_nest = []
    // CHECK:   storage = [{layout = #sair.named_mapping<[] -> ()>, space = "register"}]
    // CHECK: } {
    // CHECK:   ^{{.*}}(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index):
    // CHECK:     %[[V4:.*]] = affine.apply
    // CHECK:       affine_map<(d0, d1) -> ((d1 - d0) ceildiv 4)>(%[[ARG0]], %[[ARG1]])
    // CHECK:     sair.return %[[V4]]
    // CHECK: } : #sair.shape<()>, (index, index) -> index

    // CHECK: %[[V5:.*]] = sair.alloc %[[V3]]
    // CHECK:   : !sair.value<(), memref<?xf32>>

    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    // CHECK: %[[V1]] = sair.from_scalar %{{.*}} : !sair.value<(), index>
    %1 = sair.from_scalar %arg1 { instances = [{}] } : !sair.value<(), index>
    // CHECK: %[[V2]] = sair.from_scalar %{{.*}} : !sair.value<(), index>
    %2 = sair.from_scalar %arg2 { instances = [{}] } : !sair.value<(), index>
    %3 = sair.dyn_range %1, %2 step 4 { instances = [{}] } : !sair.dyn_range
    %4 = sair.copy[d0:%3] %0 {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
        storage = [{
          name = "B", space = "memory",
          layout = #sair.named_mapping<[d0:"A"] -> (d0)>
        }]
      }]
    } : !sair.value<d0:dyn_range, f32>
    sair.exit { instances = [{}] }
  }
  return
}

// CHECK-LABEL: @loop_nest
func @loop_nest(%arg0: f32) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.placeholder {instances = [{operands = []}]} : !sair.static_range<16, 4>

    // CHECK: %[[V0:.*]] = sair.map[d0:%[[D0]]] attributes {
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "A"}]
    // CHECK: } {
    // CHECK:   ^{{.*}}(%[[ARG0:.*]]: index):
    // CHECK:     %[[V1:.*]] = affine.apply affine_map<(d0) -> (d0)>(%[[ARG0]])
    // CHECK:     %[[C4:.*]] = constant 4
    // CHECK:     %[[V2:.*]] = addi %[[V1]], %[[C4]]
    // CHECK:     %[[C16:.*]] = constant 16
    // CHECK:     %[[V3:.*]] = cmpi ult, %[[C16]], %[[V2]]
    // CHECK:     %[[V4:.*]] = select %[[V3]], %[[C16]], %[[V2]]
    // CHECK:     %[[V5:.*]] = affine.apply affine_map<(d0, d1) -> (d1 - d0)>
    // CHECK:     sair.return %[[V5]] : index
    // CHECK: } : #sair.shape<d0:static_range<16, 4>>, () -> index

    // CHECK: %[[V6:.*]] = sair.alloc[d0:%[[D0]]] %[[V0]](d0) {
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "A"}]
    // CHECK: }  : !sair.value<d0:static_range<16, 4>, memref<?xf32>>

    // CHECK: sair.free[d0:%[[D0]]] %[[V6]](d0) {
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "A"}]
    // CHECK: } : !sair.value<d0:static_range<16, 4>, memref<?xf32>>

    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    %1 = sair.static_range { instances = [{}] } : !sair.static_range<16>

    // CHECK: %[[V7:.*]] = sair.copy
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ],
        storage = [{
          name = "buf", space = "memory",
          layout = #sair.named_mapping<[d0:"B"] -> (d0)>
        }]
      }]
    } : !sair.value<d0:static_range<16>, f32>
    // CHECK: sair.store_to_memref[d0:%{{.*}}, d1:%{{.*}}] %[[V6]](d0), %[[V7]](unstripe(d0, d1, [4, 1]))
    // CHECK:   loop_nest = [{iter = #sair.mapping_expr<d0>, name = "A"}, {iter = #sair.mapping_expr<d1>, name = "B"}]
    // CHECK:   layout = #sair.mapping<2 : d1>
    // CHECK:   : #sair.shape<d0:static_range<16, 4> x d1:dyn_range(d0)>, memref<?xf32>

    // CHECK: %[[V8:.*]] = sair.load_from_memref[d0:%{{.*}}, d1:%{{.*}}] %[[V6]](d0)
    // CHECK:   loop_nest = [
    // CHECK:     {iter = #sair.mapping_expr<d0>, name = "A"},
    // CHECK:     {iter = #sair.mapping_expr<d1>, name = "C"},
    // CHECK:     {iter = #sair.mapping_expr<d2>, name = "D"}]
    // CHECK:   layout = #sair.mapping<3 : d1>
    // CHECK:   : memref<?xf32> -> !sair.value<d0:static_range<16, 4> x d1:dyn_range(d0) x d2:static_range<16>, f32>
    // CHECK: %[[V9:.*]] = sair.proj_any[d0:%{{.*}}] of[d1:%{{.*}}] %[[V8]](stripe(d0, [4]), stripe(d0, [4, 1]), d1)
    // CHECK:   : #sair.shape<d0:static_range<16> x d1:static_range<16>>, f32
    // CHECK: %[[V10:.*]] = sair.copy[d0:%{{.*}}] %[[V9]](d0)
    %3 = sair.copy[d0:%1] %2(d0) {
      instances = [{
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>},
          {name = "D", iter = #sair.mapping_expr<none>}
        ],
        storage = [{
          name = "buf", space = "memory",
          layout = #sair.named_mapping<[d0:"C"] -> (d0)>
        }]
      }]
    } : !sair.value<d0:static_range<16>, f32>
    %4 = sair.copy[d0:%1, d1:%1] %0 {
      instances = [{
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "C", iter = #sair.mapping_expr<stripe(d0, [4, 1])>},
          {name = "D", iter = #sair.mapping_expr<d1>}
        ],
        storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
      }]
    } : !sair.value<d0:static_range<16> x d1:static_range<16>, f32>
    // CHECK: sair.store_to_memref[d0:%{{.*}}, d1:%{{.*}}, d2:%{{.*}}] %[[V6]](d0), %[[V10]](unstripe(d0, d1, [4, 1]))
    // CHECK:   loop_nest = [
    // CHECK:       {iter = #sair.mapping_expr<d0>, name = "A"},
    // CHECK:       {iter = #sair.mapping_expr<d1>, name = "C"},
    // CHECK:       {iter = #sair.mapping_expr<d2>, name = "D"}]
    // CHECK:   layout = #sair.mapping<3 : d1>
    // CHECK:   : #sair.shape<d0:static_range<16, 4> x d1:dyn_range(d0) x d2:static_range<16>>, memref<?xf32>
    sair.exit { instances = [{}] }
  }
  return
}

// CHECK-LABEL: @sequence_attr
func @sequence_attr(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    %1 = sair.static_range { instances = [{}] } : !sair.static_range<16>

    // CHECK-DAG: sair.alloc{{.*}}sequence = 4
    // CHECK-DAG: sair.free{{.*}}sequence = 10
    // CHECK-DAG: sair.alloc{{.*}}sequence = 0
    // CHECK-DAG: sair.free{{.*}}sequence = 3

    // CHECK: sair.copy
    // CHECK-SAME: sequence = 5
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
        storage = [{
          name = "buf2", space = "memory",
          layout = #sair.named_mapping<[d0:"A"] -> (d0)>
        }],
        sequence = 2
      }]
    } : !sair.value<d0:static_range<16>, f32>
    // CHECK: sair.store_to_memref
    // CHECK-SAME: sequence = 6

    // CHECK: sair.copy
    // CHECK-SAME: sequence = 1
    %3 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "B", iter = #sair.mapping_expr<d0>}],
        storage = [{
          name = "buf1", space = "memory",
          layout = #sair.named_mapping<[d0:"B"] -> (d0)>
        }],
        sequence = 1
      }]
    } : !sair.value<d0:static_range<16>, f32>
    // CHECK: sair.store_to_memref
    // CHECK-SAME: sequence = 2

    // CHECK: sair.load_from_memref
    // CHECK-SAME: sequence = 7
    // CHECK: sair.copy
    // CHECK-SAME: sequence = 8
    %4 = sair.copy[d0:%1] %2(d0) {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
        storage = [{
          name = "buf2", space = "memory",
          layout = #sair.named_mapping<[d0:"A"] -> (d0)>
        }],
        sequence = 3
      }]
    } : !sair.value<d0:static_range<16>, f32>
    // CHECK: sair.store_to_memref
    // CHECK-SAME: sequence = 9
    sair.exit { instances = [{}] }
  }
  return
}
