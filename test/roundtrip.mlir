// RUN: sair-opt -allow-unregistered-dialect %s | sair-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: -> !sair.range
func private @range_type() -> !sair.range

// Make sure empty type-dependence list is dropped.
// CHECK: -> !sair.range
// CHECK-NOT: <()>
func private @independent_range_type() -> !sair.range<()>

// CHECK: -> !sair.range<d0:range>
func private @dependent_range_type_1() -> !sair.range<d0:range>

// CHECK: -> !sair.value<(), f32>
func private @value_type_empty_domain() -> !sair.value<(), f32>

// CHECK: -> !sair.value<d0:range x d1:range, f32>
func private @value_type() -> !sair.value<d0:range x d1:range, f32>

// CHECK: -> !sair.range<d0:range x d1:range(d0)>
func private @dependent_dimensions() -> !sair.range<d0:range x d1:range(d0)>

// CHECK: -> !sair.range<d0:range x d1:range(d0) x d2:range(d0, d1)>
func private @dependent_dimensions_2()
  -> !sair.range<d0:range x d1:range(d0) x d2:range(d0, d1)>

// CHECK-LABEL: @sair_program
func @sair_program() {
  // CHECK: sair.program
  sair.program {
    // CHECK: sair.exit
    sair.exit
  }
  return
}

// CHECK-LABEL: @sair_program_return_values
func @sair_program_return_values() {
  %c0 = constant 1.0 : f32
  %c1 = constant 1 : i32
  // CHECK: %{{.*}}:2 = sair.program
  %0:2 = sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %2 = sair.from_scalar %c0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), i32>
    %3 = sair.from_scalar %c1 : !sair.value<(), i32>
    // CHECK: sair.exit %[[V0]], %[[V1]] : f32, i32
    sair.exit %2, %3 : f32, i32
  // CHECK: } : f32, i32
  } : f32, i32
  return
}

// CHECK-LABEL: @range_op
func @range_op() {
  %c0 = constant 0 : index
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %c0 : !sair.value<(), index>
    // CHECK: %{{.*}} = sair.dyn_range %[[V0]] : !sair.range
    %1 = sair.dyn_range %0 : !sair.range
    sair.exit
  }
  return
}

// CHECK-LABEL: @range_with_step
func @range_with_step(%arg0: index, %arg1: index) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[V1:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg1 : !sair.value<(), index>
    // CHECK: %{{.*}} = sair.dyn_range %[[V0]], %[[V1]] step 2 : !sair.range
    %2 = sair.dyn_range %0, %1 step 2 : !sair.range
    sair.exit
  }
  return
}

// CHECK-LABEL: @static_range_op
func @static_range_op() {
  sair.program {
    // CHECK: %{{.*}} = sair.static_range : !sair.static_range<42>
    %0 = sair.static_range : !sair.static_range<42>
    sair.exit
  }
  return
}

// CHECK-LABEL: @static_range_with_step
func @static_range_with_step() {
  sair.program {
    // CHECK: %{{.*}} = sair.static_range : !sair.static_range<42, 2>
    %0 = sair.static_range : !sair.static_range<42, 2>
    sair.exit
  }
  return
}

// CHECK-LABEL: @dependent_range_op
func @dependent_range_op(%arg0 : index) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range %[[V0]] : !sair.range
    %1 = sair.dyn_range %0 : !sair.range
    // CHECK: %[[V1:.*]] = sair.copy
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, index>

    // CHECK: %[[D1:.*]] = sair.dyn_range[d0:%[[D0]]] %[[V1]](d0)
    // CHECK-SAME: : !sair.range<d0:range>
    %3 = sair.dyn_range[d0:%1] %2(d0) : !sair.range<d0:range>
    // CHECK: %[[V2:.*]] = sair.copy
    %4 = sair.copy[d0:%1, d1:%3] %0
      : !sair.value<d0:range x d1:range(d0), index>

    // CHECK: %{{.*}} = sair.dyn_range[d0:%[[D0]], d1:%[[D1]]] %[[V2]](d0, d1)
    // CHECK-SAME: : !sair.range<d0:range x d1:range(d0)>
    %5 = sair.dyn_range[d0:%1, d1:%3] %4(d0, d1)
      : !sair.range<d0:range x d1:range(d0)>
    sair.exit
  }
  return
}

// CHECK-LABEL: @generic_mapping
func @generic_mapping() {
  // CHECK: "foo"() {mapping = #sair.mapping<3 : d0, d2, d1, none>}
  "foo"() {mapping = #sair.mapping<3 : d0, d2, d1, none>} : () -> ()
  // CHECK: "bar"() {mapping = #sair.mapping<4 : d0, d1, d2>}
  "bar"() {mapping = #sair.mapping<4 : d0, d1, d2>} : () -> ()
}

// CHECK-LABEL: @generic_mapping_empty
func @generic_mapping_empty() {
  // CHECK: "foo"() {mapping = #sair.mapping<0>}
  "foo"() {mapping = #sair.mapping<0>} : () -> ()
}

// CHECK-LABEL: @shape_attribute
func @shape_attribute() {
  // CHECK: "foo"() {shape = #sair.shape<d0:range x d1:range(d0)>}
  "foo"() {shape = #sair.shape<d0:range x d1:range(d0)>} : () -> ()
}

// CHECK-LABEL: @copy
func @copy(%arg0 : f32) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range
    %0 = sair.static_range : !sair.static_range<8>
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.copy[d0:%[[D0]], d1:%[[D0]]] %[[V0]]
    // CHECK-SAME: : !sair.value<d0:static_range<8> x d1:static_range<8>, f32>
    %2 = sair.copy[d0:%0, d1:%0] %1
      : !sair.value<d0:static_range<8> x d1:static_range<8>, f32>
    // CHECK: %{{.*}} = sair.copy[d0:%[[D0]], d1:%[[D0]]] %[[V1]](d1, d0)
    // CHECK-SAME: : !sair.value<d0:static_range<8> x d1:static_range<8>, f32>
    %3 = sair.copy[d0:%0, d1:%0] %2(d1, d0)
      : !sair.value<d0:static_range<8> x d1:static_range<8>, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @copy_attributes
func @copy_attributes(%arg0 : f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: foo = 3
    %1 = sair.copy %0 {foo = 3} : !sair.value<(), f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @from_memref
func @from_memref(%arg0 : memref<?x?xf32>) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    // CHECK: %[[V0:.]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), memref<?x?xf32>>
    // CHECK: %{{.*}} = sair.from_memref[d0:%[[D0]]] %[[V0]] memref[d1:%[[D0]], d2:%[[D0]]]
    // CHECK: : #sair.shape<d0:range x d1:range x d2:range>, memref<?x?xf32>
    %2 = sair.from_memref[d0:%0] %1 memref[d1:%0, d2:%0] {
      buffer_name = "bufferA"
    } : #sair.shape<d0:range x d1:range x d2:range>, memref<?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @to_memref
func @to_memref(%arg0 : f32, %arg1 : memref<?x?xf32>) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), memref<?x?xf32>>
    %2 = sair.from_scalar %arg1 : !sair.value<(), memref<?x?xf32>>
    // CHECK: %[[V2:.*]] = sair.copy[d0:%{{.*}}] %[[V1]]
    %3 = sair.copy[d0:%0] %2 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, memref<?x?xf32>>
    // CHECK: %[[V0:.*]] = sair.copy
    %4 = sair.copy[d0:%0, d1:%0, d2:%0] %1 {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<d0>},
        {name = "B", iter = #sair.mapping_expr<d1>},
        {name = "C", iter = #sair.mapping_expr<d2>}
      ]
    } : !sair.value<d0:range x d1:range x d2:range, f32>
    // CHECK: sair.to_memref[d0:%[[D0]]] %[[V2]]
    // CHECK:  memref[d1:%[[D0]], d2:%[[D0]]] %[[V0]](d0, d1, d2)
    // CHECK:  : #sair.shape<d0:range x d1:range x d2:range>, memref<?x?xf32>
    sair.to_memref[d0:%0] %3(d0) memref[d1:%0, d2:%0] %4(d0, d1, d2) {
      buffer_name = "bufferA"
    } : #sair.shape<d0:range x d1:range x d2:range>, memref<?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @load_from_memref
func @load_from_memref(%arg0 : memref<?x?xf32>) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    // CHECK: %[[V0:.]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), memref<?x?xf32>>
    // CHECK: %{{.*}} = sair.load_from_memref[d0:%[[D0]], d1:%[[D0]], d2:%[[D0]]] %[[V0]]
    // CHECK:   layout = #sair.mapping<3 : d1, d2>
    // CHECK: : memref<?x?xf32> -> !sair.value<d0:range x d1:range x d2:range, f32>
    %2 = sair.load_from_memref[d0:%0, d1:%0, d2:%0] %1 {
      layout = #sair.mapping<3: d1, d2>
    } : memref<?x?xf32> -> !sair.value<d0:range x d1:range x d2:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @store_to_memref
func @store_to_memref(%arg0 : f32, %arg1 : memref<?x?xf32>) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), memref<?x?xf32>>
    %2 = sair.from_scalar %arg1 : !sair.value<(), memref<?x?xf32>>
    // CHECK: sair.store_to_memref[d0:%[[D0]], d1:%[[D0]], d2:%[[D0]]] %[[V1]], %[[V0]]
    // CHECK:   layout = #sair.mapping<3 : d1, d2>
    // CHECK:  : #sair.shape<d0:range x d1:range x d2:range>, memref<?x?xf32>
    sair.store_to_memref[d0:%0, d1:%0, d2:%0] %2, %1{
      layout = #sair.mapping<3: d1, d2>
    } : #sair.shape<d0:range x d1:range x d2:range>, memref<?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @map
func @map(%arg0 : f32, %arg1: i32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.from_scalar %arg1 : !sair.value<(), i32>
    // CHECK: %[[V0:.*]] = sair.copy{{.*}} : !sair.value<{{.*}}, f32>
    %3 = sair.copy[d0:%0, d1:%0] %1 : !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[V1:.*]] = sair.copy{{.*}} : !sair.value<{{.*}}, i32>
    %4 = sair.copy[d0:%0, d1:%0] %2 : !sair.value<d0:range x d1:range, i32>

    // CHECK: sair.map[d0:%[[D0]], d1:%[[D0]]] %[[V0]](d0, d1), %[[V1]](d1, d0) {
    sair.map[d0:%0, d1:%0] %3(d0, d1), %4(d1, d0) {
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %[[ARG0:.*]]: f32, %[[ARG1:.*]]: i32):
    ^bb0(%arg2: index, %arg3: index, %arg4: f32, %arg5: i32):
      // CHECK: sair.return %[[ARG0]], %[[ARG1]]
      sair.return %arg4, %arg5 : f32, i32
    // CHECK: } : #sair.shape<d0:range x d1:range>, (f32, i32) -> (f32, i32)
    } : #sair.shape<d0:range x d1:range>, (f32, i32) -> (f32, i32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @map_noargs
func @map_noargs() {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    // CHECK: sair.map[d0:%[[D0]], d1:%[[D0]]] {
    sair.map[d0:%0, d1:%0] {
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index):
    ^bb0(%arg0: index, %arg1: index):
      %c0 = constant 0.0 : f32
      %c1 = constant 1 : i32
      // CHECK: sair.return {{.*}} : f32, i32
      sair.return %c0, %c1 : f32, i32
    // CHECK: } : #sair.shape<d0:range x d1:range>, () -> (f32, i32)
    } : #sair.shape<d0:range x d1:range>, () -> (f32, i32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @return_noargs
func @return_noargs() {
  sair.program {
    sair.map {
      ^bb0:
        // CHECK-EXACT: sair.return
        sair.return
    } : #sair.shape<()>, () -> ()
    sair.exit
  }
  return
}

// CHECK-LABEL: @map_reduce
func @map_reduce(%arg0 : f32, %arg1 : i32, %arg2 : f64) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.from_scalar %arg1 : !sair.value<(), i32>
    %3 = sair.from_scalar %arg2 : !sair.value<(), f64>
    // CHECK: %[[V0:.*]] = sair.copy{{.*}} : !sair.value<{{.*}}, f32>
    %4 = sair.copy[d0:%0, d1:%0] %1 : !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[V1:.*]] = sair.copy{{.*}} : !sair.value<{{.*}}, i32>
    %5 = sair.copy[d0:%0, d1:%0] %2 : !sair.value<d0:range x d1:range, i32>
    // CHECK: %[[V2:.*]] = sair.copy{{.*}} : !sair.value<{{.*}}, f64>
    %6 = sair.copy[d0:%0, d1:%0] %3 : !sair.value<d0:range x d1:range, f64>

    // CHECK: sair.map_reduce[d0:%[[D0]], d1:%[[D0]]] %[[V0]](d0, d1)
    // CHECK:          reduce[d2:%[[D0]]] %[[V1]](d0, d2), %[[V2]](d2, d0)
    // CHECK:          attributes {foo = "bar"} {
    sair.map_reduce[d0:%0, d1:%0] %4(d0, d1)
             reduce[d2:%0] %5(d0, d2), %6(d2, d0)
             attributes {foo = "bar"} {
    // CHECK: ^{{.*}}({{.*}}: index, {{.*}}: index, {{.*}}: index, %[[ARG0:.*]]: f32, {{.*}}: i32, {{.*}}: f64):
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: f32, %arg7: i32, %arg8: f64):
      // CHECK: sair.return %[[ARG0]] : f32
      sair.return %arg6 : f32
    // CHECK: } : #sair.shape<d0:range x d1:range x d2:range>, (i32, f64) -> f32
    } : #sair.shape<d0:range x d1:range x d2:range>, (i32, f64) -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @from_scalar
func @from_scalar() {
  // CHECK: %[[V0:.*]] = constant 1 : index
  %0 = constant 1 : index
  sair.program {
    // CHECK: %{{.*}} sair.from_scalar %[[V0]] : !sair.value<(), index>
    %1 = sair.from_scalar %0 : !sair.value<(), index>
    sair.exit
  }
  return
}

// CHECK-LABEL: @loop_nest_attr
func @loop_nest_attr(%arg0: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy[d0:%{{.*}}, d1:%{{.*}}] %[[V0]] {
    sair.copy[d0:%0, d1:%0] %1 {
      loop_nest = [
        // CHECK: {iter = #sair.mapping_expr<none>, name = "loopA"}
        {name = "loopA", iter = #sair.mapping_expr<none>},
        // CHECK: {iter = #sair.mapping_expr<d0>, name = "loopB"}
        {name = "loopB", iter = #sair.mapping_expr<d0>},
        // CHECK: {iter = #sair.mapping_expr<d1>, name = "loopC"}
        {name = "loopC", iter = #sair.mapping_expr<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    sair.copy[d0:%0] %1 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @proj_any
func @proj_any(%arg0: f32) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range : !sair.static_range<4>
    %0 = sair.static_range : !sair.static_range<4>
    // CHECK: %[[D1:.*]] = sair.static_range : !sair.static_range<8>
    %1 = sair.static_range : !sair.static_range<8>
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V0:.*]] = sair.copy
    %3 = sair.copy[d0:%0, d1:%1] %2
      : !sair.value<d0:static_range<4> x d1:static_range<8>, f32>
    // CHECK: sair.proj_any[d0:%[[D0]]] of[d1:%[[D1]]] %[[V0]](d0, d1)
    %4 = sair.proj_any[d0:%0] of[d1:%1] %3(d0, d1)
      : #sair.shape<d0:static_range<4> x d1:static_range<8>>, f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @proj_last
func @proj_last(%arg0: f32) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range : !sair.static_range<4>
    %0 = sair.static_range : !sair.static_range<4>
    // CHECK: %[[D1:.*]] = sair.static_range : !sair.static_range<8>
    %1 = sair.static_range : !sair.static_range<8>
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V0:.*]] = sair.copy
    %3 = sair.copy[d0:%0, d1:%1] %2
      : !sair.value<d0:static_range<4> x d1:static_range<8>, f32>
    // CHECK: sair.proj_last[d0:%[[D0]]] of[d1:%[[D1]]] %[[V0]](d0, d1)
    %4 = sair.proj_last[d0:%0] of[d1:%1] %3(d0, d1)
      : #sair.shape<d0:static_range<4> x d1:static_range<8>>, f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @proj_last_different_shape
func @proj_last_different_shape(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range : !sair.static_range<4>
    // CHECK: sair.proj_last of[d0:%{{.*}}] %{{.*}} : #sair.shape<d0:static_range<4>>, f32
    %2 = sair.proj_last of[d0:%1] %0 : #sair.shape<d0:static_range<4>>, f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @fby

func @fby(%arg0: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %1 = sair.dyn_range %sn : !sair.range
    // CHECK: %[[V1:.*]] = sair.copy[{{.*}}] %[[V0]]
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, f32>

    // CHECK: %[[D1:.*]] = sair.dyn_range
    %3 = sair.dyn_range %sn : !sair.range
    // CHECK: %[[V2:.*]] = sair.fby[d0:%[[D0]]] %[[V1]](d0) then[d1:%[[D1]]] %[[V3:.*]](d0, d1)
    // CHECK:   : !sair.value<d0:range x d1:range, f32>
    %4 = sair.fby[d0:%1] %2(d0) then[d1:%3] %5(d0, d1)
      : !sair.value<d0:range x d1:range, f32>
    // CHECK: %[[V3]] = sair.copy[d0:%[[D0]], d1:%[[D1]]] %[[V2]](d0, d1)
    %5 = sair.copy[d0:%1, d1:%3] %4(d0, d1) : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @undef
func @undef() {
  // CHECK: %[[UNDEF:.*]] = sair.undef : f32
  %0 = sair.undef : f32
  sair.program {
    // CHECK: sair.from_scalar %[[UNDEF]]
    %1 = sair.from_scalar %0 : !sair.value<(), f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @mapping_expr_attr
func @mapping_expr_attr() {
  // CHECK: "foo"() {mapping_expr = #sair.mapping_expr<d0>}
  "foo"() {mapping_expr = #sair.mapping_expr<d0>} : () -> ()
  // CHECK: "foo"() {mapping_expr = #sair.mapping_expr<none>}
  "foo"() {mapping_expr = #sair.mapping_expr<none>} : () -> ()
  // CHECK: "foo"() {mapping_expr = #sair.mapping_expr<stripe(d0, [4])>}
  "foo"() {mapping_expr = #sair.mapping_expr<stripe(d0, [4])>} : () -> ()
  // CHECK: "foo"() {mapping_expr = #sair.mapping_expr<stripe(d0, [4, 1])>}
  "foo"() {mapping_expr = #sair.mapping_expr<stripe(d0, [4, 1])>} : () -> ()
  // CHECK: "foo"() {mapping_expr = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>}
  "foo"() {mapping_expr = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>} : () -> ()
}

func @stripe_mined_loop() {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    sair.map[d0:%0] attributes {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
        {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
      ]
    } {
      ^bb0(%arg0: index):
        sair.return
    } : #sair.shape<d0:range>, () -> ()
    sair.exit
  }
  return
}

func @stripe_mapping(%arg0: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[D0:.*]] = sair.dyn_range
    %0 = sair.dyn_range %sn : !sair.range
    // CHECK: %[[D1:.*]] = sair.dyn_range %{{.*}} step 2
    %1 = sair.dyn_range %sn step 2 : !sair.range
    // CHECK: %[[I0:.*]]:2 = sair.map
    %2, %3 = sair.map[d0:%1] {
      ^bb0(%arg1: index):
        %c2 = constant 2 : index
        %4 = addi %arg1, %c2 : index
        sair.return %arg1, %4 : index, index
    } : #sair.shape<d0:range>, () -> (index, index)
    // CHECK: %[[D2:.*]] = sair.dyn_range[d0:%[[D1]]] %[[I0]]#0(d0), %[[I0]]#1(d0)
    %4 = sair.dyn_range[d0:%1] %2(d0), %3(d0) : !sair.range<d0:range>
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %5 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.copy[d0:%[[D0]]] %[[V0]]
    %6 = sair.copy[d0:%0] %5 : !sair.value<d0:range, f32>
    // CHECK: %[[V2:.*]] = sair.copy[d0:%[[D1]], d1:%[[D2]]] %[[V1]](unstripe(d0, d1, [2, 1]))
    %7 = sair.copy[d0:%1, d1:%4] %6(unstripe(d0, d1, [2, 1]))
      : !sair.value<d0:range x d1:range(d0), f32>
    // CHECK: sair.copy[d0:%[[D0]]] %[[V2]](stripe(d0, [2]), stripe(d0, [2, 1]))
    %8 = sair.copy[d0:%0] %7(stripe(d0, [2]), stripe(d0, [2, 1]))
      : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @named_mapping
func @named_mapping() {
  // CHECK: #sair.named_mapping<[] -> ()>
  "foo"() { bar = #sair.named_mapping<[] -> ()>} : () -> ()
  // CHECK: #sair.named_mapping<[d0:"A", d1:"B"] -> (d0, d1)>
  "foo"() { bar = #sair.named_mapping<[d0:"A", d1:"B"] -> (d0, d1)>} : () -> ()
}

// CHECK-LABEL: @alloc_simple
func @alloc_simple() {
  sair.program {
    // CHECK: = sair.alloc : !sair.value<(), memref<42x42xf32>>
    sair.alloc : !sair.value<(), memref<42x42xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @alloc
func @alloc(%arg0: index) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.dyn_range %sn : !sair.range
    %idx = sair.from_scalar %arg0 : !sair.value<(), index>
    %2 = sair.copy[d0:%0] %idx : !sair.value<d0:range, index>
    %3 = sair.copy[d0:%1] %idx : !sair.value<d0:range, index>
    // CHECK: = sair.alloc[d0:%{{.*}}, d1:%{{.*}}] %{{.*}}(d0), %{{.*}}(d1) : !sair.value<d0:range x d1:range, memref<?x?xf32>>
    sair.alloc[d0:%0, d1:%1] %2(d0), %3(d1) : !sair.value<d0:range x d1:range, memref<?x?xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @alloc_nosize
func @alloc_nosize(%arg0: index) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.dyn_range %sn : !sair.range
    %idx = sair.from_scalar %arg0 : !sair.value<(), index>
    %2 = sair.copy[d0:%0] %idx : !sair.value<d0:range, index>
    %3 = sair.copy[d0:%1] %idx : !sair.value<d0:range, index>
    // CHECK: = sair.alloc[d0:%{{.*}}, d1:%{{.*}}] : !sair.value<d0:range x d1:range, memref<42x42xf32>>
    sair.alloc[d0:%0, d1:%1] : !sair.value<d0:range x d1:range, memref<42x42xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @free_simple
func @free_simple() {
  sair.program {
    %0 = sair.alloc : !sair.value<(), memref<42x42xf32>>
    // CHECK: sair.free %{{.*}} : !sair.value<(), memref<42x42xf32>>
    sair.free %0 : !sair.value<(), memref<42x42xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @sair_free
func @sair_free(%arg0: index) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.dyn_range %sn : !sair.range
    %idx = sair.from_scalar %arg0 : !sair.value<(), index>
    %2 = sair.copy[d0:%0] %idx : !sair.value<d0:range, index>
    %3 = sair.copy[d0:%1] %idx : !sair.value<d0:range, index>
    %4 = sair.alloc[d0:%0, d1:%1] %2(d0), %3(d1) : !sair.value<d0:range x d1:range, memref<?x?xf32>>
    // CHECK: sair.free[d0:%{{.*}}, d1:%{{.*}}] %{{.*}}(d1, d0) : !sair.value<d0:range x d1:range, memref<?x?xf32>>
    sair.free[d0:%1, d1:%0] %4(d1, d0) : !sair.value<d0:range x d1:range, memref<?x?xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @free_nosize
func @free_nosize(%arg0: index) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.dyn_range %sn : !sair.range
    %idx = sair.from_scalar %arg0 : !sair.value<(), index>
    %2 = sair.copy[d0:%0] %idx : !sair.value<d0:range, index>
    %3 = sair.copy[d0:%1] %idx : !sair.value<d0:range, index>
    %4 = sair.alloc[d0:%0, d1:%1] : !sair.value<d0:range x d1:range, memref<42x42xf32>>
    // CHECK: sair.free[d0:%{{.*}}, d1:%{{.*}}] %{{.*}}(d0, d1) : !sair.value<d0:range x d1:range, memref<42x42xf32>>
    sair.free[d0:%0, d1:%1] %4(d0, d1) : !sair.value<d0:range x d1:range, memref<42x42xf32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @storage_stripe
func @storage_stripe(%arg0: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.dyn_range %sn : !sair.range
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
        {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
      ],
      storage = [{
        name = "B", space = "memory",
        layout = #sair.named_mapping<[d0:"B"] -> (d0)>
      }]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @storage_no_layout
func @storage_no_layout(%arg0: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.dyn_range %sn : !sair.range
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
      storage = [{name = "B", space = "memory"}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @placeholder
func @placeholder(%arg0: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.placeholder[d0:%0] : !sair.range<d0:range>
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %3 = sair.copy[d0:%0, d1:%1] %2 : !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @placeholder_with_loop_nest
func @placeholder_with_loop_nest(%arg0: f32) {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.placeholder : !sair.range
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %3 = sair.copy[d0:%0] %2 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    %4 = sair.copy[d0:%1] %2 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @free_with_mapping
func @free_with_mapping() {
  %n = constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.dyn_range %sn : !sair.range
    %1 = sair.alloc[d0:%0] : !sair.value<d0:range, memref<f32>>
    %2 = sair.placeholder : !sair.range
    %3 = sair.placeholder[d0:%2] : !sair.range<d0:range>
    sair.free[d0:%2, d1:%3] %1(unstripe(d0, d1, [4, 1]))
      : !sair.value<d0:range x d1:range(d0), memref<f32>>
    sair.exit
  }
  return
}

// CHECK-LABEL: @mapping_any_expr
func @mapping_any_expr() {
  // CHECK: "foo"() {bar = #sair.mapping_expr<?>} : () -> ()
  "foo"() {bar = #sair.mapping_expr<?>} : () -> ()
  return
}

// CHECK-LABEL: @sequence_attr
func @sequence_attr() {
  sair.program {
    %0 = sair.static_range : !sair.static_range<42>
    // CHECK: sair.alloc
    // CHECK-SAME: sequence = 1
    %1 = sair.alloc[d0:%0] { sequence = 1 } : !sair.value<d0:static_range<42>, memref<f32>>
    // CHECK: sair.free
    // CHECK-SAME: sequence = 3
    sair.free[d0:%0] %1(d0) { sequence = 3 } : !sair.value<d0:static_range<42>, memref<f32>>
    sair.exit
  }
  return
}

// Should not error on fby's "then" operand having the same sequence number
// as the user of "fby" result in case of use-def loop.
// CHECK-LABEL: @sequence_same_fby_then
func @sequence_same_fby_then(%arg0: f32) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<42>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.copy %1 { sequence = 1 } : !sair.value<(), f32>
    %3 = sair.fby %2 then[d0:%0] %4(d0) : !sair.value<d0:static_range<42>, f32>
    // CHECK: sair.map
    // CHECK-SAME: sequence = 2
    %4 = sair.map[d0:%0] %3(d0) attributes { sequence = 2 } {
    ^bb0(%arg1: index, %arg2: f32):
      sair.return %arg2 : f32
    } : #sair.shape<d0:static_range<42>>, (f32) -> (f32)
    sair.exit
  }
  return
}

// CHECK-LABEL: @static_range_type
func @static_range_type() {
  // CHECK: !sair.static_range<8>
  "foo"() : () -> !sair.static_range<8>
  // CHECK: !sair.static_range<8>
  "foo"() : () -> !sair.static_range<8, 1>
  // CHECK: !sair.static_range<8, 2>
  "foo"() : () -> !sair.static_range<8, 2>

  "foo"() {
    // CHECK: #sair.shape<d0:static_range<8, 2>>
    attr = #sair.shape<d0:static_range<8, 2>>
  } : () -> ()
}
