// RUN: sair-opt -allow-unregistered-dialect %s | sair-opt -allow-unregistered-dialect | FileCheck %s

// CHECK: -> !sair.range
func @range_type() -> !sair.range

// Make sure empty type-dependence list is dropped.
// CHECK: -> !sair.range
// CHECK-NOT: <()>
func @independent_range_type() -> !sair.range<()>

// CHECK: -> !sair.range<d0:range>
func @dependent_range_type_1() -> !sair.range<d0:range>

// CHECK: -> !sair.value<(), f32>
func @value_type_empty_domain() -> !sair.value<(), f32>

// CHECK: -> !sair.value<d0:range x d1:range, f32>
func @value_type() -> !sair.value<d0:range x d1:range, f32>

// CHECK: -> !sair.range<d0:range x d1:range(d0)>
func @dependent_dimensions() -> !sair.range<d0:range x d1:range(d0)>

// CHECK: -> !sair.range<d0:range x d1:range(d0) x d2:range(d0, d1)>
func @dependent_dimensions_2()
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
    // CHECK: %{{.*}} = sair.static_range 42 : !sair.range
    %0 = sair.static_range 42 : !sair.range
    sair.exit
  }
  return
}

// CHECK-LABEL: @static_range_with_step
func @static_range_with_step() {
  sair.program {
    // CHECK: %{{.*}} = sair.static_range 42 step 2 : !sair.range
    %0 = sair.static_range 42 step 2 : !sair.range
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

// CHECK-LABEL: @generic_access_pattern
func @generic_access_pattern() {
  // CHECK: "foo"() {access_pattern = #sair.pattern<3 : d0, d2, d1, none>}
  "foo"() {access_pattern = #sair.pattern<3 : d0, d2, d1, none>} : () -> ()
  // CHECK: "bar"() {access_pattern = #sair.pattern<4 : d0, d1, d2>}
  "bar"() {access_pattern = #sair.pattern<4 : d0, d1, d2>} : () -> ()
}

// CHECK-LABEL: @generic_access_pattern_empty
func @generic_access_pattern_empty() {
  // CHECK: "foo"() {access_pattern = #sair.pattern<0>}
  "foo"() {access_pattern = #sair.pattern<0>} : () -> ()
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
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.copy[d0:%[[D0]], d1:%[[D0]]] %[[V0]]
    // CHECK-SAME: : !sair.value<d0:range x d1:range, f32>
    %2 = sair.copy[d0:%0, d1:%0] %1 : !sair.value<d0:range x d1:range, f32>
    // CHECK: %{{.*}} = sair.copy[d0:%[[D0]], d1:%[[D0]]] %[[V1]](d1, d0)
    // CHECK-SAME: : !sair.value<d0:range x d1:range, f32>
    %3 = sair.copy[d0:%0, d1:%0] %2(d1, d0)
      : !sair.value<d0:range x d1:range, f32>
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
// CHECK: %[[ARG0:.*]]: memref<?x?xf32>
func @from_memref(%arg0 : memref<?x?xf32>) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %{{.*}} = sair.from_memref[d0:%[[D0]], d1:%[[D0]]] %[[ARG0]]
    // CHECK: : memref<?x?xf32> -> !sair.value<d0:range x d1:range, f32>
    %1 = sair.from_memref[d0:%0, d1:%0] %arg0
      : memref<?x?xf32> -> !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @to_memref
// CHECK: %{{.*}}: f32, %[[ARG0:.*]]: memref<?x?xf32>
func @to_memref(%arg0 : f32, %arg1 : memref<?x?xf32>) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V0:.*]] = sair.copy
    %2 = sair.copy[d0:%0, d1:%0] %1 : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.to_memref[d0:%[[D0]], d1:%[[D0]]] %[[V0]](d0, d1), %[[ARG0]]
    // CHECK:  : memref<?x?xf32>
    sair.to_memref[d0:%0, d1:%0] %2(d0, d1), %arg1 : memref<?x?xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @map
func @map(%arg0 : f32, %arg1: i32) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
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
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
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
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range
    %0 = sair.static_range 8 : !sair.range
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

// CHECK-LABEL: @iterator_attr
func @iterator_attr() {
  "foo"() {
    // CHECK: attr0 = #sair.iter<d0>
    attr0 = #sair.iter<d0>,
    // CHECK: attr1 = #sair.iter<d1 step 4>
    attr1 = #sair.iter<d1 step 4>
  } : () -> ()
}

// CHECK-LABEL: @loop_nest_attr
func @loop_nest_attr(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy[d0:%{{.*}}, d1:%{{.*}}] %1 {
    sair.copy[d0:%0, d1:%0] %1 {
      loop_nest = [
        // CHECK: {iter = #sair.iter<remat>, name = "loopA"}
        {name = "loopA", iter = #sair.iter<remat>},
        // CHECK: {iter = #sair.iter<d0>, name = "loopB"}
        {name = "loopB", iter = #sair.iter<d0>},
        // CHECK: {iter = #sair.iter<d1>, name = "loopC"}
        {name = "loopC", iter = #sair.iter<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    sair.copy[d0:%0] %1 {
      loop_nest = [{name = "loopA", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @proj_any
func @proj_any(%arg0: f32) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range 4
    %0 = sair.static_range 4 : !sair.range
    // CHECK: %[[D1:.*]] = sair.static_range 8
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V0:.*]] = sair.copy
    %3 = sair.copy[d0:%0, d1:%1] %2 : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.proj_any[d0:%[[D0]]] of[d1:%[[D1]]] %[[V0]](d0, d1)
    %4 = sair.proj_any[d0:%0] of[d1:%1] %3(d0, d1)
      : #sair.shape<d0:range x d1:range>, f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @proj_last
func @proj_last(%arg0: f32) {
  sair.program {
    // CHECK: %[[D0:.*]] = sair.static_range 4
    %0 = sair.static_range 4 : !sair.range
    // CHECK: %[[D1:.*]] = sair.static_range 8
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[V0:.*]] = sair.copy
    %3 = sair.copy[d0:%0, d1:%1] %2 : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.proj_last[d0:%[[D0]]] of[d1:%[[D1]]] %[[V0]](d0, d1)
    %4 = sair.proj_last[d0:%0] of[d1:%1] %3(d0, d1)
      : #sair.shape<d0:range x d1:range>, f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @fby

func @fby(%arg0: f32) {
  sair.program {
    // CHECK: %[[V0:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[D0:.*]] = sair.static_range 4
    %1 = sair.static_range 4 : !sair.range
    // CHECK: %[[V1:.*]] = sair.copy[{{.*}}] %[[V0]]
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, f32>

    // CHECK: %[[D1:.*]] = sair.static_range 8
    %3 = sair.static_range 8 : !sair.range
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
