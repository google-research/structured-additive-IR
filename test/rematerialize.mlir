// RUN: sair-opt -sair-rematerialize %s | FileCheck %s
// RUN: sair-opt -sair-rematerialize -mlir-print-op-generic %s | FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: @remat_copy
func @remat_copy(%arg0: f32) {
  sair.program {
    // CHECK: %[[INIT:.*]] = sair.from_scalar
    // GENERIC: %[[INIT:.*]] = "sair.from_scalar"
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[RANGE:.*]] = sair.static_range
    // GENERIC: %[[RANGE:.*]] = "sair.static_range"
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[RESULT:.*]] = sair.copy[d0:%[[RANGE]]] %[[INIT]]
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "A"}]
    // CHECK: !sair.value<d0:range, f32>
    //
    // CHECK: %[[VALUE:.*]] = sair.proj_any of[d0:%[[RANGE]]] %[[RESULT]](d0)
    // CHECK: #sair.shape<d0:range>
    //
    // Ensure that the access pattern has the expected use domain, and that the
    // resulting value has the right type.
    // GENERIC: "sair.copy"(%[[RANGE]], %[[INIT]])
    // GENERIC-SAME: access_pattern_array = [#sair.pattern<1>]
    // GENERIC-SAME: (!sair.range, !sair.value<(), f32>) -> !sair.value<d0:range, f32>
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>
    // CHECK: sair.copy[d0:%[[RANGE]]] %[[VALUE]]
    %3 = sair.copy[d0:%1] %2 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "remat_copy"

// CHECK-LABEL: @remat_copy_3d
func @remat_copy_3d(%arg0: f32) {
  sair.program {
    // CHECK: %[[INIT:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[RANGE1:.*]] = sair.static_range
    // CHECK: %[[RANGE2:.*]] = sair.static_range
    // CHECK: %[[RANGE3:.*]] = sair.static_range
    %1 = sair.static_range 8 : !sair.range
    %10 = sair.static_range 9 : !sair.range
    %11 = sair.static_range 10 : !sair.range
    // CHECK: %[[RESULT:.*]] = sair.copy[d0:%[[RANGE1]], d1:%[[RANGE2]], d2:%[[RANGE3]]] %[[INIT]]
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "A"},
    // CHECK:              {iter = #sair.iter<d1>, name = "B"},
    // CHECK:              {iter = #sair.iter<d2>, name = "C"}]
    // CHECK: !sair.value<d0:range x d1:range x d2:range, f32>
    //
    // CHECK: %[[VALUE:.*]] = sair.proj_any of[d0:%[[RANGE1]], d1:%[[RANGE2]], d2:%[[RANGE3]]]
    // CHECK: %[[RESULT]](d0, d1, d2)
    // CHECK: #sair.shape<d0:range x d1:range x d2:range>
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>},
                   {name = "B", iter = #sair.iter<remat>},
                   {name = "C", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>
    // CHECK: sair.copy[{{.*}}] %[[VALUE]]
    %3 = sair.copy[d0:%1, d1:%10, d2:%11] %2 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>},
                   {name = "B", iter = #sair.iter<d1>},
                   {name = "C", iter = #sair.iter<d2>}]
    } : !sair.value<d0:range x d1:range x d2:range, f32>
    sair.exit
  }
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "remat_copy_3d"

// CHECK-LABEL: @remat_copy_several
func @remat_copy_several(%arg0: f32) {
  sair.program {
    // CHECK: %[[INIT:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[RANGE1:.*]] = sair.static_range
    // CHECK: %[[RANGE2:.*]] = sair.static_range
    // CHECK: %[[RANGE3:.*]] = sair.static_range
    %range1 = sair.static_range 8 : !sair.range
    %range2 = sair.static_range 10 : !sair.range
    %range3 = sair.static_range 12 : !sair.range
    // CHECK: %[[RESULT:.*]] = sair.copy[d0:%[[RANGE1]], d1:%[[RANGE2]]] %[[INIT]]
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "X"},
    // CHECK:              {iter = #sair.iter<d1>, name = "A"}]
    // CHECK: !sair.value<d0:range x d1:range, f32>
    //
    // CHECK: %[[VALUE:.*]] = sair.proj_any of[d0:%[[RANGE1]], d1:%[[RANGE2]]] %[[RESULT]](d0, d1)
    %2 = sair.copy %0 {
      loop_nest = [{name = "X", iter = #sair.iter<remat>},
                   {name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>

    // CHECK: %[[RESULT:.*]] = sair.copy[d0:%[[RANGE2]], d1:%[[RANGE1]]] %[[VALUE]]
    // CHECK: loop_nest = [{iter = #sair.iter<d1>, name = "X"},
    // CHECK:              {iter = #sair.iter<d0>, name = "A"}]}
    // CHECK: !sair.value<d0:range x d1:range, f32>
    //
    // CHECK: sair.proj_any[d0:%[[RANGE2]]] of[d1:%[[RANGE1]]] %[[RESULT]](d0, d1)
    %3 = sair.copy[d0:%range2] %2 {
      loop_nest = [{name = "X", iter = #sair.iter<remat>},
                   {name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.copy[{{.*}}] %[[VALUE]]
    %4 = sair.copy[d0:%range1, d1:%range2, d2:%range3] %2 {
      loop_nest = [{name = "X", iter = #sair.iter<d0>},
                   {name = "A", iter = #sair.iter<d1>},
                   {name = "C", iter = #sair.iter<d2>}]
    } : !sair.value<d0:range x d1:range x d2:range, f32>
    sair.exit
  }
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "remat_copy_several"

// CHECK-LABEL: @remat_map
func @remat_map(%arg0: f32) {
  sair.program {
    // CHECK: %[[INIT:.*]] = sair.from_scalar
    // GENERIC: %[[INIT:.*]] = "sair.from_scalar"
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[RANGE:.*]] = sair.static_range
    // GENERIC: %[[RANGE:.*]] = "sair.static_range"
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[INPUT:.*]] = sair.copy[{{.*}}] %[[INIT]]
    // GENERIC: %[[INPUT:.*]] = "sair.copy"(%[[RANGE]], %[[INIT]])
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>

    // CHECK: %[[REMAT:.*]]:2 = sair.map[d0:%[[RANGE]], d1:%[[RANGE]]] %[[INPUT]](d0)
    // CHECK: loop_nest = [{iter = #sair.iter<d1>, name = "B"},
    // CHECK:              {iter = #sair.iter<d0>, name = "C"}]
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: f32):
    // CHECK: #sair.shape<d0:range x d1:range>
    //
    // CHECK: sair.proj_any[d0:%[[RANGE]]] of[d1:%[[RANGE]]] %[[REMAT]]#0(d0, d1)
    // CHECK: %[[RESULT:.*]] = sair.proj_any[d0:%[[RANGE]]] of[d1:%[[RANGE]]] %[[REMAT]]#1(d0, d1)
    //
    // Ensure that the access pattern has the expected use domain, and that the
    // resulting value has the right type.
    // GENERIC: "sair.map"(%[[RANGE]], %[[RANGE]], %[[INPUT]])
    // GENERIC: access_pattern_array = [#sair.pattern<2 : d0>]
    // GENERIC: (!sair.range, !sair.range, !sair.value<d0:range, f32>) ->
    // GENERIC-SAME: (!sair.value<d0:range x d1:range, f32>,
    // GENERIC-SAME:  !sair.value<d0:range x d1:range, f32>)
    %3:2 = sair.map[d0:%1] %2(d0) attributes {
      loop_nest = [{name = "B", iter = #sair.iter<remat>},
                   {name = "C", iter = #sair.iter<d0>}]
    } {
    ^bb0(%idx: index, %in: f32):
      %4 = addf %in, %in : f32
      sair.return %4, %in : f32, f32
    } : #sair.shape<d0:range>, (f32) -> (f32, f32)

    // CHECK: sair.map[{{.*}}] %[[RESULT]]
    // GENERIC: "sair.map"
    sair.map[d0:%1] %3#1(d0) attributes {
      loop_nest = [{name = "B", iter = #sair.iter<d0>}]
    } {
    ^bb0(%idx: index, %in: f32):
      %4 = mulf %in, %in : f32
      sair.return %4 : f32
    } : #sair.shape<d0:range>, (f32) -> (f32)
    sair.exit
  }
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "remat_map"

// CHECK-LABEL: @remat_map_reduce
func @remat_map_reduce(%arg0: f32) {
  sair.program {
    // CHECK: %[[SCALAR:.*]] = sair.from_scalar
    // GENERIC: %[[SCALAR:.*]] = "sair.from_scalar"
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[RANGE:.*]] = sair.static_range
    // GENERIC: %[[RANGE:.*]] = "sair.static_range"
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[INIT:.*]] = sair.copy[{{.*}}] %[[SCALAR]]
    // GENERIC: %[[INIT:.*]] = "sair.copy"(%[[RANGE]], %[[SCALAR]])
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, f32>
    // CHECK: %[[INPUT:.*]] = sair.copy[{{.*}}] %[[SCALAR]]
    // GENERIC: %[[INPUT:.*]] = "sair.copy"(%[[RANGE]], %[[RANGE]], %[[SCALAR]])
    %3 = sair.copy[d0:%1, d1:%1] %0 : !sair.value<d0:range x d1:range, f32>

    // CHECK: %[[REMAT:.*]] = sair.map_reduce[d0:%[[RANGE]], d1:%[[RANGE]]] %[[INIT]](d0)
    // CHECK:                          reduce[d2:%[[RANGE]]] %[[INPUT]](d0, d2)
    // CHECK: loop_nest = [{iter = #sair.iter<d1>, name = "A"},
    // CHECK:              {iter = #sair.iter<d0>, name = "B"},
    // CHECK:              {iter = #sair.iter<d2>, name = "C"}]
    // CHECK: ^{{.*}}(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: f32, %{{.*}}: f32):
    // CHECK: #sair.shape<d0:range x d1:range x d2:range>
    //
    // CHECK: %[[RESULT:.*]] = sair.proj_any[d0:%[[RANGE]]] of[d1:%[[RANGE]]] %[[REMAT]](d0, d1)
    //
    // Ensure that the access pattern has the expected use domain, and that the
    // resulting value has the right type.
    // GENERIC: "sair.map_reduce"(%[[RANGE]], %[[RANGE]], %[[RANGE]], %[[INIT]], %[[INPUT]])
    // GENERIC: access_pattern_array = [#sair.pattern<3 : d0>,
    // GENERIC-SAME:                    #sair.pattern<3 : d0, d2>]
    // GENERIC-SAME: shape = #sair.shape<d0:range x d1:range x d2:range>
    // GENERIC:      (!sair.range, !sair.range, !sair.range,
    // GENERIC-SAME: !sair.value<d0:range, f32>, !sair.value<d0:range x d1:range, f32>) ->
    // GENERIC-SAME: !sair.value<d0:range x d1:range, f32>
    %4 = sair.map_reduce[d0:%1] %2(d0) reduce[d1:%1] %3(d0, d1) attributes {
      loop_nest = [{name = "A", iter = #sair.iter<remat>},
                   {name = "B", iter = #sair.iter<d0>},
                   {name = "C", iter = #sair.iter<d1>}]
    } {
    ^bb0(%idx1: index, %idx2: index, %left: f32, %right: f32):
      %5 = addf %left, %right : f32
      sair.return %5 : f32
    } : #sair.shape<d0:range x d1:range>, (f32) -> (f32)

    // CHECK: sair.map[{{.*}}] %[[RESULT]]
    sair.map[d0:%1] %4(d0) attributes {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } {
    ^bb0(%idx: index, %in: f32):
      %5 = mulf %in, %in : f32
      sair.return %5 : f32
    } : #sair.shape<d0:range>, (f32) -> (f32)
    sair.exit
  }
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "remat_map_reduce"

// CHECK-LABEL: @remat_copy_dependent
func @remat_copy_dependent(%arg0: f32, %arg1: index) {
  sair.program {
    // CHECK: %[[SCALAR:.*]] = sair.from_scalar
    // GENERIC: %[[SCALAR:.*]] = "sair.from_scalar"
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.from_scalar %arg1 : !sair.value<(), index>

    // CHECK: %[[STATIC_RANGE:.*]] = sair.static_range
    // GENERIC: %[[STATIC_RANGE:.*]] = "sair.static_range"
    // GENERIC: "sair.copy"
    %2 = sair.static_range 8 : !sair.range
    %3 = sair.copy[d0:%2] %1 : !sair.value<d0:range, index>

    // CHECK: %[[DYNAMIC_RANGE:.*]] = sair.dyn_range[d0:%[[STATIC_RANGE]]] %{{.*}}
    // GENERIC: %[[DYNAMIC_RANGE:.*]] = "sair.dyn_range"(%[[STATIC_RANGE]]
    %4 = sair.dyn_range[d0:%2] %3(d0) : !sair.range<d0:range>

    // CHECK: %[[REMAT:.*]] = sair.copy[d0:%[[STATIC_RANGE]], d1:%[[DYNAMIC_RANGE]]] %[[SCALAR]]
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "A"},
    // CHECK:              {iter = #sair.iter<d1>, name = "B"}]
    // CHECK: !sair.value<d0:range x d1:range(d0), f32>
    //
    // CHECK: %[[RESULT:.*]] = sair.proj_any of[d0:%[[STATIC_RANGE]], d1:%[[DYNAMIC_RANGE]]] %[[REMAT]](d0, d1)
    // CHECK: #sair.shape<d0:range x d1:range(d0)>
    //
    // Ensure that the access pattern has the expected use domain, and that the
    // resulting value has the right type.
    // GENERIC: "sair.copy"(%[[STATIC_RANGE]], %[[DYNAMIC_RANGE]], %[[SCALAR]])
    // GENERIC-SAME: access_pattern_array = [#sair.pattern<2>]
    // GENERIC-SAME: (!sair.range, !sair.range<d0:range>, !sair.value<(), f32>) ->
    // GENERIC-SAME: !sair.value<d0:range x d1:range(d0), f32>
    %5 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>},
                   {name = "B", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>

    // CHECK: sair.copy[{{.*}}] %[[RESULT]]
    // GENERIC: "sair.copy"
    %6 = sair.copy[d0:%2, d1:%4] %5 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>},
                   {name = "B", iter = #sair.iter<d1>}]
    } : !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  return
}
// In the generic form, the function (symbol) name is an attribute and is
// printed after the body region.
// GENERIC-LABEL: sym_name = "remat_copy_dependent"

// CHECK-LABEL: @remat_copy_dependent_partial
func @remat_copy_dependent_partial(%arg0: f32, %arg1: index) {
  sair.program {
    // CHECK: %[[SCALAR:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.from_scalar %arg1 : !sair.value<(), index>

    // CHECK: %[[STATIC_RANGE:.*]] = sair.static_range
    %2 = sair.static_range 8 : !sair.range
    %3 = sair.copy[d0:%2] %1 : !sair.value<d0:range, index>

    // CHECK: %[[DYNAMIC_RANGE:.*]] = sair.dyn_range[d0:%[[STATIC_RANGE]]] %{{.*}}
    %4 = sair.dyn_range[d0:%2] %3(d0) : !sair.range<d0:range>

    // CHECK: %[[REMAT:.*]] = sair.copy[d0:%[[STATIC_RANGE]], d1:%[[DYNAMIC_RANGE]]] %[[SCALAR]]
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "A"},
    // CHECK:              {iter = #sair.iter<d1>, name = "B"}]
    // CHECK: !sair.value<d0:range x d1:range(d0), f32>
    //
    // CHECK: %[[RESULT:.*]] = sair.proj_any[d0:%[[STATIC_RANGE]]] of[d1:%[[DYNAMIC_RANGE]]] %[[REMAT]](d0, d1)
    // CHECK: #sair.shape<d0:range x d1:range(d0)>
    %5 = sair.copy[d0:%2] %0 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>},
                   {name = "B", iter = #sair.iter<remat>}]
    } : !sair.value<d0:range, f32>

    // CHECK: sair.copy[{{.*}}] %[[RESULT]]
    %6 = sair.copy[d0:%2, d1:%4] %5(d0) {
      loop_nest = [{name = "A", iter = #sair.iter<d0>},
                   {name = "B", iter = #sair.iter<d1>}]
    } : !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  return
}
