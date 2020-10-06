// RUN: sair-opt -sair-rematerialize %s | sair-opt | FileCheck %s

// CHECK-LABEL: @remat_copy
func @remat_copy(%arg0: f32) {
  sair.program {
    // CHECK: %[[INIT:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: %[[RANGE:.*]] = sair.static_range
    %1 = sair.static_range 8 : !sair.range
    // CHECK: %[[RESULT:.*]] = sair.copy[d0:%[[RANGE]]] %[[INIT]]
    // CHECK: loop_nest = [{iter = #sair.iter<d0>, name = "A"}]
    // CHECK: !sair.value<d0:range, f32>
    //
    // CHECK: %[[VALUE:.*]] = sair.proj_any of[d0:%[[RANGE]]] %[[RESULT]](d0)
    // CHECK: #sair.shape<d0:range>
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

