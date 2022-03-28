// RUN: sair-opt -sair-materialize-instances %s | FileCheck %s

// CHECK-LABEL: @instances
func.func @instances(%arg0: f32) {
  sair.program {
    // CHECK: %[[SCALAR0:.*]] = sair.from_scalar
    // CHECK: %[[SCALAR1:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 { instances = [
      {operands = [#sair.instance<0>]},
      {operands = [#sair.instance<0>]}] } : !sair.value<(), f32>
    // CHECK: %[[RANGE:.*]] = sair.static_range
    %1 = sair.static_range { instances = [{operands = []}] } : !sair.static_range<4>
    // CHECK: sair.copy[d0:%[[RANGE]]] %[[SCALAR1]]
    // CHECK-COUNT-2: #sair.instance<0>
    %2 = sair.copy[d0:%1] %0 { instances = [{operands = [#sair.instance<0>, #sair.instance<1>]}] }  : !sair.value<d0:static_range<4>, f32>
    sair.exit { instances = [{operands = []}] }
  }
  func.return
}

// CHECK-LABEL: @instances_and_copies
func.func @instances_and_copies(%arg0: f32) {
  sair.program {
    // CHECK: %[[SCALAR0:.*]] = sair.from_scalar
    // CHECK: %[[SCALAR1:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 {
      instances = [
        {operands = [#sair.instance<0>]},
        {operands = [#sair.instance<0>]}],
      copies = [[
        {copy_of = #sair.instance<0>},
        {copy_of = #sair.copy<0>}]] }
    : !sair.value<(), f32>
    // CHECK: %[[COPY0:.*]] = sair.copy %[[SCALAR0]]
    // CHECK-SAME: #sair.instance<0>
    // CHECK: %[[COPY1:.*]] = sair.copy %[[COPY0]]
    // CHECK-SAME: #sair.instance<0>

    %1 = sair.static_range {
      instances = [{operands = []}] } : !sair.static_range<4>

    // CHECK: sair.copy[d0:%{{.*}}] %[[COPY1]]
    // CHECK-COUNT-2: #sair.instance<0>
    %2 = sair.copy[d0:%1] %0 {
      instances = [
        {operands = [#sair.instance<0>, #sair.copy<1>]}] }
      : !sair.value<d0:static_range<4>, f32>
    sair.exit { instances = [{operands = []}] }
  }
  func.return
}

// CHECK-LABEL: @map_multi_result
func.func @map_multi_result(%arg0: f32) {
  sair.program {
    // CHECK: %[[SCALAR:.*]] = sair.from_scalar
    %0 = sair.from_scalar %arg0 {
      instances = [{operands = [#sair.instance<0>]}] }
      : !sair.value<(), f32>

    // CHECK: %[[RANGE0:.*]] = sair.static_range
    // CHECK: %[[RANGE1:.*]] = sair.static_range
    // CHECK: %[[RANGE2:.*]] = sair.static_range
    %1 = sair.static_range {
      instances = [
        {operands = []},
        {operands = []},
        {operands = []}] } : !sair.static_range<4>

    // CHECK: sair.copy[d0:%[[RANGE0]], d1:%[[RANGE1]]] %[[SCALAR]]
    // CHECK-COUNT-3: #sair.instance<0>
    %2 = sair.copy[d0:%1, d1:%1] %0 {
      instances = [{
        operands = [
          #sair.instance<0>,
          #sair.instance<1>,
          #sair.instance<0>]
      }]} : !sair.value<d0:static_range<4> x d1:static_range<4>, f32>

    // Also make sure use-def cycles are not an issue with op removal by using
    // cyclic fby.
    // CHECK: %[[FBY:.*]] = sair.fby[d0:%[[RANGE1]], d1:%[[RANGE0]]] %{{.*}}(d1, d0)
    // CHECK:                   then[d2:%[[RANGE2]]] %[[MAP_COPY:.*]](d1, d0, d2)
    %3 = sair.fby[d0:%1, d1:%1] %2(d1,d0) then[d2:%1] %4#1(d1,d0,d2) {
      instances = [{
        operands = [
          #sair.instance<1>,
          #sair.instance<0>,
          #sair.instance<2>,
          #sair.instance<0>,
          #sair.copy<0>
        ]
      }]} : !sair.value<d0:static_range<4> x d1:static_range<4> x d2:static_range<4>, f32>

    // CHECK: %[[MAP0:.*]]:2 = sair.map[d0:%[[RANGE0]], d1:%[[RANGE0]], d2:%[[RANGE0]]] %[[FBY]]
    // CHECK-COUNT-4: #sair.instance<0>
    // CHECK: %[[MAP1:.*]]:2 = sair.map[d0:%[[RANGE2]], d1:%[[RANGE1]], d2:%[[RANGE0]]] %[[FBY]]
    // CHECK-COUNT-4: #sair.instance<0>
    // CHECK-NOT: sair.copy %{{.*}} %[[MAP0]]
    // CHECK: %[[MAP_COPY]] = sair.copy[d0:%[[RANGE2]], d1:%[[RANGE1]], d2:%[[RANGE0]]] %[[MAP1]]
    %4:2 = sair.map[d0:%1, d1:%1, d2:%1] %3(d0, d1, d2) attributes {
      instances = [{
        operands = [
          #sair.instance<0>,
          #sair.instance<0>,
          #sair.instance<0>,
          #sair.instance<0>
        ]
      }, {
        operands = [
          #sair.instance<2>,
          #sair.instance<1>,
          #sair.instance<0>,
          #sair.instance<0>
        ]
      }],
      copies = [
        [{copy_of = #sair.instance<0>}],
        [{copy_of = #sair.instance<1>}]
      ]
    } {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: f32):
      sair.return %arg4, %arg4 : f32, f32
    } : #sair.shape<d0:static_range<4> x d1:static_range<4> x d2:static_range<4>>, (f32) -> (f32, f32)
    sair.exit { instances = [{operands = []}] }
  }
  func.return
}

// CHECK-LABEL: @erase_zero_instance
func.func @erase_zero_instance(%arg0: f32) {
  sair.program {
    // CHECK-NOT: sair.from_scalar
    sair.from_scalar %arg0 { instances = [] } : !sair.value<(), f32>
    sair.exit { instances = [{operands = []}] }
  }
  func.return
}

