// RUN: sair-opt %s -sair-lower-proj-any | FileCheck %s

// CHECK-LABEL: @eliminate
func.func @eliminate(%arg0: f32) {
  %n = arith.constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.dyn_range %sn : !sair.dyn_range
    // CHECK: %[[V1:.*]] = sair.copy[d0:%{{.*}}] %[[V0]]
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    %3 = sair.proj_any of[d0:%1] %2(d0) : #sair.shape<d0:dyn_range>, f32
    // CHECK: sair.copy[d0:%{{.*}}] %[[V1]](d0)
    %4 = sair.copy[d0:%1] %3 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @convert_to_proj_last
func.func @convert_to_proj_last(%arg0: f32) {
  %n = arith.constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.dyn_range %sn : !sair.dyn_range
    // CHECK: %[[V1:.*]] = sair.copy[d0:%{{.*}}] %[[V0]]
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    // CHECK: %[[V2:.*]] = sair.proj_last of[d0:%{{.*}}] %[[V1]](d0)
    // CHECK:   instances = [{}, {}]
    // CHECK:   : #sair.shape<d0:dyn_range>, f32
    %3 = sair.proj_any of[d0:%1] %2(d0) { instances = [{}, {}] } : #sair.shape<d0:dyn_range>, f32
    // CHECK: sair.copy[d0:%{{.*}}] %[[V2]]
    %4 = sair.copy[d0:%1] %3 {
      instances = [{
        loop_nest = [{name = "loopB", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    sair.exit
  }
  return
}
