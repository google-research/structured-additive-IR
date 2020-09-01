// RUN: sair-opt -sair-assign-default-loop-nest %s | FileCheck %s

func @default_loop_nest(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 16 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy[d0:%{{.*}}, d1:%{{.*}}] %{{.*}} {loop_nest = [
    // CHECK:   {iter = #sair.iter<d0>, name = "{{.*}}"},
    // CHECK:   {iter = #sair.iter<d1>, name = "{{.*}}"}
    // CHECK: ]} : !sair.value<d0:range x d1:range, f32>
    sair.copy[d0:%0, d1:%0] %1 : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}
