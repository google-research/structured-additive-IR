// RUN: sair-opt -sair-assign-default-loop-nest %s | FileCheck %s

func.func @default_loop_nest(%arg0: f32) {
  sair.program {
    %0 = sair.static_range : !sair.static_range<16>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // CHECK: sair.copy[d0:%{{.*}}, d1:%{{.*}}] %{{.*}} {instances = [{loop_nest = [
    // CHECK:   {iter = #sair.mapping_expr<d0>, name = "{{.*}}"},
    // CHECK:   {iter = #sair.mapping_expr<d1>, name = "{{.*}}"}
    // CHECK: ]}]} : !sair.value<d0:static_range<16> x d1:static_range<16>, f32>
    sair.copy[d0:%0, d1:%0] %1 {
      instances = [{}]
    } : !sair.value<d0:static_range<16> x d1:static_range<16>, f32>
    sair.exit
  }
  func.return
}
