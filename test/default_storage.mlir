// RUN: sair-opt %s -sair-assign-default-storage | FileCheck %s

// CHECK-LABEL: @memory_space_is_set
func @memory_space_is_set() {
  sair.program {
    // CHECK: %{{.*}} = sair.map attributes {
    // CHECK: storage = [{layout = #sair.named_mapping<[] -> ()>, space = "register"}]
    %0 = sair.map attributes { loop_nest = [] } {
      ^bb0:
        %c1 = constant 1.0 : f32
        sair.return %c1 : f32
    } : #sair.shape<()>, () -> f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @preserve_memory_space
func @preserve_memory_space() {
  sair.program {
    // CHECK: %{{.*}} = sair.map attributes {
    // CHECK: storage = [{layout = #sair.named_mapping<[] -> ()>, name = "A", space = "memory"}]
    %0 = sair.map attributes {
      loop_nest = [],
      storage =[{space = "memory", name = "A", layout = #sair.named_mapping<[] -> ()>}]
    } {
      ^bb0:
        %c1 = constant 1.0 : f32
        sair.return %c1 : f32
    } : #sair.shape<()>, () -> f32
    sair.exit
  }
  return
}

// CHECK-LABEL: @multi_dim
func @multi_dim(%arg0: f32, %arg1: memref<8x8xf32>) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %memref = sair.from_scalar %arg1 : !sair.value<(), memref<8x8xf32>>
    // CHECK: %[[V1:.*]] = sair.copy[d0:%{{.*}}, d1:%{{.*}}] %[[V0]]
    %2 = sair.copy[d0:%0, d1:%0] %1 {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        {name = "loopB", iter = #sair.mapping_expr<d1>}
      ]
      // CHECK: storage = [{
      // CHECK:   layout = #sair.named_mapping<[d0:"loopA", d1:"loopB"] -> (d0, d1)>
      // CHECK:   name = "buffer_0", space = "memory"
      // CHECK: }]
    } : !sair.value<d0:range x d1:range, f32>
    // CHECK: sair.copy[d0:%{{.*}}, d1:%{{.*}}] %[[V1]](d0, d1)
    %3 = sair.copy[d0:%0, d1:%0] %2(d0, d1) {
      loop_nest = [
        {name = "loopC", iter = #sair.mapping_expr<d0>},
        {name = "loopD", iter = #sair.mapping_expr<d1>}
      ]
      // CHECK: storage = [{
      // CHECK:   layout = #sair.named_mapping<[d0:"loopC", d1:"loopD"] -> (d0, d1)>
      // CHECK:   name = "out", space = "memory"
      // CHECK: }]
    } : !sair.value<d0:range x d1:range, f32>
    sair.to_memref %memref memref[d0:%0, d1:%0] %3(d0, d1) {
      buffer_name = "out"
    }  : #sair.shape<d0:range x d1:range>, memref<8x8xf32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @to_memref_proj_fby
func @to_memref_proj_fby(%arg0: f32, %arg1: memref<f32>) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // CHECK: %[[V0:.*]] = sair.from_scalar %{{.*}} : !sair.value<(), f32>
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %2 = sair.from_scalar %arg1 : !sair.value<(), memref<f32>>

    // CHECK: sair.copy %[[V0]]
    // CHECK:   storage = [{
    // CHECK:     layout = #sair.named_mapping<[] -> ()>
    // CHECK:     name = "out", space = "memory"
    // CHECK:   }]
    %3 = sair.copy %1 {loop_nest = []} : !sair.value<(), f32>
    // CHECK: %[[V1:.*]] = sair.fby
    %4 = sair.fby %3 then[d0:%0] %5(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.copy[d0:%{{.*}}] %[[V1]](d0)
    // CHECK:   storage = [{
    // CHECK:     layout = #sair.named_mapping<[] -> ()>
    // CHECK:     name = "out", space = "memory"
    // CHECK:   }]
    %5 = sair.copy[d0:%0] %4(d0) {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    %6 = sair.proj_last of[d0:%0] %4(d0) : #sair.shape<d0:range>, f32
    sair.to_memref %2 memref %6 {
      buffer_name = "out"
    } : #sair.shape<()>, memref<f32>
    sair.exit
  }
  return
}
