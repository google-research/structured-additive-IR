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
      storage =[{space = "memory", name = "A"}]
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

// CHECK-LABEL: @propagate_storage
func @propagate_storage(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.fby %0 then[d0:%1] %3(d0) : !sair.value<d0:range, f32>
    // CHECK: sair.copy
    // CHECK: storage = [{layout = #sair.named_mapping<[] -> ()>, space = "register"}]
    %3 = sair.copy[d0:%1] %2(d0) {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @non_rectangular
func @non_rectangular_shape(%arg0: f32, %arg1: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.from_scalar %arg1 : !sair.value<(), index>
    %2 = sair.static_range 8 : !sair.range
    %3 = sair.dyn_range[d0:%2] %1 : !sair.range<d0:range>
    // CHECK: storage = [{
    // CHECK:   layout = #sair.named_mapping<[d0:"loopB"] -> (d0)>,
    // CHECK:   name = "[[BUFFER:.*]]", space = "memory"
    // CHECK: }]
    %4 = sair.copy[d0:%2, d1:%3] %0 {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        {name = "loopB", iter = #sair.mapping_expr<d1>}
      ]
    } : !sair.value<d0:range x d1:range(d0), f32>
    %5 = sair.copy[d0:%2, d1:%3] %4(d0, d1) {
      loop_nest = [
        {name = "loopA", iter = #sair.mapping_expr<d0>},
        {name = "loopC", iter = #sair.mapping_expr<d1>}
      ]
    } : !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  return
}

// CHECK-LABEL: @buffer_reuse
func @buffer_reuse(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 4 : !sair.range
    %2 = sair.static_range 8 : !sair.range

    // First use.
    // CHECK: layout = #sair.named_mapping<[d0:"loopA"] -> (none, d0)>
    %3 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}],
      storage = [{name = "buffer", space = "memory"}]
    } : !sair.value<d0:range, f32>
    %4 = sair.copy[d0:%1] %3(d0){
      loop_nest = [{name = "loopB", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>

    // Second use
    // CHECK: layout = #sair.named_mapping<[d0:"loopC"] -> (d0, none)>
    %5 = sair.copy[d0:%2] %0 {
      loop_nest = [{name = "loopC", iter = #sair.mapping_expr<d0>}],
      storage = [{name = "buffer", space = "memory"}]
    } : !sair.value<d0:range, f32>
    %6 = sair.copy[d0:%2] %5(d0){
      loop_nest = [{name = "loopD", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}
