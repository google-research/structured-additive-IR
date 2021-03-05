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
