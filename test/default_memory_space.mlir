// RUN: sair-opt %s -sair-assign-default-memory-space | FileCheck %s

// CHECK-LABEL: @memory_space_is_set
func @memory_space_is_set() {
  sair.program {
    // CHECK: %{{.*}} = sair.map attributes {memory_space = [{{[0-9]+}}]}
    %0 = sair.map {
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
    // CHECK: %{{.*}} = sair.map attributes {memory_space = [1]}
    %0 = sair.map attributes {memory_space=[1]} {
      ^bb0:
        %c1 = constant 1.0 : f32
        sair.return %c1 : f32
    } : #sair.shape<()>, () -> f32
    sair.exit
  }
  return
}
