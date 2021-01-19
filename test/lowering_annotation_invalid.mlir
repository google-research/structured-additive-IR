// RUN: sair-opt -split-input-file -verify-diagnostics %s

func @array_wrong_size() {
  sair.program {
    // expected-error @+1 {{wrong number of entries for the memory_space attribute}}
    sair.map attributes { memory_space = [0] } {
      ^bb0:
        sair.return
    } : #sair.shape<()>, () -> ()
    sair.exit
  }
  return
}

// -----

func @unexpected_memory_space() {
  sair.program {
    // expected-error @+1 {{unexpected memory space}}
    sair.map attributes { memory_space = [-1] } {
      ^bb0:
        %c1 = constant 1.0 : f32
        sair.return %c1 : f32
    } : #sair.shape<()>, () -> (f32)
    sair.exit
  }
  return
}

// -----

func @index_in_ram(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    // expected-error @+1 {{index variables cannot be allocated in memory}}
    %1 = sair.copy %0 {memory_space = [1]} : !sair.value<(), index>
    sair.exit
  }
  return
}
