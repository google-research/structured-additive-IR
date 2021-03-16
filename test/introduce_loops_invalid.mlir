// RUN: sair-opt -split-input-file -sair-introduce-loops -verify-diagnostics %s

func @must_lower_to_map() {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{operation must be lowered to sair.map}}
    sair.map_reduce reduce[d0: %0] attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } {
      ^bb0(%arg0: index):
        sair.return
    } : #sair.shape<d0:range>, () -> ()
    sair.exit
  }
  return
}

// -----

func @missing_loop_nest_attribute() {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{missing loop_nest attribute}}
    sair.map[d0: %0] {
      ^bb0(%arg0: index):
        sair.return
    } : #sair.shape<d0:range>, () -> ()
    sair.exit
  }
  return
}

// -----

func @proj_any_must_be_eliminated() {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.map[d0:%0] attributes {
      loop_nest=[{name = "A", iter = #sair.mapping_expr<d0>}]
    } {
      ^bb0(%arg0: index):
        %2 = constant 1.0 : f32
        sair.return %2 : f32
    } : #sair.shape<d0:range>, () -> f32
    // expected-error @+1 {{sair.proj_any operations must be eliminated before introducing loops}}
    %3 = sair.proj_any of[d0:%0] %1(d0) : #sair.shape<d0:range>, f32
    sair.exit %3 : f32
  } : f32
  return
}

// -----

func @strip_mined_loop() {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{loop must not rematerialize or be strip-mined}}
    sair.map[d0:%0] attributes {
      loop_nest = [
        {name = "A", iter = #sair.mapping_expr<stripe(d0, 4)>},
        {name = "B", iter = #sair.mapping_expr<stripe(d0, 1 size 4)>}
      ]
    } {
      ^bb0(%arg0: index):
        sair.return
    } : #sair.shape<d0:range>, () -> ()
    sair.exit
  }
  return
}

// -----

func @unable_to_create_default_value() {
  %0 = sair.program {
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{unable to create a default value of type 'memref<f32>'}}
    %2 = sair.map[d0:%1] attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } {
      ^bbo(%arg0: index):
        %3 = memref.alloc() : memref<f32>
        sair.return %3 : memref<f32>
    } : #sair.shape<d0:range>, () -> memref<f32>
    %4 = sair.proj_last of[d0:%1] %2(d0) : #sair.shape<d0:range>, memref<f32>
    sair.exit %4 : memref<f32>
  } : memref<f32>
  return
}

// -----

func @proj_of_fby(%arg0: f32) {
  %0 = sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.fby %0 then[d0:%1] %3(d0) : !sair.value<d0:range, f32>
    %3 = sair.map[d0:%1] %2(d0) attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
      storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
    } {
      ^bb0(%arg1: index, %arg2: f32):
        sair.return %arg2 : f32
    } : #sair.shape<d0:range>, (f32) -> f32
    // expected-error @+1 {{insert copies between sair.fby and users located after producing loops before calling loop introduction}}
    %4 = sair.proj_last of[d0:%1] %2(d0) : #sair.shape<d0:range>, f32
    sair.exit %4 : f32
  } : f32
  return
}

// -----

func @foo() { return }

func @size_not_in_register(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.map %0 attributes {
      loop_nest = []
    } {
      ^bb0(%arg1: index):
        sair.return %arg1 : index
    } : #sair.shape<()>, (index) -> (index)
    // expected-error @+1 {{range bounds must be stored in registers}}
    %2 = sair.dyn_range %1 : !sair.range
    sair.map[d0:%2] attributes {
      loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
    } {
      ^bb0(%arg1: index):
        call @foo() : () -> ()
        sair.return
    } : #sair.shape<d0:range>, () -> ()
    sair.exit
  }
  return
}

// -----

func @placeholder() {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{placeholders must be replaced by actual dimensions before introducing loops}}
    %1 = sair.placeholder : !sair.range
    sair.map[d0:%1] attributes {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } {
      ^bb0(%arg1: index):
        sair.return
    } : #sair.shape<d0:range>, () -> ()
    sair.map[d0:%0] attributes {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } {
      ^bb0(%arg1: index):
        sair.return
    } : #sair.shape<d0:range>, () -> ()
    sair.exit
  }
  return
}
