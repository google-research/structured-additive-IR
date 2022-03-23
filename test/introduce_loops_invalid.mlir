// RUN: sair-opt -split-input-file -sair-introduce-loops -verify-diagnostics %s

func.func @must_lower_to_map() {
  sair.program {
    %0 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    // expected-error @+1 {{operation must be lowered to sair.map}}
    sair.map_reduce reduce[d0: %0] attributes {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
      }]
    } {
      ^bb0(%arg0: index):
        sair.return
    } : #sair.shape<d0:static_range<8>>, () -> ()
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @missing_loop_nest_attribute() {
  sair.program {
    %0 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    // expected-error @+1 {{missing loop_nest attribute}}
    sair.map[d0: %0] attributes {
      instances = [{}]
    } {
      ^bb0(%arg0: index):
        sair.return
    } : #sair.shape<d0:static_range<8>>, () -> ()
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @proj_any_must_be_eliminated() {
  sair.program {
    %0 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    %1 = sair.map[d0:%0] attributes {
      instances = [{loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]}]
    } {
      ^bb0(%arg0: index):
        %2 = arith.constant 1.0 : f32
        sair.return %2 : f32
    } : #sair.shape<d0:static_range<8>>, () -> f32
    // expected-error @+1 {{sair.proj_any operations must be eliminated before introducing loops}}
    %3 = sair.proj_any of[d0:%0] %1(d0) { instances = [{}] } : #sair.shape<d0:static_range<8>>, f32
    sair.exit %3 { instances = [{}] } : f32
  } : f32
  return
}

// -----

func.func @strip_mined_loop() {
  sair.program {
    %0 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    // expected-error @+1 {{loop must not rematerialize or be strip-mined}}
    sair.map[d0:%0] attributes {
      instances = [{
        loop_nest = [
          {name = "A", iter = #sair.mapping_expr<stripe(d0, [4])>},
          {name = "B", iter = #sair.mapping_expr<stripe(d0, [4, 1])>}
        ]
      }]
    } {
      ^bb0(%arg0: index):
        sair.return
    } : #sair.shape<d0:static_range<8>>, () -> ()
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @unable_to_create_default_value() {
  %0 = sair.program {
    %1 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    // expected-error @+1 {{unable to create a default value of type 'memref<f32>'}}
    %2 = sair.map[d0:%1] attributes {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]
      }]
    } {
      ^bbo(%arg0: index):
        %3 = memref.alloc() : memref<f32>
        sair.return %3 : memref<f32>
    } : #sair.shape<d0:static_range<8>>, () -> memref<f32>
    %4 = sair.proj_last of[d0:%1] %2(d0) { instances = [{}] } : #sair.shape<d0:static_range<8>>, memref<f32>
    sair.exit %4 { instances = [{}] } : memref<f32>
  } : memref<f32>
  return
}

// -----

func.func @proj_of_fby(%arg0: f32) {
  %0 = sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    %1 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    %2 = sair.fby %0 then[d0:%1] %3(d0) { instances = [{}] } : !sair.value<d0:static_range<8>, f32>
    %3 = sair.map[d0:%1] %2(d0) attributes {
      instances = [{
        loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}],
        storage = [{space = "register", layout = #sair.named_mapping<[] -> ()>}]
      }]
    } {
      ^bb0(%arg1: index, %arg2: f32):
        sair.return %arg2 : f32
    } : #sair.shape<d0:static_range<8>>, (f32) -> f32
    // expected-error @+1 {{insert copies between sair.fby and users located after producing loops before calling loop introduction}}
    %4 = sair.proj_last of[d0:%1] %2(d0) { instances = [{}] } : #sair.shape<d0:static_range<8>>, f32
    sair.exit %4 { instances = [{}] } : f32
  } : f32
  return
}

// -----

func.func @foo() { return }

func.func @size_not_in_register(%arg0: index) {
  sair.program {
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), index>
    %1 = sair.map %0 attributes {
      instances = [{loop_nest = []}]
    } {
      ^bb0(%arg1: index):
        sair.return %arg1 : index
    } : #sair.shape<()>, (index) -> (index)
    // expected-error @+1 {{range bounds must be stored in registers}}
    %2 = sair.dyn_range %1 { instances = [{}] } : !sair.dyn_range
    sair.map[d0:%2] attributes {
      instances = [{loop_nest = [{name = "A", iter = #sair.mapping_expr<d0>}]}]
    } {
      ^bb0(%arg1: index):
        call @foo() : () -> ()
        sair.return
    } : #sair.shape<d0:dyn_range>, () -> ()
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @placeholder() {
  sair.program {
    %0 = sair.static_range { instances = [{}] } : !sair.static_range<8>
    // expected-error @+1 {{placeholders must be replaced by actual dimensions before introducing loops}}
    %1 = sair.placeholder { instances = [{}] } : !sair.static_range<8>
    sair.map[d0:%1] attributes {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } {
      ^bb0(%arg1: index):
        sair.return
    } : #sair.shape<d0:static_range<8>>, () -> ()
    sair.map[d0:%0] attributes {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } {
      ^bb0(%arg1: index):
        sair.return
    } : #sair.shape<d0:static_range<8>>, () -> ()
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func.func @copies(%arg0: f32) {
  sair.program {
    // expected-error @+1 {{operations must have exactly one instance when introducing loops}}
    sair.from_scalar %arg0 {
      copies = [[{sequence = 0}]]
    } : !sair.value<(), f32>
    sair.exit
  }
  return
}
