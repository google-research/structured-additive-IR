// RUN: sair-opt -sair-lower-proj-any -split-input-file -verify-diagnostics %s

func @source_not_normalized(%arg0: f32) {
  %n = arith.constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.dyn_range %sn : !sair.dyn_range
    // expected-error @+1 {{operation iteration space not normalized}}
    %2 = sair.copy %0 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<none>}]
      }]
    } : !sair.value<(), f32>
    %3 = sair.proj_any of %2 : #sair.shape<()>, f32
    %4 = sair.copy[d0:%1] %3 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    sair.exit
  }
  return
}

// -----

func @result_not_normalized(%arg0: f32) {
  %n = arith.constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.dyn_range %sn : !sair.dyn_range
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    %3 = sair.proj_any of[d0:%1] %2(d0) : #sair.shape<d0:dyn_range>, f32
    // expected-error @+1 {{operation iteration space not normalized}}
    %4 = sair.copy %3 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<none>}]
      }]
    } : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @cannot_lower(%arg0: f32) {
  %n = arith.constant 8 : index
  sair.program {
    %sn = sair.from_scalar %n : !sair.value<(), index>
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.dyn_range %sn : !sair.dyn_range
    %2 = sair.copy[d0:%1] %0 {
      instances = [{
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    // expected-error @+1 {{cannot lower operation to proj_last on scalars}}
    %3 = sair.proj_any[d0:%1] of %2(d0) : #sair.shape<d0:dyn_range>, f32
    %4 = sair.copy[d0:%1] %3(d0) {
      instances = [{
        loop_nest = [{name = "loopB", iter = #sair.mapping_expr<d0>}]
      }]
    } : !sair.value<d0:dyn_range, f32>
    sair.exit
  }
  return
}

// -----

func @copies(%arg0: f32) {
  sair.program {
    // expected-error @+1 {{copies must be materialized before lowering proj_any operations}}
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.proj_any of %0 {
      copies = [[{sequence = 0}]]
    } : #sair.shape<()>, f32
    sair.exit
  }
  return
}

// -----

func @instances(%arg0: f32) {
  sair.program {
    // expected-error @below {{instances must be materialized before lowering proj_any operations}}
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.proj_any of %0 {
      instances = [{}, {}]
    } : #sair.shape<()>, f32
    sair.exit
  }
  return
}
