// RUN: sair-opt -sair-lower-proj-any -split-input-file -verify-diagnostics %s

func @source_not_normalized(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{operation iteration space not normalized}}
    %2 = sair.copy %0 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<none>}]
    } : !sair.value<(), f32>
    %3 = sair.proj_any of %2 : #sair.shape<()>, f32
    %4 = sair.copy[d0:%1] %3 {
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @result_not_normalized(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    %3 = sair.proj_any of[d0:%1] %2(d0) : #sair.shape<d0:range>, f32
    // expected-error @+1 {{operation iteration space not normalized}}
    %4 = sair.copy %3 {
        loop_nest = [{name = "loopA", iter = #sair.mapping_expr<none>}]
    } : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @cannot_lower(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "loopA", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    // expected-error @+1 {{cannot lower operation to proj_last on scalars}}
    %3 = sair.proj_any[d0:%1] of %2(d0) : #sair.shape<d0:range>, f32
    %4 = sair.copy[d0:%1] %3(d0) {
        loop_nest = [{name = "loopB", iter = #sair.mapping_expr<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}