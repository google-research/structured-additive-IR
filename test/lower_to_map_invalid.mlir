// RUN: sair-opt -split-input-file -sair-lower-to-map -verify-diagnostics %s

func @no_expansion_pattern(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{no target expansion pattern specified}}
    %1 = sair.copy %0 : !sair.value<(), f32>
    sair.exit
  }
  return
}
