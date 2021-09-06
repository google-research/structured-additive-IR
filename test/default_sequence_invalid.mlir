// RUN: sair-opt -sair-assign-default-sequence -split-input-file -verify-diagnostics %s

func @no_instance(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{Sair operations must have a single instance to assign a default sequence}}
    %1 = sair.copy %0 : !sair.value<(), f32>
    sair.exit
  }
  return
}
