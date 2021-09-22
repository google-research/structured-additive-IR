// RUN: sair-opt -sair-materialize-instances -split-input-file -verify-diagnostics

func @no_instances(%arg0: f32) {
  sair.program {
    // expected-error@below {{expected ops to have instances}}
    sair.exit
  }
  return
}

// -----

func @user_of_zero_instance(%arg0: f32) {
  sair.program {
    // expected-error@below {{operation has zero instances but its results are in use}}
    %0 = sair.from_scalar %arg0 { instances = [] } : !sair.value<(), f32>
    // expected-note@below {{user found}}
    sair.copy %0 : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @no_operands(%arg0: f32) {
  sair.program {
    // expected-error@below {{expected 'operands' field of 'instances' to be specified}}
    %0 = sair.from_scalar %arg0 { instances = [{}] } : !sair.value<(), f32>
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func @no_operands_copies(%arg0: f32) {
  sair.program {
    // expected-error@below {{expected the source of copy to be specified}}
    %0 = sair.from_scalar %arg0 {
      instances = [
        {operands = [#sair.instance<0>]},
        {operands = [#sair.instance<0>]}],
      copies = [[{}, {}]] } : !sair.value<(), f32>
    sair.exit { instances = [{}] }
  }
  return
}

// -----

func @unit_operand(%arg0: f32) {
  sair.program {
    // expected-error@below {{expceted concerete instance or copy as operand #0}}
    %0 = sair.from_scalar %arg0 {
      instances = [{operands = [unit]}]
    } : !sair.value<(), f32>
    sair.exit { instances = [{operands = []}] }
  }
  return
}

// -----

func @unit_copy(%arg0: f32) {
  sair.program {
    // expected-error@below {{expected the source of copy to be specified}}
    %0 = sair.from_scalar %arg0 {
      instances = [{operands = [#sair.instance<0>]}],
      copies = [[{copy_of = unit}]]
    } : !sair.value<(), f32>
    sair.exit { instances = [{operands = []}] }
  }
  return
}
