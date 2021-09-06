// RUN: sair-opt -sair-assign-default-expansion %s -split-input-file -verify-diagnostics

func @main() {
  sair.program {
    // expected-error @+1 {{not supported}}
    sair.map_reduce reduce attributes {instances = [{}]} {
      ^bb0:
        sair.return
    } : #sair.shape<()>, () -> ()
    sair.exit
  }
  return
}
