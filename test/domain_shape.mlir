// RUN: sair-opt -allow-unregistered-dialect %s -test-domain-shape | FileCheck %s

module {

// CHECK: "test.accessed_shape"() {label = @dimensions, result = #sair.shape<d0:dyn_range x d1:dyn_range(d0)>}
"test.accessed_shape"() {
  label = @dimensions,
  shape = #sair.shape<d0:dyn_range x d1:dyn_range x d2:dyn_range(d0)>,
  mapping = #sair.mapping<3: d0, d2>
} : () -> ()

// CHECK: "test.accessed_shape"() {label = @stripe,
// CHECK-SAME: result = #sair.shape<d0:dyn_range x d1:dyn_range(d0) x d2:dyn_range(d0, d1)>}
"test.accessed_shape"() {
  label = @stripe,
  shape = #sair.shape<d0:dyn_range x d1:dyn_range(d0)>,
  mapping = #sair.mapping<2: d0, stripe(d1, [4]), stripe(d1, [4, 1])>
} : () -> ()

// CHECK: "test.accessed_shape"() {label = @unstripe,
// CHECK-SAME: result = #sair.shape<d0:dyn_range x d1:dyn_range(d0)>}
"test.accessed_shape"() {
  label = @unstripe,
  shape = #sair.shape<d0:dyn_range x d1:dyn_range(d0) x d2:dyn_range(d0, d1)>,
  mapping = #sair.mapping<3: d0, unstripe(d1, d2, [4, 1])>
} : () -> ()

}
