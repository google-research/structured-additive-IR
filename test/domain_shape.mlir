// RUN: sair-opt -allow-unregistered-dialect %s -test-domain-shape | FileCheck %s

module {

// CHECK: "test.accessed_shape"() {label = @dimensions, result = #sair.shape<d0:range x d1:range(d0)>}
"test.accessed_shape"() {
  label = @dimensions,
  shape = #sair.shape<d0:range x d1:range x d2:range(d0)>,
  pattern = #sair.pattern<3: d0, d2>
} : () -> ()

// CHECK: "test.accessed_shape"() {label = @stripe,
// CHECK-SAME: result = #sair.shape<d0:range x d1:range(d0) x d2:range(d0, d1)>}
"test.accessed_shape"() {
  label = @stripe,
  shape = #sair.shape<d0:range x d1:range(d0)>,
  pattern = #sair.pattern<2: d0, stripe(d1, 4), stripe(d1, 1 size 4)>
} : () -> ()

// CHECK: "test.accessed_shape"() {label = @unstripe,
// CHECK-SAME: result = #sair.shape<d0:range x d1:range(d0)>}
"test.accessed_shape"() {
  label = @unstripe,
  shape = #sair.shape<d0:range x d1:range(d0) x d2:range(d0, d1)>,
  pattern = #sair.pattern<3: d0, unstripe(d1, d2, [4])>
} : () -> ()

}
