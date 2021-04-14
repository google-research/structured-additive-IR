// RUN: sair-opt -allow-unregistered-dialect %s -test-mapping-exprs \
// RUN:   -mlir-print-local-scope | FileCheck %s

module {

// CHECK: "test.has_none_exprs"() {label = @none, result = true}
"test.has_none_exprs"() {label = @none, expr = #sair.mapping_expr<none>} : () -> ()

// CHECK: "test.has_none_exprs"() {label = @dim, result = false}
"test.has_none_exprs"() {label = @dim, expr = #sair.mapping_expr<d0>} : () -> ()

// CHECK: "test.has_none_exprs"() {label = @stripe_false, result = true}
"test.has_none_exprs"() {
  label = @stripe_false, expr = #sair.mapping_expr<stripe(none, [2])>
} : () -> ()

// CHECK: "test.has_none_exprs"() {label = @stripe_true, result = false}
"test.has_none_exprs"() {
  label = @stripe_true, expr = #sair.mapping_expr<stripe(d0, [2])>
} : () -> ()

// CHECK: "test.has_none_exprs"() {label = @unstripe_false, result = true}
"test.has_none_exprs"() {
  label = @unstripe_false, expr = #sair.mapping_expr<unstripe(none, [1])>
} : () -> ()

// CHECK: "test.has_none_exprs"() {label = @unstripe_true, result = false}
"test.has_none_exprs"() {
  label = @unstripe_true, expr = #sair.mapping_expr<unstripe(d0, [1])>
} : () -> ()

// CHECK: "test.min_domain_size"() {result = 3 : index}
"test.min_domain_size"() {
  expr = #sair.mapping_expr<unstripe(stripe(d2, [4]), none, [2, 1])>
} : () -> ()

// CHECK: "test.substitute_dims"() {result = #sair.mapping_expr<unstripe(stripe(d1, [4]), none, [4, 1])>}
"test.substitute_dims"() {
  expr = #sair.mapping_expr<unstripe(stripe(d0, [4]), none, [4, 1])>,
  substitutions = [#sair.mapping_expr<d1>]
} : () -> ()

// CHECK: "test.unify"() {label = @same_dim, result = #sair.mapping_expr<d0>}
"test.unify"() {
  label = @same_dim,
  expr = #sair.mapping_expr<d0>,
  other = #sair.mapping_expr<d0>
} : () -> ()

// CHECK: "test.unify"() {label = @none, result = #sair.mapping_expr<d0>}
"test.unify"() {
  label = @none,
  expr = #sair.mapping_expr<d0>,
  other = #sair.mapping_expr<none>
} : () -> ()

// CHECK: "test.unify"() {label = @different_dims, result}
"test.unify"() {
  label = @different_dims,
  expr = #sair.mapping_expr<d0>,
  other = #sair.mapping_expr<d1>
} : () -> ()

// CHECK: "test.unify"() {label = @stripe, result = #sair.mapping_expr<stripe(d0, [4])>}
"test.unify"() {
  label = @stripe,
  expr = #sair.mapping_expr<stripe(d0, [4])>,
  other = #sair.mapping_expr<stripe(none, [4])>
} : () -> ()

// CHECK: "test.unify"() {label = @stripe_different_factors, result}
"test.unify"() {
  label = @stripe_different_factors,
  expr = #sair.mapping_expr<stripe(d0, [4])>,
  other = #sair.mapping_expr<stripe(none, [8])>
} : () -> ()

// CHECK: "test.unify"() {label = @stripe_different_size, result}
"test.unify"() {
  label = @stripe_different_size,
  expr = #sair.mapping_expr<stripe(d0, [16, 4])>,
  other = #sair.mapping_expr<stripe(none, [8, 4])>
} : () -> ()

// CHECK: "test.unify"() {label = @unstripe_same_factors,
// CHECK-SAME: result = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>}
"test.unify"() {
  label = @unstripe_same_factors,
  expr = #sair.mapping_expr<unstripe(d0, none, [4, 1])>,
  other = #sair.mapping_expr<unstripe(none, d1, [4, 1])>
} : () -> ()


// CHECK: "test.unify"() {label = @unstripe_compatible_factors,
// CHECK-SAME: result = #sair.mapping_expr<unstripe(d0, none, d1, [4, 2, 1])>}
"test.unify"() {
  label = @unstripe_compatible_factors,
  expr = #sair.mapping_expr<unstripe(d0, none, [4, 1])>,
  other = #sair.mapping_expr<unstripe(none, none, d1, [4, 2, 1])>
} : () -> ()


// CHECK: "test.unify"() {label = @unstripe_incompatible_factors, result}
"test.unify"() {
  label = @unstripe_incompatible_factors,
  expr = #sair.mapping_expr<unstripe(d0, none, [4, 1])>,
  other = #sair.mapping_expr<unstripe(none, d1, [8, 1])>
} : () -> ()

// CHECK: "test.find_in_inverse"() {label = @stripe, result = #sair.mapping_expr<d2>}
"test.find_in_inverse"() {
  label = @stripe,
  expr = #sair.mapping_expr<stripe(d1, [4, 1])>,
  inverse = #sair.mapping<3 : d0, unstripe(d1, d2, [4, 1])>
} : () -> ()

// CHECK: "test.find_in_inverse"() {label = @unstripe, result = #sair.mapping_expr<d0>}
"test.find_in_inverse"() {
  label = @unstripe,
  expr = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>,
  inverse = #sair.mapping<1: stripe(d0, [4]), stripe(d0, [4, 1])>
} : () -> ()

// CHECK: "test.set_inverse"() {label = @dimension,
// CHECK-SAME: result = [#sair.mapping_expr<none>, #sair.mapping_expr<d0>]}
"test.set_inverse"() {
  label = @dimension,
  expr = #sair.mapping_expr<d1>,
  context = #sair.mapping_expr<d0>,
  inverses = [#sair.mapping_expr<none>, #sair.mapping_expr<none>]
} : () -> ()

// CHECK: "test.set_inverse"() {label = @none, result = []}
"test.set_inverse"() {
  label = @none,
  expr = #sair.mapping_expr<none>,
  context = #sair.mapping_expr<d0>,
  inverses = []
} : () -> ()

// CHECK: "test.set_inverse"() {label = @stripe,
// CHECK-SAME: result = [#sair.mapping_expr<unstripe(d0, none, [4, 1])>]}
"test.set_inverse"() {
  label = @stripe,
  expr = #sair.mapping_expr<stripe(d0, [4])>,
  context = #sair.mapping_expr<d0>,
  inverses = [#sair.mapping_expr<none>]
} : () -> ()

// CHECK: "test.set_inverse"() {label = @unstripe,
// CHECK-SAME: result = [#sair.mapping_expr<stripe(d0, [4])>, #sair.mapping_expr<stripe(d0, [4, 1])>]}
"test.set_inverse"() {
  label = @unstripe,
  expr = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>,
  context = #sair.mapping_expr<d0>,
  inverses = [#sair.mapping_expr<none>, #sair.mapping_expr<none>]
} : () -> ()

// CHECK: "test.unification_constraints"() {label = @valid
// CHECK-SAME: result = [#sair.mapping_expr<d2>, #sair.mapping_expr<none>]
"test.unification_constraints"() {
  label = @valid,
  expr = #sair.mapping_expr<unstripe(stripe(d0, [4]), d1, none, [4, 2, 1])>,
  other = #sair.mapping_expr<unstripe(stripe(d2, [4]), none, d3, [4, 2, 1])>,
  domain_size = 2
} : () -> ()

// CHECK: "test.unification_constraints"() {label = @invalid_factor_0, result}
"test.unification_constraints"() {
  label = @invalid_factor_0,
  expr = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>,
  other = #sair.mapping_expr<unstripe(d0, d1, [8, 1])>,
  domain_size = 2
} : () -> ()

// CHECK: "test.unification_constraints"() {label = @invalid_factor_1, result}
"test.unification_constraints"() {
  label = @invalid_factor_1,
  expr = #sair.mapping_expr<unstripe(d0, d1, [8, 1])>,
  other = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>,
  domain_size = 2
} : () -> ()

// CHECK: "test.unification_constraints"() {label = @invalid_inner, result}
"test.unification_constraints"() {
  label = @invalid_inner,
  expr = #sair.mapping_expr<unstripe(stripe(d0, [4]), [1])>,
  other = #sair.mapping_expr<unstripe(stripe(d0, [8]), [1])>,
  domain_size = 1
} : () -> ()

// CHECK: "test.as_affine_expr"() {label = @dim, result = affine_map<(d0) -> (d0)>}
"test.as_affine_expr"() {
  label = @dim,
  expr = #sair.mapping_expr<d0>
} : () -> ()

// CHECK: "test.as_affine_expr"() {label = @stripe,
// CHECK-SAME: result = affine_map<(d0) -> ((d0 floordiv 4) * 4)>}
"test.as_affine_expr"() {
  label = @stripe,
  expr = #sair.mapping_expr<stripe(d0, [4])>
} : () -> ()

// CHECK: "test.as_affine_expr"() {label = @unstripe,
// CHECK-SAME: result = affine_map<(d0, d1) -> (d1)>}
"test.as_affine_expr"() {
  label = @unstripe,
  expr = #sair.mapping_expr<unstripe(d0, d1, [4, 1])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @stripe_unstripe_ok,
// CHECK-SAME: result = #sair.mapping_expr<d1>
"test.canonicalize"() {
  label = @stripe_unstripe_ok,
  expr = #sair.mapping_expr<stripe(unstripe(d0, d1, d2, [8, 4, 1]), [8, 4])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @stripe_unstripe_ok_no_size,
// CHECK-SAME: result = #sair.mapping_expr<d0>
"test.canonicalize"() {
  label = @stripe_unstripe_ok_no_size,
  expr = #sair.mapping_expr<stripe(unstripe(d0, d1, [4, 1]), [4])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @stripe_unstripe_partial,
// CHECK-SAME: result = #sair.mapping_expr<stripe(unstripe(d1, d2, [4, 1]), [3])>
"test.canonicalize"() {
  label = @stripe_unstripe_partial,
  expr = #sair.mapping_expr<stripe(unstripe(d0, d1, d2, [8, 4, 1]), [8, 3])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @stripe_unstripe_fail_size,
// CHECK-SAME: result = #sair.mapping_expr<stripe(unstripe(d0, d1, d2, [8, 4, 1]), [4])>
"test.canonicalize"() {
  label = @stripe_unstripe_fail_size,
  expr = #sair.mapping_expr<stripe(unstripe(d0, d1, d2, [8, 4, 1]), [4])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @unstripe_stripe_ok,
// CHECK-SAME: result = #sair.mapping_expr<d0>
"test.canonicalize"() {
  label = @unstripe_stripe_ok,
  expr = #sair.mapping_expr<unstripe(stripe(d0, [8]), stripe(d0, [8, 4]), stripe(d0, [8, 4, 1]), [8, 4, 1])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @unstripe_stripe_fail_step,
// CHECK-SAME: result = #sair.mapping_expr<unstripe(stripe(d0, [8]), stripe(d0, [8, 4]), stripe(d0, [8, 4, 1]), [8, 3, 1])>
"test.canonicalize"() {
  label = @unstripe_stripe_fail_step,
  expr = #sair.mapping_expr<unstripe(stripe(d0, [8]), stripe(d0, [8, 4]), stripe(d0, [8, 4, 1]), [8, 3, 1])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @trivial_stripe,
// CHECK-SAME: result = #sair.mapping_expr<d0>
"test.canonicalize"() {
  label = @trivial_stripe,
  expr = #sair.mapping_expr<stripe(d0, [1])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @trivial_unstripe,
// CHECK-SAME: result = #sair.mapping_expr<d0>
"test.canonicalize"() {
  label = @trivial_unstripe,
  expr = #sair.mapping_expr<unstripe(d0, [1])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @unstripe_stripe_partial,
// CHECK-SAME: result = #sair.mapping_expr<unstripe(d0, d1, d2, [11, 7, 1])>
"test.canonicalize"() {
  label = @unstripe_stripe_partial,
  expr = #sair.mapping_expr<unstripe(d0, d1, stripe(d2, [4]), stripe(d2, [4, 1]), [11, 7, 4, 1])>
} : () -> ()

// CHECK: "test.canonicalize"() {label = @unstripe_collapse,
// CHECK-SAME: result = #sair.mapping_expr<unstripe(d0, d1, d2, [11, 7, 1])>
"test.canonicalize"() {
  label = @unstripe_collapse,
  expr = #sair.mapping_expr<unstripe(d0, unstripe(d1, d2, [7, 1]), [11, 1])>
} : () -> ()

}
