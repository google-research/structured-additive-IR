// RUN: sair-opt -split-input-file -verify-diagnostics %s

// expected-error @+1 {{invalid sair type}}
func @invalid_type() -> !sair.foo

// -----

// expected-error @+1 {{unexpected nul or EOF}}
func @unfinished_range_dep() -> !sair.range<d0:range, d1:range

// -----

// expected-error @+1 {{expected 'x' or '>'}}
func @garbage_in_range_dep() -> !sair.range<d0:range? d1:range>

// -----

// expected-error @+1 {{dimension 'd1' is out of range (0 dimensions)}}
func @invalid_dep() -> !sair.range<d0:range(d1)>
// -----

// expected-error @+1 {{invalid dimension name}}
func @invalid_dep() -> !sair.range<d0:range(x)>

// -----

// expected-error @+1 {{non-transitive dependency}}
func @non_transitive_dependency() -> !sair.range<d0:range x d1:range(d0) x d2:range(d1)>

// -----

// expected-error @+1 {{dimension d0 appears twice}}
func @duplicate_dependency() -> !sair.range<d0:range x d1:range(d0, d0)>

// -----

// expected-error @+1 {{expected 'd1'}}
func @shape_dim_redefinition() -> !sair.range<d0:range x d0:range>

// -----

// expected-error @+1 {{the access pattern must map all dimensions}}
func @shape_none_dim() -> !sair.range<d0:range x d1:range(none)>

// -----

func @range_must_be_positive() {
  sair.program {
    // expected-error @+1 {{step must be strictly positive}}
    %0 = sair.static_range 18 step 0 : !sair.range
    sair.exit
  }
  return
}

// -----

func @operand_access_pattern_dim_not_mapped(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, f32>
    // expected-error @+1 {{expected access pattern to a concrete element, got 'none'}}
    %3 = sair.copy %2(none) : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @operand_access_pattern_dim_not_mapped_raw(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1] %0 : !sair.value<d0:range, f32>
    // expected-error @+1 {{all dimensions of the accessed domain must be mapped}}
    %3 = "sair.copy"(%2) {
      access_pattern_array = [#sair.pattern<0 : none>]
    } : (!sair.value<d0:range, f32>) -> !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @dimension_out_of_range() {
  // expected-error @+1 {{dimension 'd2' is out of range}}
  "foo"() {access_pattern = #sair.pattern<2 : d0, d1, d2>} : () -> ()
}

// -----

// expected-note @+1 {{prior use here}}
func @range_op_invalid_type(%arg0 : !sair.value<(), index>) {
  sair.program {
    // expected-error @+1 {{expects different type}}
    %1 = sair.dyn_range[d0:%arg0] %arg0 : !sair.range<d0:range>
    sair.exit
  }
  return
}

// -----

func @domain_unexpected_num_dims(%arg0 : !sair.value<(), index>) {
  sair.program {
    %0 = sair.dyn_range %arg0 : !sair.range
    // expected-error @+1 {{2 operands present, but expected 1}}
    %1 = sair.dyn_range[d0:%0, d1:%0] %arg0 : !sair.range<d0:range>
    sair.exit
  }
  return
}

// -----

func @domain_unexpected_dimension_type(%arg0 : !sair.value<(), index>) {
  sair.program {
    // expected-note @+1 {{prior use here}}
    %0 = sair.dyn_range %arg0 : !sair.range
    // expected-error @+1 {{expects different type}}
    %1 = sair.dyn_range[d0:%0, d1:%0] %arg0 : !sair.range<d0:range x d1:range(d0)>
    sair.exit
  }
  return
}

// -----

func @domain_dim_redefinition(%arg0 : !sair.value<(), index>) {
  sair.program {
    %0 = sair.dyn_range %arg0 : !sair.range
    // expected-error @+1 {{expected 'd1'}}
    %1 = sair.dyn_range[d0:%0, d0:%0] %arg0 : !sair.range<range x range>
    sair.exit
  }
  return
}

// -----

func @dimension_use_before_def(%arg0 : f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{dimension used before its definition}}
    %1 = sair.copy[d0:%2] %0 : !sair.value<d0:range, f32>
    // expected-note @+1 {{definition here}}
    %2 = sair.static_range 8 : !sair.range
    sair.exit
  }
  return
}

// -----

func @operand_use_before_def(%arg0 : f32) {
  sair.program {
    // expected-error @+1 {{operand used before its definition}}
    %0 = sair.copy %1 : !sair.value<(), f32>
    // expected-note @+1 {{definition here}}
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @invalid_access_pattern(%arg0 : !sair.range,
                             // expected-note @+1 {{prior use here}}
                             %arg1 : !sair.value<d0:range x d1:range(d0), index>) {
  sair.program {
    // expected-error @+1 {{expects different type}}
    %0 = sair.dyn_range[d0: %arg0, d1:%arg0] %arg1(d0, d1) : !sair.range<d0:range x d1:range>
    sair.exit
  }
  return
}

// -----

func @invalid_attr_name() {
  // expected-error @+1 {{unexpected Sair attribute}}
  "foo"() {shape=#sair.invalid_attr_name<()>} : () -> ()
}

// -----

func @copy_exected_same_element_type(%arg0 : f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{same element type}}
    %1 = "sair.copy"(%0) {access_pattern_array=[#sair.pattern<0>]}
      : (!sair.value<(), f32>) -> (!sair.value<(), i32>)
    sair.exit
  }
  return
}

// -----

func @invalid_use_domain_size(%arg0 : f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{invalid use domain size}}
    %1 = "sair.copy"(%0) {access_pattern_array=[#sair.pattern<1>]}
      : (!sair.value<(), f32>) -> (!sair.value<(), f32>)
    sair.exit
  }
  return
}

// -----

func @copy_expected_value() {
  sair.program {
    // expected-error @+1 {{expected a sair value access}}
    sair.copy : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @from_memref_exected_same_element_type(%arg0 : memref<f32>) {
  sair.program {
    // expected-error @+1 {{same element type}}
    %0 = "sair.from_memref"(%arg0) : (memref<f32>) -> (!sair.value<(), i32>)
    sair.exit
  }
  return
}

// -----

func @hyper_rectangular_domain(%arg0: index, %arg1 : memref<?x?xf32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), index>
    %2 = sair.dyn_range[d0:%0] %1 :!sair.range<d0:range>

    // expected-error @+1 {{hyper-rectangular}}
    %3 = sair.from_memref[d0:%0, d1:%2] %arg1
      : memref<?x?xf32> -> !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  return
}

// -----

func @same_rank(%arg0 : memref<?xf32>) {
  sair.program {
    // expected-error @+1 {{same rank}}
    %0 = sair.from_memref %arg0 : memref<?xf32> -> !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @map_wrong_body_argument_count(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{expects 2 body arguments}}
    sair.map[d0:%0] %1(d0) {
    ^bb0(%arg1: index):
      sair.return %arg1 : index
    } : #sair.shape<d0:range>, (i32) -> (i32)
    sair.exit
  }
  return
}

// -----

func @map_wrong_body_argument_type(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{expects first 1 body arguments to have type 'index'}}
    sair.map[d0:%0] %1(d0) {
    ^bb0(%arg1: i32, %arg2: i32):
      sair.return %arg2 : i32
    } : #sair.shape<d0:range>, (i32) -> (i32)
    sair.exit
  }
  return
}

// -----

func @map_wrong_body_argument_trailing_type(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{expects trailing body arguments to have the same element type as operands}}
    sair.map[d0:%0] %1(d0) {
    ^bb0(%arg1: index, %arg2: i64):
      sair.return %arg2 : i64
    } : #sair.shape<d0:range>, (i32) -> (i32)
    sair.exit
  }
  return
}

// -----

func @map_wrong_terminator(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{expects body to be terminated with 'sair.return'}}
    sair.map[d0:%0] %1(d0) {
    ^bb0(%arg1: index, %arg2: i32):
      "op"() : () -> ()
    } : #sair.shape<d0:range>, (i32) -> (i32)
    sair.exit
  }
  return
}

// -----

func @map_wrong_terminator_operand(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{expects element types of results to match operand types of the body terminator}}
    sair.map[d0:%0] %1(d0) {
    ^bb0(%arg1: index, %arg2: i32):
      // expected-note @+1 {{body terminator}}
      sair.return %arg2 : i32
    } : #sair.shape<d0:range>, (i32) -> (i64)
    sair.exit
  }
  return
}

// -----

func @map_reduce_wrong_trailing_arg_count(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    sair.map_reduce[d0:%0] %1(d0) reduce[d1:%0] %1(d1) {
    ^bb0(%arg1: i32):
      sair.return %arg1 : i32
    } : #sair.shape<d0:range x d1:range>, (i32, i32) -> i32
    // expected-error @-1 {{expected 1 arguments in the trailing function type}}
    sair.exit
  }
  return
}

// -----

func @map_reduce_wrong_trailing_res_count(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    sair.map_reduce[d0:%0] %1(d0) reduce[d1:%0] %1(d1) {
    ^bb0(%arg1: i32):
      sair.return %arg1 : i32
    } : #sair.shape<d0:range x d1:range>, (i32) -> (i32, i32)
    // expected-error @-1 {{expected 1 results in the trailing function type}}
    sair.exit
  }
  return
}


// -----

func @map_reduce_wrong_body_argument_count(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{expects 4 body arguments}}
    sair.map_reduce[d0:%0] %1(d0) reduce[d1:%0] %1(d1) {
    ^bb0(%arg1: i32):
      sair.return %arg1 : i32
    } : #sair.shape<d0:range x d1:range>, (i32) -> i32
    sair.exit
  }
  return
}

// -----

func @map_reduce_wrong_terminator_type(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{expects element types of results to match operand types of the body terminator}}
    sair.map_reduce[d0:%0] %1(d0) reduce[d1:%0] %1(d1) {
    ^bb0(%arg1: index, %arg2: index, %arg3: i32, %arg4: i32):
      // expected-note @+1 {{body terminator}}
      sair.return %arg1 : index
    } : #sair.shape<d0:range x d1:range>, (i32) -> i32
    sair.exit
  }
  return
}

// -----

func @map_reduce_init_accessing_reduction(%arg0 : memref<?xi32>) {
  sair.program {
    %0 = sair.static_range 8 :!sair.range
    %1 = sair.from_memref[d0:%0] %arg0 : memref<?xi32> -> !sair.value<d0:range, i32>
    // expected-error @+1 {{an operand access pattern references a dimension that depends on the operand}}
    %3 = "sair.map_reduce"(%0, %0, %1, %1) ( {
      ^bb0(%arg1: index, %arg2: index, %arg3: i32, %arg4: i32):
        "sair.return"(%arg3) : (i32) -> ()
      }) {access_pattern_array = [#sair.pattern<2:d1>, #sair.pattern<2:d1>],
          operand_segment_sizes = dense<1> : vector<4xi64>,
          shape = #sair.shape<d0:range x d1:range>}
       : (!sair.range, !sair.range, !sair.value<d0:range, i32>, !sair.value<d0:range, i32>)
       -> !sair.value<d0:range, i32>
    sair.exit
  }
  return
}

// -----

func @map_reduce_unexpected_shape() {
  sair.program {
    // expected-error @+1 {{unexpected shape}}
    "sair.map_reduce"() ({
      ^bb0:
        %0 = constant 1.0 : f32
        sair.return %0 : f32
    }) {
      access_pattern_array = [],
      operand_segment_sizes = dense<0> : vector<4xi64>,
      shape = #sair.shape<()>
    } : () -> !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @from_scalar_element_type() {
// expected-note @+1 {{prior use here}}
  %0 = constant 0 : index
  sair.program {
    // expected-error @+1 {{expects different type}}
    sair.from_scalar %0 : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @from_scalar_element_type_generic_form() {
  %0 = constant 0 : index
  sair.program {
    // expected-error @+1 {{expects different type}}
    "sair.from_scalar" (%0) : (index) -> !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @sair_program_non_sair_op() {
  // expected-error @+1 {{expected only Sair operations in the body}}
  sair.program {
    // expected-note @+1 {{found}}
    %0 = constant 0 : index
    sair.exit
  }
  return
}

// -----

func @sair_op_outside_sair_program() {
  // expected-error @+1 {{expected to be immediately contained in a 'sair.program'}}
  %0 = sair.static_range 42 : !sair.range
  return
}

// -----

func @sair_value_defined_outside_sair_program(%arg0: !sair.value<(), f32>) {
  sair.program {
    // expected-error @+1 {{sair values must be defined in the region they are used}}
    %0 = sair.copy %arg0 : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @sair_dimension_defined_outside_sair_program(%arg0: !sair.range) {
  %0 = constant 1.0 : f32
  sair.program {
    %1 = sair.from_scalar %0 : !sair.value<(), f32>
    // expected-error @+1 {{sair dimensions must be defined in the region they are used}}
    %2 = sair.copy[d0:%arg0] %1 : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @sair_program_wrong_terminator() {
  // expected-error @+1 {{expected a sair.exit terminator}}
  sair.program {
    sair.static_range 8 : !sair.range
  }
  return
}

// -----

func @sair_exit_wrong_num_operands() {
  %0 = sair.program {
    // expected-error @+1 {{expected 1 operands, found 0}}
    sair.exit
  } : f32
  return
}

// -----

func @sair_exit_wrong_type() {
  %c0 = constant 1 : i32
  %0 = sair.program {
    %1 = sair.from_scalar %c0 : !sair.value<(), i32>
    // expected-error @+1 {{sair.exit operands must match the return type of the sair.program: expected 'f32', found 'i32'}}
    sair.exit %1 : i32
  } : f32
  return
}

// -----

func @sair_exit_type_operands_mismatch() {
  sair.program {
    // expected-error @+1 {{expected 0 types}}
    sair.exit : f32
  }
}

// -----

func @sair_iterator_step_non_positive() {
  // expected-error @+1 {{step must be positive}}
  "foo"() { attr = #sair.iter<d0 step 0> } : () -> ()
}

// -----

func @expected_loop_attr() {
  sair.program {
    // expected-error @+1 {{expected a `Loop` attribute}}
    sair.map attributes { loop_nest = [0] } {
      ^bb0:
        sair.return
    } : #sair.shape<()>, () -> ()
    sair.exit
  }
}

// -----

func @loop_name_used_twice(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{name "A" used twice in the same loop nest}}
    sair.copy[d0: %0, d1:%0] %1 {
      loop_nest = [
        {name = "A", iter = #sair.iter<d0>},
        {name = "A", iter = #sair.iter<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  sair.return
}

// -----

func @dim_not_covered_by_loop_nest(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{not all dimensions are covered by the loop nest}}
    sair.copy[d0: %0] %1 { loop_nest = [] } : !sair.value<d0:range, f32>
    sair.exit
  }
  sair.return
}

// -----

func @loop_dependencies_not_covered(%arg0: index, %arg1: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), index>
    %2 = sair.dyn_range[d0:%0] %1 : !sair.range<d0:range>
    %3 = sair.from_scalar %arg1 : !sair.value<(), f32>

    // expected-error @+1 {{dimension 'd1' must be nested in dimension 'd0'}}
    sair.copy[d0: %0, d1: %2] %3 {
      loop_nest = [
        {name = "A", iter = #sair.iter<d1>},
        {name = "B", iter = #sair.iter<d0>}
      ]
    } : !sair.value<d0:range x d1:range(d0), f32>
    sair.exit
  }
  sair.return
}

// -----

func @unknown_dim_in_loop_nest(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{dimension 'd0' is out of range of the domain}}
    sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<(), f32>
    sair.exit
  }
  sair.return
}

// -----

func @loop_step_increasing(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{loop "B" step must be less than in outer loops}}
    sair.copy[d0: %0] %1 {
      loop_nest = [
        {name = "A", iter = #sair.iter<d0 step 2>},
        {name = "B", iter = #sair.iter<d0 step 4>}
      ]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @loop_fusion_different_prefix(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    sair.copy[d0: %0, d1: %0] %1 {
      loop_nest = [
        {name = "A", iter=#sair.iter<d0>},
        {name = "B", iter=#sair.iter<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    // expected-error @+1 {{occurrences of loop "B" must be contiguous and nested in the same loops}}
    sair.copy[d0: %0, d1: %0] %1 {
      loop_nest = [
        {name = "C", iter=#sair.iter<d0>},
        {name = "B", iter=#sair.iter<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}

// -----

func @loop_fusion_not_contiguous(%arg0: f32) {
  sair.program {
    %0 = sair.static_range 8 : !sair.range
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    sair.copy[d0: %0] %1 { loop_nest = [{name = "A", iter=#sair.iter<d0>}] }
      : !sair.value<d0:range, f32>
    sair.copy %1 : !sair.value<(), f32>
    // expected-error @+1 {{occurrences of loop "A" must be contiguous and nested in the same loops}}
    sair.copy[d0: %0] %1 { loop_nest = [{name = "A", iter=#sair.iter<d0>}] }
      : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @iter_field_missing(%arg0: f32) {
  sair.program {
    %1 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{loop "A" must have the 'iter' field set in at least one operation}}
    sair.copy %1 { loop_nest = [{name = "A", iter=#sair.iter<remat>}] }
      : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @loop_definition_mismatch(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>

    %1 = sair.static_range 8 : !sair.range
    // expected-note @+1 {{previous occurrence here}}
    sair.copy[d0: %1] %0 {loop_nest = [{name = "A", iter=#sair.iter<d0>}]}
      : !sair.value<d0:range, f32>

    %2 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{loop "A" dimension does not match previous occurrences}}
    sair.copy[d0: %2] %0 {loop_nest = [{name = "A", iter=#sair.iter<d0>}]}
      : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @loop_name_not_registered(%arg0: f32) {
  "sair.program" () ({
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>

    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{loop "A" is not declared in the parent operation}}
    sair.copy[d0: %1] %0 {loop_nest = [{name = "A", iter=#sair.iter<d0>}]}
      : !sair.value<d0:range, f32>
    sair.exit
  }) {
    loop_name_table = []
  }: () -> ()
  return
}

// -----

func @init_nested_in_reduction_loop(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{operation cannot be nested in loop "A"}}
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>
    // expected-note @+1 {{because of this operation}}
    sair.map_reduce %2 reduce[d0:%1] attributes {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } {
      ^bb0(%arg1: index, %arg2: f32):
         sair.return %arg2 : f32
    } : #sair.shape<d0:range>, () -> (f32)
    sair.exit
  }
  return
}

// -----

func @dimension_defined_in_loop_nest(%arg0: index, %arg1: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    // expected-error @+1 {{rematerialized loop "A" indirectly uses the range before it is defined}}
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), index>
    // expected-note @+1 {{range defined here}}
    %3 = sair.dyn_range %2 : !sair.range
    %4 = sair.copy[d0:%3] %1 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}],
      memory_space = [1]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @proj_last_dependency(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.copy[d0:%1] %0 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    %3 = sair.proj_last of[d0:%1] %2(d0) : #sair.shape<d0:range>, f32
    // expected-error @+1 {{loop "A" must be closed before this operation}}
    %4 = sair.copy %3 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>
    sair.exit
  }
  return
}

// -----

func @mapped_dimensions(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // expected-note @+1 {{dependency from this operation}}
    %2 = sair.copy[d0:%1, d1:%1] %0 {
      loop_nest = [
        {name = "A", iter = #sair.iter<d0>},
        {name = "B", iter = #sair.iter<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    // expected-error @+1 {{loop "A" violates a data dependency}}
    %3 = sair.copy[d0:%1, d1:%1] %2(d1, d0) {
      loop_nest = [
        {name = "A", iter = #sair.iter<d0>},
        {name = "B", iter = #sair.iter<d1>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}

// -----

func @dimension_size_loop_nest(%arg0: index, %arg1: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), index>
    %1 = sair.from_scalar %arg1 : !sair.value<(), f32>
    %2 = sair.static_range 8 : !sair.range

    %3 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), index>
    %4 = sair.dyn_range %3 : !sair.range

    // expected-error @+1 {{operation cannot be nested in loop "A"}}
    %5 = sair.copy[d0:%2, d1:%4] %1 {
      loop_nest = [
        {name = "A", iter = #sair.iter<d0>},
        {name = "B", iter = #sair.iter<d1>}
      ],
      memory_space = [1]
    } : !sair.value<d0:range x d1:range, f32>
    sair.exit
  }
  return
}

// -----

func @fby_must_fuse(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    %2 = sair.fby %0 then[d0:%1] %4(d0) : !sair.value<d0:range, f32>
    // expected-error @+1 {{loop "B" must be open at or before this operation}}
    %3 = sair.copy[d0:%1] %2(d0) {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    %4 = sair.copy[d0:%1] %3(d0) {
      loop_nest = [{name = "B", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}

// -----

func @fby_of_proj_dependency(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{cannot take the previous value of the operand along 'd0'}}
    %2 = sair.fby %0 then[d0:%1] %4(d0) : !sair.value<d0:range, f32>
    %3 = sair.map[d0:%1, d1:%1] %2(d0) attributes {
      loop_nest = [
        {name = "A", iter = #sair.iter<d1>},
        {name = "B", iter = #sair.iter<d0>}
      ]
    } {
      ^bb0(%arg1: index, %arg2: index, %arg3: f32):
        sair.return %arg3 : f32
    } : #sair.shape<d0:range x d1:range>, (f32) -> f32
    %4 = sair.proj_last[d0:%1] of[d1:%1] %3(d0, d1)
      : #sair.shape<d0:range x d1:range>, f32
    sair.exit
  }
  return
}

// -----

func @fby_of_fby_dependency(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    %1 = sair.static_range 8 : !sair.range
    // expected-error @+1 {{cannot take the previous value of the operand along 'd0'}}
    %2 = sair.fby %0 then[d0:%1] %5(d0) : !sair.value<d0:range, f32>
    %3 = sair.fby[d0:%1] %2(d0) then[d1:%1] %4(d0, d1)
      : !sair.value<d0:range x d1:range, f32>
    %4 = sair.copy[d0:%1, d1:%1] %3(d0, d1) {
      loop_nest = [
        {name = "A", iter = #sair.iter<d1>},
        {name = "B", iter = #sair.iter<d0>}
      ]
    } : !sair.value<d0:range x d1:range, f32>
    %5 = sair.proj_last[d0:%1] of[d1:%1] %4(d0, d1)
      : #sair.shape<d0:range x d1:range>, f32
    sair.exit
  }
  return
}

// -----

func @wrong_order_for_remat(%arg0: f32) {
  sair.program {
    %0 = sair.from_scalar %arg0 : !sair.value<(), f32>
    // expected-error @+1 {{rematerialized loop "A" indirectly uses the range before it is defined}}
    %2 = sair.copy %0 {
      loop_nest = [{name = "A", iter = #sair.iter<remat>}]
    } : !sair.value<(), f32>
    // expected-note @+1 {{range defined here}}
    %1 = sair.static_range 8 : !sair.range
    %3 = sair.copy[d0:%1] %2 {
      loop_nest = [{name = "A", iter = #sair.iter<d0>}]
    } : !sair.value<d0:range, f32>
    sair.exit
  }
  return
}
