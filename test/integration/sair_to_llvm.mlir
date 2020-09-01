// RUN: sair-opt -sair-assign-default-memory-space -convert-sair-to-llvm %s | mlir-cpu-runner -e from_scalar | FileCheck %s
// RUN: sair-opt -sair-assign-default-memory-space -convert-sair-to-llvm %s | mlir-cpu-runner -e from_to_memref | FileCheck %s

// All functions should return 1.0 on success.
// CHECK: 1.0

// Helper function that returns 1.0 if the two memrefs are equal and 0.0
// otherwise.
func @check_memrefs_equal(%lhs: memref<8xi32>, %rhs: memref<8xi32>) -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index
  %c0f = constant 0.0 : f32
  %c1f = constant 1.0 : f32
  br ^bb0(%c0 : index)

// Loop on the memrefs.
^bb0(%0: index):
  %1 = cmpi "slt", %0, %c8 : index
  // Return 1.0 if we reached the end without error.
  cond_br %1, ^bb1(%0 : index), ^bb2(%c1f : f32)
^bb1(%2: index):
  %4 = load %lhs[%2] : memref<8xi32>
  %5 = load %rhs[%2] : memref<8xi32>
  %3 = addi %2, %c1 : index
  %6 = cmpi "eq", %4, %5 : i32
  // Returns 0.0 if we found an error.
  cond_br %6, ^bb0(%3 : index), ^bb2(%c0f : f32)

^bb2(%7: f32):
  return %7 : f32
}

func @from_scalar() -> f32 {
  %0 = constant 1.0 : f32
  %2 = sair.program {
    %1 = sair.from_scalar %0 : !sair.value<(), f32>
    sair.exit %1 : f32
  } : f32
  return %2 : f32
}

func @from_to_memref() -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c8 = constant 8 : index

  // Create two memrefs such that %0[i] = 2*%1[i].
  %0 = alloca() : memref<8xi32>
  %1 = alloca() : memref<8xi32>
  scf.for %i = %c0 to %c8 step %c1 {
    %2 = index_cast %i : index to i32
    %3 = addi %2, %2 : i32
    store %2, %0[%i] : memref<8xi32>
    store %3, %1[%i] : memref<8xi32>
  }

  // Multiply the elements of %1 by two.
  sair.program {
    %2 = sair.static_range 8 : !sair.range
    %3 = sair.from_memref[d0:%2] %0
      : memref<8xi32> -> !sair.value<d0:range, i32>
    %4 = sair.map[d0:%2] %3(d0) {
      ^bb0(%arg0: index, %arg1: i32):
        %5 = addi %arg1, %arg1 : i32
        sair.return %5 : i32
    } : #sair.shape<d0:range>, (i32) -> i32
    sair.to_memref[d0:%2] %4(d0), %0 : memref<8xi32>
    sair.exit
  }

  // Check that %0 and %1 are equal.
  %2 = call @check_memrefs_equal(%0, %1) : (memref<8xi32>, memref<8xi32>) -> f32
  return %2 : f32
}
