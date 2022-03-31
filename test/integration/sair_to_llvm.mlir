// RUN: sair-opt -sair-default-lowering-attributes -convert-sair-to-llvm %s | mlir-cpu-runner -e from_scalar | FileCheck %s
// RUN: sair-opt -sair-default-lowering-attributes -convert-sair-to-llvm %s | mlir-cpu-runner -e from_to_memref | FileCheck %s

// All functions should func.return 1.0 on success.
// CHECK: 1.0

// Helper function that returns 1.0 if the two memrefs are equal and 0.0
// otherwise.
func.func @check_memrefs_equal(%lhs: memref<8xi32>, %rhs: memref<8xi32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0f = arith.constant 0.0 : f32
  %c1f = arith.constant 1.0 : f32
  cf.br ^bb0(%c0 : index)

// Loop on the memrefs.
^bb0(%0: index):
  %1 = arith.cmpi slt, %0, %c8 : index
  // Return 1.0 if we reached the end without error.
  cf.cond_br %1, ^bb1(%0 : index), ^bb2(%c1f : f32)
^bb1(%2: index):
  %4 = memref.load %lhs[%2] : memref<8xi32>
  %5 = memref.load %rhs[%2] : memref<8xi32>
  %3 = arith.addi %2, %c1 : index
  %6 = arith.cmpi eq, %4, %5 : i32
  // Returns 0.0 if we found an error.
  cf.cond_br %6, ^bb0(%3 : index), ^bb2(%c0f : f32)

^bb2(%7: f32):
  func.return %7 : f32
}

func.func @from_scalar() -> f32 {
  %0 = arith.constant 1.0 : f32
  %2 = sair.program {
    %1 = sair.from_scalar %0 : !sair.value<(), f32>
    sair.exit %1 : f32
  } : f32
  func.return %2 : f32
}

func.func @from_to_memref() -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // Create two memrefs such that %0[i] = 2*%1[i].
  %0 = memref.alloca() : memref<8xi32>
  %1 = memref.alloca() : memref<8xi32>
  scf.for %i = %c0 to %c8 step %c1 {
    %2 = arith.index_cast %i : index to i32
    %3 = arith.addi %2, %2 : i32
    memref.store %2, %0[%i] : memref<8xi32>
    memref.store %3, %1[%i] : memref<8xi32>
  }

  // Multiply the elements of %1 by two.
  sair.program {
    %2 = sair.static_range : !sair.static_range<8>
    %6 = sair.from_scalar %0 : !sair.value<(), memref<8xi32>>
    %3 = sair.from_memref %6 memref[d0:%2] {
      buffer_name = "bufferA"
    }  : #sair.shape<d0:static_range<8>>, memref<8xi32>
    %4 = sair.map[d0:%2] %3(d0) {
      ^bb0(%arg0: index, %arg1: i32):
        %5 = arith.addi %arg1, %arg1 : i32
        sair.return %5 : i32
    } : #sair.shape<d0:static_range<8>>, (i32) -> i32
    sair.to_memref %6 memref[d0:%2] %4(d0) {
      buffer_name = "bufferB"
    }  : #sair.shape<d0:static_range<8>>, memref<8xi32>
    sair.exit
  }

  // Check that %0 and %1 are equal.
  %2 = func.call @check_memrefs_equal(%0, %1) : (memref<8xi32>, memref<8xi32>) -> f32
  func.return %2 : f32
}
