# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generate pass declarations for Linalg->Sair conversion.
set(LLVM_TARGET_DEFINITIONS sair_from_linalg.td)
mlir_tablegen(sair_from_linalg.h.inc -gen-pass-decls -name SAIRFromLinalg)
add_public_tablegen_target(sair_from_linalg_inc_gen)

# Linalg->Sair conversion library.
add_mlir_library(sair_from_linalg
  sair_from_linalg.cc

  DEPENDS
  sair_from_linalg_inc_gen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLinalg
  MLIRMemRef
  MLIRStandard
  MLIRSupport
  sair_dialect
  )

# Generate pass declarations for Sair lowering.
set(LLVM_TARGET_DEFINITIONS default_lowering_attributes.td)
mlir_tablegen(default_lowering_attributes.h.inc -gen-pass-decls -name DefaultLoweringAttributes)
add_public_tablegen_target(sair_default_lowering_attributes_inc_gen)

# Sair lowering library.
add_mlir_library(sair_default_lowering_attributes
  default_lowering_attributes.cc

  DEPENDS
  sair_default_lowering_attributes_inc_gen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  sair_dialect
  )

# Generate pass declarations for Sair transformations.
set(LLVM_TARGET_DEFINITIONS lowering.td)
mlir_tablegen(lowering.h.inc -gen-pass-decls -name Lowering)
add_public_tablegen_target(sair_lowering_inc_gen)

# Sair transformation library.
add_mlir_library(sair_lowering
  lowering.cc
  inline_trivial_ops.cc
  introduce_loops.cc
  lower_to_map.cc
  materialize_buffers.cc
  normalize_loops.cc

  DEPENDS
  sair_lowering_inc_gen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineToStandard
  MLIRIR
  MLIRLLVMIR
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRSCFToStandard
  MLIRStandard
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTransforms
  MLIRSideEffectInterfaces
  sair_default_lowering_attributes
  sair_dialect
  )
