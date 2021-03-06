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

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(SAIR_TEST_DEPS
  sair-opt
  )

add_lit_testsuite(check-sair
  "Running SAIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}

  DEPENDS
  ${SAIR_TEST_DEPS}
  )

add_lit_testsuites(SAIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${SAIR_TEST_DEPS})

# Generate test passes declaration
set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name Test)
add_public_tablegen_target(sair_test_passes_inc_gen)

# Test passes library
add_mlir_library(sair_test_passes
  passes.cc

  DEPENDS
  sair_test_passes_inc_gen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRLinalg
  MLIRStandard
  MLIRSupport
  sair_dialect
  )
