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

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(sair LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(SAIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SAIR_TOOLS_DIR ${SAIR_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Generate Sair Ops.
set(LLVM_TARGET_DEFINITIONS sair_ops.td)
mlir_tablegen(sair_ops.h.inc -gen-op-decls)
mlir_tablegen(sair_ops.cc.inc -gen-op-defs)
add_public_tablegen_target(sair_ops_inc_gen)

# Generate Sair interfaces and structured attributes.
set(LLVM_TARGET_DEFINITIONS sair_base.td)
mlir_tablegen(sair_op_interfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(sair_op_interfaces.cc.inc -gen-op-interface-defs)
mlir_tablegen(sair_structs.h.inc -gen-struct-attr-decls)
mlir_tablegen(sair_structs.cc.inc -gen-struct-attr-defs)
add_public_tablegen_target(sair_op_interfaces_inc_gen)

# Main dialect library.
add_mlir_library(sair_dialect
  canonicalization_patterns.cc
  sair_attributes.cc
  sair_dialect.cc
  sair_op_interfaces.cc
  sair_ops.cc
  sair_types.cc

  DEPENDS
  sair_ops_inc_gen
  sair_op_interfaces_inc_gen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRSideEffectInterfaces
  MLIRDerivedAttributeOpInterface
  )

# Utility library for registering Sair with MLIR.
add_mlir_library(sair_registration
  sair_registration.cc

  LINK_LIBS PUBLIC
  sair_default_lowering_attributes
  sair_dialect
  sair_from_linalg
  sair_lowering
  )

# sair-opt pass driver.
get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
set(OPT_LIBS
  ${mlir_libs}
  MLIROptLib
  sair_registration
  )

add_llvm_executable(sair-opt
  sair_opt.cc
  )
llvm_update_compile_flags(sair-opt)
target_link_libraries(sair-opt
  PRIVATE
  ${OPT_LIBS}
  sair_lowering
  )

enable_testing()
add_subdirectory(test)
add_subdirectory(transforms)
