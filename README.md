# Structured Additive Intermediate Representation (Sair)

An intermediate program representation, based on [MLIR](http://mlir.llvm.org)
and designed to express implementation decisions on a program without losing the
high-level structure of the code. It encodes implementation choices as
attributes added on to existing operations (hence "additive" in the name).

## Coding Guidelines

Follow [Google C++ Style
Guide](https://google.github.io/styleguide/cppguide.html) whenever possible.
Note that MLIR uses LLVM style that is often incompatible with Google style;
only use LLVM style for MLIR overrides or CRTP hooks.

Follow MLIR
[testing guide](https://mlir.llvm.org/getting_started/TestingGuide/). In
particular, IR attributes, types and operations should be tested using textual
IR format via FileCheck. Non-IR components should be tested with unit tests.

## Build Instructions

Prerequisites:
 * C++ compiler supporting C++17;
 * git, cmake, ninja (or make)

Instructions:

First, fetch LLVM repository from git.

```
cd <llvm-path>
git clone https://github.com/llvm/llvm-project
cd llvm-project
```

Checkout the latest version of LLVM compatible with Sair, as specified in
`LLVM_VERSION` in the repository.

```
VERSION=`cat <sair-path>/LLVM_VERSION`
git checkout $VERSION
```

(If MLIR, LLVM and relevant headers are already installed, skip this step).

Configure and compile LLVM with MLIR project enabled by following [MLIR
instructions](https://mlir.llvm.org/getting_started/). Make sure to set up the
installation path. The following is only given as example:

```
mkdir build
cd build
cmake ../llvm -DLLVM_ENABLE_PROJECTS='mlir' \
  -DLLVM_TARGETS_TO_BUILD='host;NVPTX;AMDGPU' \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=On \
  -DLLVM_INCLUDE_Utils=On \
  -DCMAKE_INSTALL_PREFIX=<mlir-prefix> \
  -G Ninja  # Omit for "make" build.
ninja install -j <num_procs>  # make install -j <num_procs>
```

With MLIR installed, configure Sair.

```
cd <sair-path>
mkdir build
cd build
cmake .. \
  -DCMAKE_PREFIX_PATH=<mlir-prefix> \
  -DLLVM_EXTERNAL_LIT=<path-to-lit> \
  -G Ninja  # Omit for "make" build.
```

The `LLVM_EXTERNAL_LIT` line is necessary to configure LLVM's testing
infrastructure to run Sair tests and can be omitted if one does not intend to
run Sair tests. `llvm-lit` is often provided by system packages, or is available
in the _build_ path of LLVM since it is intended for use at build time.
Therefore, `<path-to-lit>` can point to the system installation of `llvm-lit`,
if any, or to `<llvm-path>/build/bin/llvm-lit` if LLVM was built from source.

One can now build and test Sair.

```
# To compile, run:
ninja sair-opt -j <num_procs>  # make sair-opt

# To check test (if Lit was configured), run:
ninja check-sair -j <num_procs>  # make check-sair
```

The compilation produces a single standalone statically-linked binary `sair-opt`
that can be moved.
