#!/usr/bin/env bash
# Builds the OpenCL backend, its tests and benchmark without LibTorch or the
# GUI, so it also works on Windows (MSYS2 MinGW g++).
#
#   OPENCL_INCLUDE  directory that contains CL/cl.h
#   OPENCL_LIB      OpenCL import library or linker flag
#
# Defaults: the CUDA 12.9 toolkit on Windows, the system library elsewhere.
# Run from the repository root:  bash scripts/build_opencl.sh
set -euo pipefail

CXX=${CXX:-g++}
OUT=${OUT:-build-opencl}

if [[ "${OS:-}" == "Windows_NT" ]]; then
  CUDA_DIR=${CUDA_DIR:-"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"}
  OPENCL_INCLUDE=${OPENCL_INCLUDE:-"$CUDA_DIR/include"}
  # Linking System32/OpenCL.dll through -L picks up the wrong C runtime, so use
  # the toolkit's import library.
  OPENCL_LIB=${OPENCL_LIB:-"$CUDA_DIR/lib/x64/OpenCL.lib"}
  EXE=.exe
else
  OPENCL_INCLUDE=${OPENCL_INCLUDE:-/usr/include}
  OPENCL_LIB=${OPENCL_LIB:--lOpenCL}
  EXE=
fi

FLAGS=(-std=c++17 -O2 -Wall -Wextra -Iinclude -I"$OPENCL_INCLUDE" -pthread)
SOURCES=(src/board.cpp src/game_position.cpp src/inference_queue.cpp
         src/mcts.cpp src/opencl_network.cpp)

mkdir -p "$OUT"
OBJECTS=()
for source in "${SOURCES[@]}"; do
  object="$OUT/$(basename "${source%.cpp}").o"
  "$CXX" "${FLAGS[@]}" -c "$source" -o "$object"
  OBJECTS+=("$object")
done

"$CXX" "${FLAGS[@]}" -Itests tests/opencl_tests.cpp "${OBJECTS[@]}" \
  "$OPENCL_LIB" -o "$OUT/opencl_tests$EXE"
"$CXX" "${FLAGS[@]}" tools/opencl_bench.cpp "${OBJECTS[@]}" \
  "$OPENCL_LIB" -o "$OUT/opencl_bench$EXE"

echo "built $OUT/opencl_tests$EXE and $OUT/opencl_bench$EXE"
