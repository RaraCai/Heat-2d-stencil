#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
./build/release/stencil_serial
