---
status: implemented-with-open-gaps
goal: Add a Dockerfile for building and running the Nemotron cutile workspace with Rust 1.94, CUDA 13.2, and the requested Python LLM toolchain
prompt: "Implement the plan."
created: 2026-03-19T12:14:19Z
finished: 2026-03-19T12:28:02Z
---

# Executed Plan — Add CUDA Dev Dockerfile

## Summary

Added a CUDA development Dockerfile for this repository plus a `.dockerignore` that keeps build contexts small enough to be practical. The image installs Rust `1.94.0`, a Python virtualenv, PyTorch, Triton, vLLM, and a CUDA-enabled `llama.cpp`, then builds the workspace inside `/workspace`.

## What Was Done

- Added `Dockerfile` based on `nvidia/cuda:13.2.0-devel-ubuntu24.04`.
- Installed the native packages needed for Rust, Python, CUDA-side builds, tokenizers/onig, and C/C++/CMake workflows.
- Installed Rust `1.94.0` through `rustup`.
- Installed a Python virtualenv at `/opt/venv` and configured:
  - PyTorch from the official CUDA `13.0` wheel index
  - Triton from PyPI
  - vLLM from the official CUDA `13.0` release wheel, resolved with `uv --index-strategy unsafe-best-match` so PyTorch's wheel index and PyPI can be combined safely enough for this container build
- Built `llama.cpp` from source with `GGML_CUDA=ON` and installed `llama-cli` plus `llama-server`.
- Added `.dockerignore` entries for `.git`, `target`, `data`, `.vscode`, and `ai-docs`.
- Added build arguments so heavy layers can be skipped for local sanity checks:
  - `INSTALL_PYTHON_STACK`
  - `INSTALL_LLAMA_CPP`
  - `BUILD_WORKSPACE`

## Key Decisions

- Kept the base image on CUDA `13.2` because that was explicitly requested, even though upstream PyTorch/vLLM packaging is presently centered on CUDA `13.0`.
- Installed PyTorch and vLLM against the CUDA `13.0` wheel line inside the CUDA `13.2` image because that is the most practical binary-first path available today.
- Built `llama.cpp` from source because upstream usage in this repo is developer-oriented and the user explicitly wanted it present in the image.
- Added skip-style build args so Dockerfile validation does not require rebuilding the full Python and CUDA toolchain every time.

## Verification

Completed verification:

```bash
docker build \
  --platform linux/amd64 \
  --build-arg INSTALL_PYTHON_STACK=0 \
  --build-arg INSTALL_LLAMA_CPP=0 \
  --build-arg BUILD_WORKSPACE=0 \
  -t nemotron-cutile-dev:syntax-check .

docker build \
  --platform linux/amd64 \
  --build-arg INSTALL_PYTHON_STACK=1 \
  --build-arg INSTALL_LLAMA_CPP=0 \
  --build-arg BUILD_WORKSPACE=0 \
  -t nemotron-cutile-dev:python-check .

cargo test --workspace --quiet
```

Observed results:

- Remote `tn` syntax-check build: passed
- Remote `tn` Python stack build: passed
- Local workspace tests: passed (`211 passed`, `1 ignored`)
- Remote `llama.cpp` build: reached CUDA 13.2 configuration and active CUDA compilation on `tn`, but the SSH connection to `tn` timed out before completion and the host stayed unreachable afterward

## Open Gaps

1. `tn` is still on NVIDIA driver `580.126.20`, so running a CUDA `13.2` container there remains blocked until the host driver is upgraded.
2. The Python package layer is intentionally practical rather than fully pinned end-to-end; exact upstream wheel contents may change over time.
3. `llama.cpp` configuration/build was only partially verified because the `tn` host stopped responding over SSH mid-build.
