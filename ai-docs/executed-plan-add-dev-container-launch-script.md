---
status: implemented-with-open-gaps
goal: Add a repeatable Docker development launcher that bind mounts the repository read-write for VSCode attach workflows
prompt: "create dev.sh scripts, build and run Docker container, and bind mount the git repo rw for VSCode attach to container development"
created: 2026-03-19T13:01:09Z
finished: 2026-03-19T13:20:30Z
---

# Executed Plan — Add Dev Container Launch Script

## Summary

Added a root `dev.sh` helper for building and running a long-lived GPU development container and updated the Docker image so it is usable for bind-mounted, VSCode-attach development rather than only one-shot builds.

## What Was Done

- Added `dev.sh` with subcommands for `build`, `run`, `start`, `stop`, `exec`, `logs`, `ps`, and `rm`.
- Configured `dev.sh run` to:
  - build the image
  - run the container detached
  - mount the repository at `/workspace` read-write
  - request `--gpus all`
  - keep the container alive with the image default command so VSCode can attach later
- Updated `Dockerfile` to create a non-root user from build args (`USER_UID`, `USER_GID`, `USERNAME`) so edits inside the mounted repo preserve host ownership semantics.
- Updated the Docker image environment to match the repo's cargo CUDA/LLVM expectations.
- Switched the image default command from `bash` to `sleep infinity` so the container remains attachable after startup.

## Key Decisions

- Defaulted `BUILD_WORKSPACE=0` for the dev launcher because the bind-mounted repo will replace `/workspace` anyway and VSCode attach flows usually want a fast image build.
- Defaulted `INSTALL_LLAMA_CPP=0` in `dev.sh` because `llama.cpp` is supported by the Dockerfile but expensive enough that it should be opt-in for normal editor-attach bring-up.
- Kept the launcher at repo root as `dev.sh` so it is easy to run directly on the server checkout.
- Used a detached long-lived container model rather than `docker run --rm -it` because VSCode attach requires a stable running container.

## Verification

Verification completed on `tn` with one remaining containerized Rust-build blocker:

```bash
bash -n dev.sh
cargo test --workspace --quiet
./dev.sh run
docker ps --filter name=^nemotron-cutile-dev$
docker inspect nemotron-cutile-dev
docker exec nemotron-cutile-dev bash -lc 'id && pwd'
docker exec nemotron-cutile-dev bash -lc 'python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"'
docker exec nemotron-cutile-dev nvidia-smi -L
docker exec nemotron-cutile-dev bash -lc 'cd /workspace && cargo build --workspace --quiet'
```

Observed results:

- `bash -n dev.sh`: passed
- local `cargo test --workspace --quiet`: passed (`211 passed`, `1 ignored`)
- `tn` image build: passed
- `tn` dev container startup: passed
- container mount: `/home/trungnt13/codes/nemotron-cutile-rs -> /workspace` with `rw=true`
- container user mapping: `uid=1000(trungnt13) gid=1000(trungnt13)`
- in-container GPU visibility: passed
  - `torch==2.9.1+cu130`
  - `torch.cuda.is_available() == True`
  - `nvidia-smi -L` reports the RTX 3090
- in-container `cargo build --workspace`: still blocked in the `cutile` toolchain setup

## Open Gaps

1. The container itself now starts with GPU access on `tn`, but the mounted-repo Rust build still fails in `cutile`'s LLVM/MLIR CMake path.
2. The current blocker is no longer `llvm-config` or missing MLIR headers; it is missing CMake-imported dependency targets from the prebuilt LLVM install path, starting with `ZLIB::ZLIB`, and the log also notes `CURL` and `LibEdit` were not found.
3. `INSTALL_LLAMA_CPP=0` remains the dev-launcher default because it keeps editor-attach bring-up practical; full `llama.cpp` remains available through Docker build args when explicitly needed.
