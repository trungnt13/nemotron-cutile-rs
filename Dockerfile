# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.94.0
ARG PYTORCH_CUDA=cu130
ARG VLLM_VERSION=0.16.0
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_CPP_REF=master
ARG CMAKE_CUDA_ARCHITECTURES=86
ARG INSTALL_PYTHON_STACK=1
ARG INSTALL_LLAMA_CPP=1
ARG BUILD_WORKSPACE=1

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    VIRTUAL_ENV=/opt/venv \
    PATH=/usr/local/cargo/bin:/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    CARGO_TERM_COLOR=always \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    libclang-dev \
    libonig-dev \
    libssl-dev \
    libzstd-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --upgrade pip setuptools wheel uv

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rustup default "${RUST_VERSION}" \
    && rustc --version \
    && cargo --version

RUN if [[ "${INSTALL_PYTHON_STACK}" == "1" ]]; then \
        uv pip install --python "${VIRTUAL_ENV}/bin/python" \
            --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" \
            torch torchvision torchaudio \
        && uv pip install --python "${VIRTUAL_ENV}/bin/python" triton \
        && uv pip install --python "${VIRTUAL_ENV}/bin/python" \
            --index-strategy unsafe-best-match \
            "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${PYTORCH_CUDA}-cp38-abi3-manylinux_2_35_x86_64.whl" \
            --extra-index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"; \
    fi

RUN if [[ "${INSTALL_LLAMA_CPP}" == "1" ]]; then \
        git clone --depth 1 --branch "${LLAMA_CPP_REF}" "${LLAMA_CPP_REPO}" /tmp/llama.cpp \
        && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
            -DGGML_CUDA=ON \
            -DGGML_NATIVE=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
        && cmake --build /tmp/llama.cpp/build --config Release -j"$(nproc)" \
        && install -m 0755 /tmp/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli \
        && install -m 0755 /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server \
        && rm -rf /tmp/llama.cpp; \
    fi

WORKDIR /workspace

COPY . .

RUN if [[ "${BUILD_WORKSPACE}" == "1" ]]; then \
        cargo build --workspace; \
    fi

CMD ["bash"]
