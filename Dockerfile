# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG RUST_VERSION=1.94.0
ARG RUST_NIGHTLY_TOOLCHAIN=nightly
ARG PYTORCH_CUDA=cu130
ARG VLLM_VERSION=0.16.0
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_CPP_REF=master
ARG CMAKE_CUDA_ARCHITECTURES=86
ARG INSTALL_PYTHON_STACK=1
ARG INSTALL_LLAMA_CPP=1
ARG BUILD_WORKSPACE=1
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    VIRTUAL_ENV=/opt/venv \
    RUST_STABLE_TOOLCHAIN=${RUST_VERSION} \
    RUSTUP_TOOLCHAIN=${RUST_NIGHTLY_TOOLCHAIN} \
    CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2 \
    CUDA_PATH=/usr/local/cuda-13.2 \
    CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21 \
    LLVM_SYS_210_PREFIX=/usr/lib/llvm-21 \
    LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config \
    MLIR_SYS_210_PREFIX=/usr/lib/llvm-21 \
    BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/llvm-21/include \
    PATH=/usr/local/cargo/bin:/opt/venv/bin:/usr/lib/llvm-21/bin:/usr/local/cuda/bin:${PATH} \
    CARGO_TERM_COLOR=always \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null \
    && echo 'deb http://apt.llvm.org/noble/ llvm-toolchain-noble-21 main' \
        >/etc/apt/sources.list.d/llvm-21.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang-21 \
    cmake \
    git \
    libclang-21-dev \
    libcurl4-openssl-dev \
    libedit-dev \
    libmlir-21-dev \
    llvm-21 \
    llvm-21-dev \
    mlir-21-tools \
    libonig-dev \
    libpolly-21-dev \
    libssl-dev \
    libzstd-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    sudo \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda /usr/local/cuda-13.2 \
    && ln -sf /usr/bin/clang-21 /usr/local/bin/clang \
    && ln -sf /usr/bin/clang++-21 /usr/local/bin/clang++ \
    && ln -sf /usr/lib/llvm-21/bin/llvm-config /usr/local/bin/llvm-config

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --upgrade pip setuptools wheel uv

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rustup default "${RUST_VERSION}" \
    && rustup toolchain install "${RUST_NIGHTLY_TOOLCHAIN}" --profile minimal \
    && rustc --version \
    && cargo --version \
    && rustup run "${RUST_NIGHTLY_TOOLCHAIN}" rustc --version

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

RUN group_name="$(getent group "${USER_GID}" | cut -d: -f1 || true)" \
    && if [[ -z "${group_name}" ]]; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    elif [[ "${group_name}" != "${USERNAME}" ]]; then \
        groupmod --new-name "${USERNAME}" "${group_name}"; \
    fi \
    && user_name="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)" \
    && if [[ -z "${user_name}" ]]; then \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --home-dir "/home/${USERNAME}" --shell /bin/bash "${USERNAME}"; \
    elif [[ "${user_name}" != "${USERNAME}" ]]; then \
        usermod --login "${USERNAME}" --home "/home/${USERNAME}" --move-home --gid "${USER_GID}" "${user_name}"; \
    fi \
    && usermod --uid "${USER_UID}" --gid "${USER_GID}" --shell /bin/bash "${USERNAME}" \
    && mkdir -p "/home/${USERNAME}" \
    && usermod -aG sudo "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/"${USERNAME}" \
    && chmod 0440 /etc/sudoers.d/"${USERNAME}" \
    && chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}" "${CARGO_HOME}" "${RUSTUP_HOME}" "${VIRTUAL_ENV}"

WORKDIR /workspace

COPY . .

RUN if [[ "${BUILD_WORKSPACE}" == "1" ]]; then \
        cargo build --workspace; \
    fi

USER ${USERNAME}

CMD ["sleep", "infinity"]
