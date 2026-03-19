#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-nemotron-cutile-dev}"
CONTAINER_NAME="${CONTAINER_NAME:-nemotron-cutile-dev}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace}"
USER_NAME="${USER_NAME:-$(id -un)}"
USER_UID="${USER_UID:-$(id -u)}"
USER_GID="${USER_GID:-$(id -g)}"
INSTALL_PYTHON_STACK="${INSTALL_PYTHON_STACK:-1}"
INSTALL_LLAMA_CPP="${INSTALL_LLAMA_CPP:-0}"
BUILD_WORKSPACE="${BUILD_WORKSPACE:-0}"
PUBLISH_PORTS="${PUBLISH_PORTS:-8000:8000}"

usage() {
    cat <<'EOF'
Usage: ./dev.sh <command>

Commands:
  build   Build the development image.
  run     Recreate and start the long-lived dev container for VSCode attach.
  start   Start an existing stopped dev container.
  stop    Stop the dev container.
  exec    Open a shell inside the running dev container.
  logs    Show container logs.
  ps      Show the dev container status.
  rm      Remove the dev container.

Environment overrides:
  IMAGE_NAME
  CONTAINER_NAME
  CONTAINER_WORKSPACE
  USER_NAME
  USER_UID
  USER_GID
  INSTALL_PYTHON_STACK
  INSTALL_LLAMA_CPP
  BUILD_WORKSPACE
  PUBLISH_PORTS
EOF
}

require_docker() {
    command -v docker >/dev/null 2>&1 || {
        echo "docker is required" >&2
        exit 1
    }
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
    docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

build_image() {
    require_docker
    docker build \
        --build-arg USERNAME="${USER_NAME}" \
        --build-arg USER_UID="${USER_UID}" \
        --build-arg USER_GID="${USER_GID}" \
        --build-arg INSTALL_PYTHON_STACK="${INSTALL_PYTHON_STACK}" \
        --build-arg INSTALL_LLAMA_CPP="${INSTALL_LLAMA_CPP}" \
        --build-arg BUILD_WORKSPACE="${BUILD_WORKSPACE}" \
        -t "${IMAGE_NAME}" \
        "${WORKSPACE_DIR}"
}

run_container() {
    require_docker

    if container_exists; then
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    fi

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --hostname "${CONTAINER_NAME}" \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --cap-add SYS_PTRACE \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e TERM="${TERM:-xterm-256color}" \
        -p "${PUBLISH_PORTS}" \
        -v "${WORKSPACE_DIR}:${CONTAINER_WORKSPACE}:rw" \
        -w "${CONTAINER_WORKSPACE}" \
        "${IMAGE_NAME}"
}

start_container() {
    require_docker
    docker start "${CONTAINER_NAME}"
}

stop_container() {
    require_docker
    docker stop "${CONTAINER_NAME}"
}

exec_container() {
    require_docker
    docker exec -it "${CONTAINER_NAME}" bash
}

logs_container() {
    require_docker
    docker logs -f "${CONTAINER_NAME}"
}

ps_container() {
    require_docker
    docker ps -a --filter "name=^${CONTAINER_NAME}$"
}

rm_container() {
    require_docker
    docker rm -f "${CONTAINER_NAME}"
}

command="${1:-}"
case "${command}" in
    build)
        build_image
        ;;
    run)
        build_image
        run_container
        ;;
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    exec)
        exec_container
        ;;
    logs)
        logs_container
        ;;
    ps)
        ps_container
        ;;
    rm)
        rm_container
        ;;
    ""|-h|--help|help)
        usage
        ;;
    *)
        echo "unknown command: ${command}" >&2
        usage >&2
        exit 1
        ;;
esac
