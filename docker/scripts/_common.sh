#!/usr/bin/env bash

set -euo pipefail

readonly DOCKER_SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly UWLAB_REPO_ROOT="$(cd -- "${DOCKER_SCRIPTS_DIR}/../.." && pwd)"

docker_uwlab_run() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: docker_uwlab_run <repo-relative-script> [script arguments ...]" >&2
        return 2
    fi

    local repo_script="$1"
    shift

    local image="${UWLAB_DOCKER_IMAGE:-bowenli1024/physcoder-uwlab:latest}"
    local gpu_request="${UWLAB_GPUS:-all}"
    local container_name="${UWLAB_CONTAINER_NAME:-uwlab-$(basename "${repo_script}" .sh)-$$}"
    local container_script="/workspace/uwlab/${repo_script}"
    local host_script="${UWLAB_REPO_ROOT}/${repo_script}"

    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker is required but was not found in PATH." >&2
        return 127
    fi
    if [[ ! -f "${host_script}" ]]; then
        echo "UWLab script not found: ${host_script}" >&2
        return 2
    fi

    # Accept a convenient comma-separated GPU list while preserving the
    # quoting Docker requires around multi-device requests.
    if [[ "${gpu_request}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        gpu_request="\"device=${gpu_request}\""
    elif [[ "${gpu_request}" == device=*","* ]]; then
        gpu_request="\"${gpu_request}\""
    fi

    # These paths contain generated artifacts and should survive the container.
    mkdir -p \
        "${UWLAB_REPO_ROOT}/Datasets" \
        "${UWLAB_REPO_ROOT}/data_storage" \
        "${UWLAB_REPO_ROOT}/logs" \
        "${UWLAB_REPO_ROOT}/outputs"

    local -a docker_args=(
        run
        --attach stdout
        --attach stderr
        --gpus "${gpu_request}"
        --network host
        --shm-size "${UWLAB_SHM_SIZE:-16g}"
        --ulimit memlock=-1
        --ulimit stack=67108864
        --name "${container_name}"
        --entrypoint /bin/bash
        -e ACCEPT_EULA=Y
        -e PRIVACY_CONSENT=Y
        -e OMNI_KIT_ALLOW_ROOT=1
        -e PYTHONUNBUFFERED=1
        -e UWLAB_PATH=/workspace/uwlab
        -e ISAACSIM_PATH=/isaac-sim
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/source,dst=/workspace/uwlab/source"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/scripts,dst=/workspace/uwlab/scripts"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/scripts_v2,dst=/workspace/uwlab/scripts_v2"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/tools,dst=/workspace/uwlab/tools"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/uwlab.sh,dst=/workspace/uwlab/uwlab.sh,readonly"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/Datasets,dst=/workspace/uwlab/Datasets"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/data_storage,dst=/workspace/uwlab/data_storage"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/logs,dst=/workspace/uwlab/logs"
        --mount "type=bind,src=${UWLAB_REPO_ROOT}/outputs,dst=/workspace/uwlab/outputs"
        --mount type=volume,src=physcoder-uwlab-kit-cache,dst=/isaac-sim/kit/cache
        --mount type=volume,src=physcoder-uwlab-ov-cache,dst=/root/.cache/ov
        --mount type=volume,src=physcoder-uwlab-pip-cache,dst=/root/.cache/pip
        --mount type=volume,src=physcoder-uwlab-gl-cache,dst=/root/.cache/nvidia/GLCache
        --mount type=volume,src=physcoder-uwlab-compute-cache,dst=/root/.nv/ComputeCache
    )

    if [[ "${UWLAB_KEEP_CONTAINER:-0}" != "1" ]]; then
        docker_args+=(--rm)
    fi
    if [[ -t 0 && -t 1 ]]; then
        docker_args+=(-it)
    fi

    local env_name
    for env_name in \
        CUDA_VISIBLE_DEVICES \
        DATASET_DIR \
        WANDB_API_KEY \
        WANDB_ENTITY \
        WANDB_MODE \
        WANDB_PROJECT; do
        if [[ -n "${!env_name:-}" ]]; then
            docker_args+=(-e "${env_name}")
        fi
    done

    if [[ -n "${UWLAB_ENV_FILE:-}" ]]; then
        if [[ ! -f "${UWLAB_ENV_FILE}" ]]; then
            echo "UWLAB_ENV_FILE does not exist: ${UWLAB_ENV_FILE}" >&2
            return 2
        fi
        docker_args+=(--env-file "${UWLAB_ENV_FILE}")
    fi

    # Mount optional root-level inputs used by sysid and pretrained launchers.
    local input_path
    for input_path in \
        "${UWLAB_REPO_ROOT}"/*.pt \
        "${UWLAB_REPO_ROOT}"/*.ckpt \
        "${UWLAB_REPO_ROOT}"/0511_resume \
        "${UWLAB_REPO_ROOT}"/references; do
        if [[ -e "${input_path}" ]]; then
            docker_args+=(--mount "type=bind,src=${input_path},dst=/workspace/uwlab/$(basename "${input_path}")")
        fi
    done

    docker_args+=(
        "${image}"
        -lc
        # Export a Bash function so every legacy `python ...` line uses Isaac Sim's
        # environment-aware Python launcher without rewriting the original scripts.
        'set -euo pipefail
        for required_path in source scripts scripts_v2 tools uwlab.sh Datasets data_storage logs outputs; do
            if [[ ! -e "/workspace/uwlab/${required_path}" ]]; then
                echo "[uwlab-docker] Missing host bind mount: /workspace/uwlab/${required_path}" >&2
                exit 2
            fi
        done
        python() { /isaac-sim/python.sh "$@"; }
        export -f python
        cd /workspace/uwlab
        echo "[uwlab-docker] Live host checkout mounted at /workspace/uwlab; streaming container stdout/stderr." >&2
        exec bash "$@"'
        uwlab-docker
        "${container_script}"
        "$@"
    )

    if [[ "${UWLAB_DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' docker "${docker_args[@]}"
        printf '\n'
        return 0
    fi

    exec docker "${docker_args[@]}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This file is a helper; run one of the other scripts in docker/scripts." >&2
    exit 2
fi
