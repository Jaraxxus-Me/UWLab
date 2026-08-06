#!/usr/bin/env bash
set -euo pipefail

export UWLAB_ENABLE_X11=${UWLAB_ENABLE_X11:-1}
export UWLAB_GPUS=${UWLAB_GPUS:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
docker_uwlab_run scripts_v2/play_cube_resync.sh "$@"
