#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../_common.sh"
docker_uwlab_run scripts_v2/tools/collect_resets_box_block.sh "$@"
