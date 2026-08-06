#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-logs/0809_trained_cube_resync/model_1000.pt}
DATASET_DIR=${DATASET_DIR:-./Datasets/ReSYNC_Rebuttal}

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Play-v0 \
    --num_envs 4 \
    --checkpoint "${CHECKPOINT}" \
    --real-time \
    env.scene.insertive_object=cube_resync \
    env.scene.receptive_object=region_resync \
    env.events.reset_from_reset_states.params.dataset_dir="${DATASET_DIR}" \
    env.events.reset_from_reset_states.params.reset_types='["ObjectAnywhereEEAnywhere"]' \
    env.events.reset_from_reset_states.params.probs='[1.0]' \
    "$@"
