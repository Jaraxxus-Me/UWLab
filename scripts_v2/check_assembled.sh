#!/usr/bin/env bash
set -euo pipefail

python scripts_v2/tools/record_partial_assemblies.py \
    --task OmniReset-UR5eRobotiq2f140-PartialAssemblies-v0 \
    --num_envs 10 --num_trajectories 10 --dataset_dir ./Datasets/OmniReset/ResyncVerification \
    env.scene.insertive_object=cube_resync env.scene.receptive_object=region_resync

python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectAnywhereEEAnywhere-v0 \
    --num_envs 4 --num_reset_states 8 --dataset_dir ./Datasets/OmniReset/ResyncVerification \
    env.scene.insertive_object=cube_resync env.scene.receptive_object=region_resync

python scripts_v2/tools/visualize_reset_states.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-v0 \
    --num_envs 4 --dataset_dir ./Datasets/OmniReset/ResyncVerification \
    --reset_type ObjectAnywhereEEAnywhere \
    env.scene.insertive_object=cube_resync env.scene.receptive_object=region_resync
