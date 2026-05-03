#!/usr/bin/env bash
set -euo pipefail

python scripts_v2/tools/sim2real/sysid_ur5e_osc.py --headless \
    --num_envs 512 \
    --real_data sysid_data_real.pt \
    --max_iter 300 \
    --sigma 0.25 \
    --armature_min_per_joint 0,0,0,0,0,0 \
    --armature_max_per_joint 15,15,15,3,3,3 \
    --friction_min_per_joint 0,0,0,0,0,0 \
    --friction_max_per_joint 40,40,40,12,12,12 \
    --viscous_friction_min_per_joint 0,0,0,0,0,0 \
    --viscous_friction_max_per_joint 50,50,50,15,15,15 \
    --delay_max 10
