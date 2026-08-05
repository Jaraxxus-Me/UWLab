DATASET_DIR=${DATASET_DIR:-./Datasets/ReSYNC_Rebuttal}

python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-v0 \
    --num_envs 16384 \
    --logger wandb \
    --headless \
    --distributed \
    env.scene.insertive_object=cube_resync \
    env.scene.receptive_object=region_resync \
    env.events.reset_from_reset_states.params.dataset_dir="${DATASET_DIR}" \
    "$@"
