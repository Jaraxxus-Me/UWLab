python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-v0 \
    --num_envs 8192 \
    --logger wandb \
    --headless \
    --resume_path cube_state_rl_expert_seed42.pt \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset