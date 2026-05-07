python scripts_v2/tools/collect_demos.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-RGB-DataCollection-v0 \
    --dataset_file Datasets/OmniResetRealWorkspace/rgb0.zarr \
    --num_envs 32 \
    --num_demos 50000 \
    --enable_cameras \
    --headless \
    env.scene.insertive_object=block \
    env.scene.receptive_object=box \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniResetRealWorkspace \
    agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path='["logs/rsl_rl/ur5e_robotiq_2f140_omnireset_agent/0505_after_adr/exported/policy.pt"]'
