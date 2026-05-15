python scripts_v2/tools/collect_demos.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-RGB-DataCollection-v0 \
    --dataset_file Datasets/OmniReset_2f140/rgb0_final.zarr \
    --num_envs 32 \
    --num_demos 100000 \
    --enable_cameras \
    --headless \
    env.scene.insertive_object=block \
    env.scene.receptive_object=box \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset_2f140 \
    env.events.reset_from_reset_states.params.reset_types='["ObjectAnywhereEEAnywhere"]' \
    env.events.reset_from_reset_states.params.probs='[1.0]' \
    agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path='["logs/rsl_rl/ur5e_robotiq_2f140_omnireset_agent/0511_after_adr/exported/policy.pt"]'
