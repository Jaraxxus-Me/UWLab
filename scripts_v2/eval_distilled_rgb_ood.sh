python scripts_v2/tools/eval_distilled_policy.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-RGB-OOD-Play-v0 \
    --checkpoint logs/rgb/0509/step_0030000.ckpt \
    --num_envs 4 \
    --num_trajectories 20 \
    --headless \
    --enable_cameras \
    --save_video \
    env.scene.insertive_object=block \
    env.scene.receptive_object=box \
    env.events.reset_from_reset_states.params.reset_types='["ObjectAnywhereEEAnywhere"]' \
    env.events.reset_from_reset_states.params.probs='[1.0]' \
    env.events.reset_from_reset_states.params.dataset_dir=./Datasets/OmniReset_2f140