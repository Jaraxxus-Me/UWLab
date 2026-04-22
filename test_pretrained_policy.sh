python demos/test_ur5e_pure_rl.py \
    --checkpoint cube_state_rl_expert_seed42.pt \
    --num_envs 1 --num_episodes 5 --max_steps 1000 \
    env.scene.insertive_object=cube \
    env.scene.receptive_object=cube \
    env.episode_length_s=100.0