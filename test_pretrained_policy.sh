python demos/test_ur5e_hybrid_policy.py \
    --checkpoint rectangle_state_rl_expert_seed0.pt \
    --num_envs 1 --num_episodes 5 --max_steps 400 \
    env.scene.insertive_object=rectangle \
    env.scene.receptive_object=wall