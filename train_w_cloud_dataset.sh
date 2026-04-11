python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node 4 \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
    --num_envs 16384 \
    --logger wandb \
    --headless \
    --distributed \
    env.scene.insertive_object=fbleg \
    env.scene.receptive_object=fbtabletop