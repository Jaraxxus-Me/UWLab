python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Finetune-v0 \
    --num_envs 16384 \
    --logger wandb \
    --headless \
    --resume_path logs/rsl_rl/ur5e_robotiq_2f140_omnireset_agent/0503_expert/model_3400.pt \
    env.scene.insertive_object=block \
    env.scene.receptive_object=box