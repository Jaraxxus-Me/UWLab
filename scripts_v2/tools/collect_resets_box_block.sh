DATASET_DIR=${DATASET_DIR:-./Datasets/PhysCoder}

python scripts_v2/tools/record_partial_assemblies.py \
    --task OmniReset-UR5eRobotiq2f140-PartialAssemblies-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 10 \
    --num_trajectories 10 \
    --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder

python scripts_v2/tools/record_grasps.py \
    --task OmniReset-Robotiq2f140-GraspSampling-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 8192 \
    --num_grasps 1000 \
    --headless \
    env.scene.object=block_physcoder

# # Object Anywhere, End-Effector Anywhere (Reaching)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectAnywhereEEAnywhere-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 8192 --num_reset_states 10000 --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder

# Object Resting, End-Effector Grasped (Near Object)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectRestingEEGrasped-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 8192 --num_reset_states 10000 --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder \
    env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=${DATASET_DIR} \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}

# Object Anywhere, End-Effector Grasped (Grasped)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectAnywhereEEGrasped-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 8192 --num_reset_states 10000 --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}

# Object Partially Assembled, End-Effector Grasped (Near Goal)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectPartiallyAssembledEEGrasped-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 8192 --num_reset_states 10000 --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder \
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=${DATASET_DIR} \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}
