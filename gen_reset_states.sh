DATASET_DIR=${DATASET_DIR:-./Datasets/OmniReset}

# Object Anywhere, End-Effector Anywhere (Reaching)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 4096 --num_reset_states 10000 --headless \
    env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop

# Object Resting, End-Effector Grasped (Near Object)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 4096 --num_reset_states 10000 --headless \
    env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop \
    env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=${DATASET_DIR} \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}

# Object Anywhere, End-Effector Grasped (Grasped)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 4096 --num_reset_states 10000 --headless \
    env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}

# Object Partially Assembled, End-Effector Grasped (Near Goal)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 \
    --dataset_dir ${DATASET_DIR} \
    --num_envs 4096 --num_reset_states 10000 --headless \
    env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop \
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=${DATASET_DIR} \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}
