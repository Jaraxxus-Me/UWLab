DATASET_DIR=${DATASET_DIR:-./Datasets/PhysCoder}
VIDEO_DIR=${VIDEO_DIR:-./outputs/reset_videos_0811_256}
RESET_NUM_ENVS=${RESET_NUM_ENVS:-256}
NUM_RESET_STATES=${NUM_RESET_STATES:-500}

python scripts_v2/tools/record_partial_assemblies.py \
    --task OmniReset-UR5eRobotiq2f140-PartialAssemblies-v0 \
    --dataset_dir "${DATASET_DIR}" \
    --num_envs 10 \
    --num_trajectories 10 \
    --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder

python scripts_v2/tools/record_grasps.py \
    --task OmniReset-Robotiq2f140-GraspSampling-v0 \
    --dataset_dir "${DATASET_DIR}" \
    --num_envs 8192 \
    --num_grasps 1000 \
    --headless \
    env.scene.object=block_physcoder

# Object Anywhere, End-Effector Anywhere (Reaching)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectAnywhereEEAnywhere-v0 \
    --dataset_dir "${DATASET_DIR}" \
    --num_envs "${RESET_NUM_ENVS}" --num_reset_states "${NUM_RESET_STATES}" --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder

# Object Resting, End-Effector Grasped (Near Object)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectRestingEEGrasped-v0 \
    --dataset_dir "${DATASET_DIR}" \
    --num_envs "${RESET_NUM_ENVS}" --num_reset_states "${NUM_RESET_STATES}" --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder \
    env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir="${DATASET_DIR}" \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="${DATASET_DIR}"

# Object Anywhere, End-Effector Grasped (Grasped)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectAnywhereEEGrasped-v0 \
    --dataset_dir "${DATASET_DIR}" \
    --num_envs "${RESET_NUM_ENVS}" --num_reset_states "${NUM_RESET_STATES}" --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="${DATASET_DIR}"

# Object Partially Assembled, End-Effector Grasped (Near Goal)
python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f140-ObjectPartiallyAssembledEEGrasped-v0 \
    --dataset_dir "${DATASET_DIR}" \
    --num_envs "${RESET_NUM_ENVS}" --num_reset_states "${NUM_RESET_STATES}" --headless \
    env.scene.insertive_object=block_physcoder env.scene.receptive_object=box_physcoder \
    env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir="${DATASET_DIR}" \
    env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir="${DATASET_DIR}"

for RESET_TYPE in \
    ObjectAnywhereEEAnywhere \
    ObjectRestingEEGrasped \
    ObjectAnywhereEEGrasped \
    ObjectPartiallyAssembledEEGrasped
do
    EXTRA_OVERRIDES=()
    if [[ "${RESET_TYPE}" == "ObjectRestingEEGrasped" ]]; then
        EXTRA_OVERRIDES+=(
            "env.events.reset_insertive_object_pose_from_reset_states.params.dataset_dir=${DATASET_DIR}"
            "env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}"
        )
    elif [[ "${RESET_TYPE}" == "ObjectAnywhereEEGrasped" ]]; then
        EXTRA_OVERRIDES+=("env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}")
    elif [[ "${RESET_TYPE}" == "ObjectPartiallyAssembledEEGrasped" ]]; then
        EXTRA_OVERRIDES+=(
            "env.events.reset_insertive_object_pose_from_partial_assembly_dataset.params.dataset_dir=${DATASET_DIR}"
            "env.events.reset_end_effector_pose_from_grasp_dataset.params.dataset_dir=${DATASET_DIR}"
        )
    fi

    python scripts_v2/tools/render_reset_videos.py \
        --task "OmniReset-UR5eRobotiq2f140-${RESET_TYPE}-v0" \
        --output_dir "${VIDEO_DIR}" \
        --source_num_envs "${RESET_NUM_ENVS}" \
        --num_videos 10 \
        --headless --enable_cameras \
        env.scene.insertive_object=block_physcoder \
        env.scene.receptive_object=box_physcoder \
        "${EXTRA_OVERRIDES[@]}"
done
