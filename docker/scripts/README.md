# UWLab Docker launchers

These wrappers run the matching `scripts_v2` launcher in
`bowenli1024/physcoder-uwlab:latest`. Run them from any directory in the
checkout, for example:

```bash
./docker/scripts/train_from_scratch.sh
./docker/scripts/play_cube_resync.sh
./docker/scripts/tools/collect_resets_box_block.sh
```

The wrappers expose host GPUs 0, 1, 2, and 3 by default, use Isaac Sim's `python.sh`, and
bind-mount `source`, `scripts`, `scripts_v2`, `Datasets`, `logs`, `outputs`, and
`data_storage`. Generated data therefore remains in the host checkout. Isaac
Sim shader, pip, GL, and CUDA compute caches use persistent Docker volumes.

## Configuration

Environment variables supported by every launcher:

- `UWLAB_DOCKER_IMAGE`: image name; defaults to
  `bowenli1024/physcoder-uwlab:latest`.
- `UWLAB_GPUS`: Docker GPU request; defaults to `0,1,2,3`. For another subset, use a
  comma-separated list such as `UWLAB_GPUS=0,1` (the `device=0,1` spelling is
  accepted too).
- `CUDA_VISIBLE_DEVICES`: GPUs visible inside the container; defaults to
  `0,1,2,3`. Override this together with `UWLAB_GPUS` when changing the GPU
  selection.
- `UWLAB_SHM_SIZE`: shared-memory size; defaults to `16g`.
- `UWLAB_ENV_FILE`: optional Docker environment file, useful for W&B settings.
- `UWLAB_CONTAINER_NAME`: override the generated container name.
- `UWLAB_KEEP_CONTAINER=1`: retain the stopped container for debugging.
- `UWLAB_DRY_RUN=1`: print the fully quoted `docker run` command without
  starting it.

W&B authentication should be supplied with `WANDB_API_KEY` (or through
`UWLAB_ENV_FILE`). Host `wandb login` state is not mounted into the container.
Common W&B settings, including `WANDB_ENTITY`, `WANDB_USERNAME`, `WANDB_MODE`,
`WANDB_PROJECT`, `WANDB_RUN_GROUP`, `WANDB_TAGS`, `WANDB_NOTES`, `WANDB_RUN_ID`,
and `WANDB_RESUME`, are forwarded when present in the host environment. W&B's
local files are written below the host-mounted `logs/wandb` directory, so
offline runs survive removal of the container.

For RSL-RL launchers, use `--log_project_name` to reliably override the W&B
project because the runner passes its configured project directly to
`wandb.init`. The installed RSL-RL logger also reads `WANDB_USERNAME` as its
explicit entity setting.

`DATASET_DIR` is also forwarded when present in the host environment.

The `train_from_scratch.sh` launcher explicitly starts four distributed
processes on GPUs 0--3. Other launchers use the selected GPU set but do not
enable distributed execution implicitly.

`play_cube_resync.sh` is a GUI launcher for the ReSYNC cube policy. It uses GPU
0 by default, forwards the host X11 display with a temporary authorization
cookie, and runs four vectorized environments with ObjectAnywhereEEAnywhere
resets. It requires `DISPLAY`, `/tmp/.X11-unix`, and the host `xauth` command.
