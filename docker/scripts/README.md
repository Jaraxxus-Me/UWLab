# UWLab Docker launchers

These wrappers run the matching `scripts_v2` launcher in
`bowenli1024/physcoder-uwlab:latest`. Run them from any directory in the
checkout, for example:

```bash
./docker/scripts/train_from_scratch.sh
./docker/scripts/tools/collect_resets_box_block.sh
```

The wrappers expose all host GPUs by default, use Isaac Sim's `python.sh`, and
bind-mount `source`, `scripts`, `scripts_v2`, `Datasets`, `logs`, `outputs`, and
`data_storage`. Generated data therefore remains in the host checkout. Isaac
Sim shader, pip, GL, and CUDA compute caches use persistent Docker volumes.

## Configuration

Environment variables supported by every launcher:

- `UWLAB_DOCKER_IMAGE`: image name; defaults to
  `bowenli1024/physcoder-uwlab:latest`.
- `UWLAB_GPUS`: Docker GPU request; defaults to `all`. For a subset, use a
  comma-separated list such as `UWLAB_GPUS=0,1` (the `device=0,1` spelling is
  accepted too).
- `CUDA_VISIBLE_DEVICES`: optionally restrict the GPUs visible to the process.
- `UWLAB_SHM_SIZE`: shared-memory size; defaults to `16g`.
- `UWLAB_ENV_FILE`: optional Docker environment file, useful for W&B settings.
- `UWLAB_CONTAINER_NAME`: override the generated container name.
- `UWLAB_KEEP_CONTAINER=1`: retain the stopped container for debugging.
- `UWLAB_DRY_RUN=1`: print the fully quoted `docker run` command without
  starting it.

`DATASET_DIR`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_MODE`, and
`WANDB_PROJECT` are forwarded when present in the host environment.

On an eight-GPU machine, all eight GPUs are available inside the container.
The existing scripts launch one Isaac Sim process and therefore normally use
one GPU. Run independent scripts with different `UWLAB_GPUS=device=N` values
to use the machine concurrently. Converting a training job to distributed
RSL-RL changes its effective environment count and optimization behavior, so
these wrappers do not enable `--distributed` implicitly.
