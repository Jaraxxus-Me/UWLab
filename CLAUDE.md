# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UW Lab is a robotics simulation and reinforcement learning framework built on top of **Isaac Lab** and **NVIDIA Isaac Sim**. It provides manager-based robotic environments, sim-to-real transfer, and OmniReset (RL for manipulation without reward engineering). Requires Python 3.11, Isaac Sim 5.1.0, and CUDA.

## Common Commands

All commands go through the `./uwlab.sh` wrapper script:

```bash
# Environment setup
./uwlab.sh -c [name]              # Create conda env (default: env_uwlab)
./uwlab.sh -u [name]              # Create uv env (default: env_uwlab)
./uwlab.sh -i                     # Install all dependencies (Isaac Lab + UW Lab packages)
./uwlab.sh -i none                # Install without RL frameworks

# Running scripts
./uwlab.sh -p <script.py> [args]  # Run a Python script with the correct interpreter

# Testing
./uwlab.sh -p tools/run_all_tests.py              # Run all tests
./uwlab.sh -p tools/run_all_tests.py --timeout 1000  # With custom timeout
./uwlab.sh -t [pytest-args]                        # Run pytest directly on tools/

# Code formatting (runs pre-commit: black, isort, flake8)
./uwlab.sh -f

# Documentation
./uwlab.sh -d                     # Build Sphinx docs
```

## Code Style

- **Formatter**: black with `--line-length 120`
- **Linter**: flake8 with `--max-line-length 120`, max-complexity 30
- **Import sorting**: isort with black profile and custom section ordering (see `pyproject.toml`)
  - Sections: STDLIB > THIRDPARTY > ASSETS_FIRSTPARTY (`uwlab_assets`) > FIRSTPARTY (`uwlab`) > EXTRA_FIRSTPARTY (`uwlab_rl`) > TASK_FIRSTPARTY (`uwlab_tasks`) > LOCALFOLDER
  - Common scientific packages (numpy, torch, scipy, etc.) are classified as extra standard library
- **Docstrings**: Google convention
- **License headers**: Required on all source files (BSD-3-Clause, or Apache 2.0 for mimic/imitation learning code)
- **Type checking**: pyright in basic mode

## Architecture

Four source packages under `source/`, each installable independently via `pip install -e`:

| Package | Purpose |
|---------|---------|
| **uwlab** | Core framework: simulation interface, environments, controllers, actuators, terrains, scene management, managers (observation/action/reward), utilities |
| **uwlab_tasks** | Task definitions in `manager_based/` and `direct/` formats |
| **uwlab_rl** | RL training utilities and wrappers; optional dependency on rsl-rl from UW-Lab GitHub |
| **uwlab_assets** | Robot models, props, and asset data |

The framework extends Isaac Lab (cloned to `_isaaclab/IsaacLab/` during install) and depends on Isaac Sim (symlinked or pip-installed at `_isaac_sim/`).

### Training Scripts

- `scripts/reinforcement_learning/` - RL training and evaluation
- `scripts/imitation_learning/` - Imitation/distillation learning
- `scripts/environments/` - Environment demos and utilities

## CI/CD

PRs trigger two parallel test jobs on a self-hosted GPU runner:
- `test-uwlab-tasks` (50 min timeout) - task-specific tests
- `test-general` (180 min timeout) - all other packages

Pre-commit checks (black, flake8, isort, pyupgrade, codespell, license headers) run on all PRs.
