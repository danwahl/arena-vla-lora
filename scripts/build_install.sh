#!/usr/bin/env bash
# Build the in-container LeRobot install — pip-installs lerobot+peft into ./install/
# using the Arena container's Python. Run once; the resulting ./install/ directory
# is mounted back into the container at eval time via PYTHONPATH.
set -euo pipefail

# Override via env var if you built the GR00T variant: ARENA_IMAGE=isaaclab_arena:cuda_gr00t_gn16
ARENA_IMAGE="${ARENA_IMAGE:-isaaclab_arena:latest}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/install"

docker run --rm --gpus=all \
  --env DOCKER_RUN_USER_ID=$(id -u) --env DOCKER_RUN_USER_NAME=$(id -un) \
  --env DOCKER_RUN_GROUP_ID=$(id -g) --env DOCKER_RUN_GROUP_NAME=$(id -gn) \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
  -v "$REPO_ROOT:/arena-vla-lora" \
  -v "$HOME/.cache:/root/.cache" \
  --entrypoint /bin/bash \
  "${ARENA_IMAGE}" \
  -c "/isaac-sim/python.sh -m pip install --target=/arena-vla-lora/install \
        'lerobot[smolvla]==0.5.1' peft==0.19.1"

echo "[done] install/ built at $REPO_ROOT/install"
