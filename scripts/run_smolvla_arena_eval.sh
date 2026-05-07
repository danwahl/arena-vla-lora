#!/usr/bin/env bash
# Drive Arena's policy_runner with our SmolVLA LeRobot adapter.
# Bypasses the LeRobot EnvHub adapter (version-skewed against current Arena).
#
# Usage: bash scripts/run_smolvla_arena_eval.sh <env_name> <ckpt_path> [num] [num_envs] [embodiment] [mode]
# Example:
#   bash scripts/run_smolvla_arena_eval.sh \
#     gr1_open_microwave \
#     outputs/train/smolvla_microwave_lora/checkpoints/last/pretrained_model \
#     100 1 gr1_pink episodes
#
# Env vars (all optional):
#   ARENA_IMAGE          override docker image (default: isaaclab_arena:latest)
#   ISAACLAB_ARENA_DIR   path to IsaacLab-Arena checkout (default: $HOME/IsaacLab-Arena)
#   HF_TOKEN             pass-through for gated HF assets
set -euo pipefail

ENV_NAME="${1:?need env name}"
CKPT="${2:?need checkpoint path (host path under repo or HF repo_id)}"
NUM="${3:-100}"
NUM_ENVS="${4:-1}"
EMBODIMENT="${5:-gr1_pink}"
MODE="${6:-episodes}"   # "episodes" triggers Arena's metric aggregation; "steps" doesn't

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARENA_IMAGE="${ARENA_IMAGE:-isaaclab_arena:latest}"
ISAACLAB_ARENA_DIR="${ISAACLAB_ARENA_DIR:-$HOME/IsaacLab-Arena}"

# Rewrite host checkpoint path → in-container path. HF repo_ids pass through unchanged.
if [[ "$CKPT" == "$REPO_ROOT"/* ]]; then
  CKPT_IN_CONTAINER="${CKPT/$REPO_ROOT/\/arena-vla-lora}"
else
  CKPT_IN_CONTAINER="$CKPT"
fi

OUT_DIR="$REPO_ROOT/outputs/eval/smolvla_arena_$(date +%Y%m%d_%H%M%S)"
OUT_DIR_IN_CONTAINER="${OUT_DIR/$REPO_ROOT/\/arena-vla-lora}"
mkdir -p "$OUT_DIR" "$REPO_ROOT/outputs/eval/arena_recordings"

docker run --rm --gpus=all --privileged \
  --ulimit memlock=-1 --ulimit stack=-1 \
  --ipc=host --net=host --runtime=nvidia \
  --env DOCKER_RUN_USER_ID=$(id -u) --env DOCKER_RUN_USER_NAME=$(id -un) \
  --env DOCKER_RUN_GROUP_ID=$(id -g) --env DOCKER_RUN_GROUP_NAME=$(id -gn) \
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  --env PYTHONPATH=/arena-vla-lora/install:/arena-vla-lora/scripts \
  --env PYTHONUNBUFFERED=1 \
  -v "$ISAACLAB_ARENA_DIR:/workspaces/isaaclab_arena" \
  -v "$REPO_ROOT:/arena-vla-lora" \
  -v "$HOME/.cache:/root/.cache" \
  -v "$REPO_ROOT/outputs/eval/arena_recordings:/tmp/isaaclab/logs" \
  --entrypoint /bin/bash \
  "$ARENA_IMAGE" \
  -c "cd /workspaces/isaaclab_arena && \
      PYTHONPATH=/arena-vla-lora/install:/arena-vla-lora/scripts:\${PYTHONPATH:-} \
      /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
        --policy_type smolvla_arena_policy.SmolVLALeRobotPolicy \
        --policy_checkpoint '${CKPT_IN_CONTAINER}' \
        $([ "${MODE}" = "episodes" ] && echo "--num_episodes ${NUM}" || echo "--num_steps ${NUM}") \
        --num_envs ${NUM_ENVS} \
        --enable_cameras \
        ${ENV_NAME} \
        --embodiment ${EMBODIMENT} \
        2>&1 | tee '${OUT_DIR_IN_CONTAINER}/run.log'"

echo "[done] eval output: $OUT_DIR"
