# LeRobot + IsaacLab-Arena + LoRA

## Requirements

Linux, NVIDIA GPU with 24+ GB, Docker + uv.

## Setup

```bash
git clone git@github.com:danwahl/arena-vla-lora.git
cd arena-vla-lora
uv sync
cp .env.example .env.local
```

Then edit `.env.local`.

## Train

LeRobot train script is monkey-patched because LeRobot CLI doesn't expose `--peft.lora_alpha`, and requires a shape-safe state-dict loader when changing state/action dimensions. Training takes ~75 min on a single RTX 4090.

```bash
LORA_ALPHA=128 SHAPE_SAFE_LOAD=1 \
LORA_MODULES_TO_SAVE=state_proj,action_in_proj,action_out_proj \
python scripts/lerobot_train_with_alpha.py \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.max_state_dim=64 --policy.max_action_dim=64 \
    --dataset.repo_id=nvidia/Arena-GR1-Manipulation-Task-v3 \
    --rename_map='{"observation.images.robot_pov_cam": "observation.images.camera1"}' \
    --policy.empty_cameras=2 \
    --steps=30000 --batch_size=12 \
    --peft.method_type=LORA --peft.r=64 \
    --eval_freq=0 --save_freq=5000 --log_freq=100 \
    --output_dir=outputs/train/smolvla_microwave_lora \
    --wandb.enable=true --wandb.project=arena-vla-lora
```
