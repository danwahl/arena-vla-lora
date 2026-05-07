"""SmolVLA (LeRobot) policy adapter for IsaacLab-Arena's policy_runner.

Plugs a LeRobot-trained SmolVLA checkpoint into Arena's `policy_runner.py` as a
registered policy ("smolvla_lerobot"). Bypasses the LeRobot EnvHub adapter
entirely (which has version-skew issues with current Arena).

Usage (after registering this module):
    /isaac-sim/python.sh isaaclab_arena/evaluation/policy_runner.py \
        --policy_type smolvla_lerobot \
        --num_steps 300 --num_envs 1 \
        gr1_open_microwave \
        --embodiment gr1_pink \
        --enable_cameras \
        --policy_checkpoint /lerobot-work/outputs/train/smolvla_microwave_lora/checkpoints/last/pretrained_model

The action chunking strategy matches what LeRobot's `select_action` does internally:
predict a chunk of `chunk_size` actions, execute them one-at-a-time, re-query
when the queue empties.
"""
from __future__ import annotations

import argparse
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase

logger = logging.getLogger(__name__)


@dataclass
class SmolVLALeRobotPolicyConfig:
    policy_checkpoint: str
    """Path (local dir or HF Hub repo_id) to a LeRobot-trained SmolVLA checkpoint
    (the directory containing `config.json` + `adapter_model.safetensors` or
    `model.safetensors`)."""

    device: str = "cuda:0"

    image_resolution: int = 512
    """Resolution to resize input images to before passing to SmolVLA. Should
    match the policy's expected input shape."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "SmolVLALeRobotPolicyConfig":
        return cls(
            policy_checkpoint=args.policy_checkpoint,
            device=getattr(args, "device", "cuda:0"),
            image_resolution=getattr(args, "smolvla_image_resolution", 512),
        )


@register_policy
class SmolVLALeRobotPolicy(PolicyBase):
    """Wrap a LeRobot SmolVLA checkpoint for use with Arena's policy_runner."""

    name = "smolvla_lerobot"
    config_class = SmolVLALeRobotPolicyConfig

    def __init__(self, config: SmolVLALeRobotPolicyConfig):
        super().__init__(config)
        self.config: SmolVLALeRobotPolicyConfig = config
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._task_description: str | None = None
        # Cache action chunks to avoid running the VLA every step.
        self._action_queues: list[deque] = []

    def _load(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        ckpt = self.config.policy_checkpoint
        logger.info(f"[smolvla] loading from {ckpt}")
        adapter_cfg_path = Path(ckpt) / "adapter_config.json"
        if adapter_cfg_path.exists():
            from peft import PeftConfig, PeftModel
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            # Load the policy config from OUR ckpt (has the correct max_state_dim/max_action_dim).
            policy_cfg = PreTrainedConfig.from_pretrained(ckpt)
            policy_cfg.device = self.config.device
            policy_cfg.pretrained_path = None  # we'll load weights manually below

            peft_cfg = PeftConfig.from_pretrained(ckpt)
            base_id = peft_cfg.base_model_name_or_path
            logger.info(f"[smolvla] PEFT adapter detected; building base with our config "
                        f"(max_state_dim={policy_cfg.max_state_dim}, "
                        f"max_action_dim={policy_cfg.max_action_dim}) from {base_id}")

            # Build the base policy with OUR (resized) config — fresh weights.
            base = SmolVLAPolicy(policy_cfg)
            # Load the base safetensors and copy keys whose shapes match.
            try:
                base_safetensor_path = hf_hub_download(repo_id=base_id, filename="model.safetensors")
            except Exception:
                base_safetensor_path = hf_hub_download(repo_id=base_id, filename="pytorch_model.bin")
            loaded = load_file(base_safetensor_path)
            own = base.state_dict()
            kept = {k: v for k, v in loaded.items() if k in own and own[k].shape == v.shape}
            skipped = [k for k in loaded if k in own and own[k].shape != v.shape] if False else \
                [k for k, v in loaded.items() if k in own and own[k].shape != v.shape]
            base.load_state_dict(kept, strict=False)
            logger.info(f"[smolvla] base loaded; kept {len(kept)}/{len(loaded)} keys; "
                        f"reinitialized layers: {skipped[:6]}{'...' if len(skipped) > 6 else ''}")

            # Now apply the LoRA adapter on top.
            self._policy = PeftModel.from_pretrained(base, ckpt, config=peft_cfg)
            self._policy.config = base.config
        else:
            self._policy = SmolVLAPolicy.from_pretrained(ckpt)
        self._policy.eval()
        # SmolVLA (500M) fits fp32 on 24 GB easily; don't force bf16 here. Forcing
        # bf16 caused fp32-vs-bf16 matmul mismatches on the base path because the
        # model has internal fp32 image-processing casts that conflict with bf16
        # parameter dtype.
        self._policy.to(self.config.device)
        self._chunk_size = getattr(self._policy.config, "n_action_steps", 50)
        # Detect the policy's expected primary image key from its input_features.
        cfg_input = getattr(self._policy.config, "input_features", None)
        self._image_key = "observation.images.camera1"  # fallback for our LoRA checkpoint
        if cfg_input:
            for k in cfg_input:
                if "images" in k:
                    self._image_key = k
                    break
        logger.info(f"[smolvla] ready; chunk_size={self._chunk_size}, "
                    f"max_state_dim={self._policy.config.max_state_dim}, "
                    f"max_action_dim={self._policy.config.max_action_dim}, "
                    f"image_key={self._image_key}")

        # Load saved preprocessor (handles tokenization + normalization)
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        self._preprocessor = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_preprocessor.json"
        )
        self._postprocessor = PolicyProcessorPipeline.from_pretrained(
            ckpt, config_filename="policy_postprocessor.json"
        )
        logger.info(f"[smolvla] preprocessor + postprocessor loaded")

    def set_task_description(self, task_description: str | None) -> str:
        self._task_description = task_description or "Reach out to the microwave and open it."
        # invalidate cached chunks on task change
        self._action_queues = []
        return super().set_task_description(self._task_description)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # invalidate any cached action chunks; next get_action triggers a fresh forward pass
        if env_ids is None:
            self._action_queues = []
        else:
            for i in env_ids.tolist():
                if i < len(self._action_queues):
                    self._action_queues[i].clear()

    def _build_lerobot_obs(self, env: gym.Env, observation) -> dict:
        """Map Arena obs → LeRobot expected input dict.

        Arena's gr1_open_microwave produces a Dict with at least a flat state vector
        plus camera images. The exact keys vary by env config; we sniff for them.
        """
        n_envs = env.unwrapped.num_envs if hasattr(env.unwrapped, "num_envs") else 1
        device = self.config.device

        # Arena nests the obs under sub-dicts ("policy", "camera_obs"); flatten both
        flat = {}
        for k, v in observation.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[kk] = vv
            else:
                flat[k] = v
        obs_dict = flat

        state_keys = ("robot_joint_pos", "joint_pos", "observation.state", "state")
        state = None
        for k in state_keys:
            if k in obs_dict:
                state = obs_dict[k]
                break
        if state is None:
            # fall back to first numeric tensor
            for v in obs_dict.values():
                if isinstance(v, torch.Tensor) and v.dtype != torch.uint8 and v.ndim <= 3:
                    state = v
                    break
        assert state is not None, f"Could not find state in obs keys {list(obs_dict.keys())}"

        # Don't pad — the saved preprocessor's normalizer expects the raw
        # 54-d (or whatever the training data shape was) and the policy
        # pads to max_state_dim internally via pad_vector().
        state = state.float().to(device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        # Defensive: if env state is bigger than policy max_state_dim, truncate.
        # This matters for the base-model path where smolvla_base's max_state_dim
        # is 32 but the GR1 env emits 54-dim state — pad_vector errors on truncate.
        max_state = getattr(self._policy.config, "max_state_dim", None)
        if max_state is not None and state.shape[-1] > max_state:
            state = state[..., :max_state]

        # Find a camera image (Arena uses keys like "robot_pov_cam" or "rgb_*")
        cam_keys = [k for k in obs_dict if "cam" in k.lower() or "rgb" in k.lower() or "image" in k.lower()]
        if not cam_keys:
            raise RuntimeError(f"No camera key found in obs {list(obs_dict.keys())}; "
                               "did you pass --enable_cameras?")
        img = obs_dict[cam_keys[0]]
        img = img.to(device)
        # normalize to [0,1] float CHW per-batch
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if img.ndim == 4 and img.shape[-1] in (1, 3):  # NHWC → NCHW
            img = img.permute(0, 3, 1, 2).contiguous()
        elif img.ndim == 3 and img.shape[-1] in (1, 3):  # HWC → 1CHW
            img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
        if img.ndim == 3:
            img = img.unsqueeze(0)
        # resize if needed
        target_res = self.config.image_resolution
        if img.shape[-1] != target_res or img.shape[-2] != target_res:
            img = F.interpolate(img, size=(target_res, target_res), mode="bilinear", align_corners=False)

        return {
            "observation.state": state,
            self._image_key: img,
            "task": [self._task_description or "Reach out to the microwave and open it."] * state.shape[0],
        }

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        if self._policy is None:
            self._load()

        action_space_shape = env.action_space.shape  # (n_envs, action_dim) or (action_dim,)
        n_envs = action_space_shape[0] if len(action_space_shape) > 1 else 1
        action_dim = action_space_shape[-1]

        # Initialize queues lazily
        if len(self._action_queues) != n_envs:
            self._action_queues = [deque() for _ in range(n_envs)]

        # If any env's queue is empty, refill via a forward pass
        if any(len(q) == 0 for q in self._action_queues):
            obs_input = self._build_lerobot_obs(env, observation)
            # Run through saved preprocessor (tokenizes "task", normalizes state, etc.)
            preprocessed = self._preprocessor(obs_input)
            # Defensive: base-model preprocessors may emit token tensors on CPU even
            # though the policy is on cuda. Move device only — don't touch dtype.
            # SmolVLA stays fp32 throughout, so no dtype reconciliation is needed.
            for k, v in list(preprocessed.items()):
                if hasattr(v, "to") and hasattr(v, "device"):
                    preprocessed[k] = v.to(self.config.device)
            with torch.inference_mode():
                # predict_action_chunk returns (n_envs, chunk_size, max_action_dim)
                chunk = self._policy.predict_action_chunk(preprocessed)
            # Match env's action_dim — truncate if policy outputs more, zero-pad if less.
            # Zero-pad path matters for base-model eval where the base was trained on a
            # smaller action space (smolvla_base outputs 6, GR1 env wants 36).
            if chunk.shape[-1] >= action_dim:
                chunk = chunk[..., :action_dim]
            else:
                chunk = F.pad(chunk, (0, action_dim - chunk.shape[-1]))
            for i in range(n_envs):
                if len(self._action_queues[i]) == 0:
                    for t in range(chunk.shape[1]):
                        self._action_queues[i].append(chunk[i, t])

        # Pop one action per env
        actions = torch.stack([q.popleft() for q in self._action_queues], dim=0)
        # Match the env's action_space shape
        if len(action_space_shape) == 1 and actions.ndim == 2:
            actions = actions.squeeze(0)
        return actions.to(self.config.device)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        g = parser.add_argument_group("SmolVLA LeRobot policy")
        g.add_argument(
            "--policy_checkpoint",
            type=str,
            required=True,
            help="Path or HF repo_id of the LeRobot SmolVLA checkpoint's pretrained_model dir.",
        )
        g.add_argument(
            "--smolvla_image_resolution",
            type=int,
            default=512,
            help="Resize input images to this resolution before passing to SmolVLA.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "SmolVLALeRobotPolicy":
        return SmolVLALeRobotPolicy(SmolVLALeRobotPolicyConfig.from_cli_args(args))
