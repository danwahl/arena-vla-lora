"""Wrapper around lerobot-train that injects lora_alpha into the policy's PEFT defaults.

LeRobot's PeftConfig dataclass exposes --peft.r but not --peft.lora_alpha (as of v0.5.1).
Upstream PEFT defaults lora_alpha=8, so --peft.r=64 silently lands on scale=alpha/r=0.125,
which is far too weak. This wrapper monkey-patches the policy's `_get_default_peft_targets()`
to inject lora_alpha (and optionally lora_dropout) before the train script runs.

Use exactly like `lerobot-train`, with two extra environment variables:
    LORA_ALPHA       — int (required for the patch to engage)
    LORA_DROPOUT     — float (optional, default unchanged)

Example:
    LORA_ALPHA=64 python scripts/lerobot_train_with_alpha.py \
        --policy.path=lerobot/smolvla_base \
        --dataset.repo_id=unitreerobotics/G1_Dex3_Pouring_Dataset \
        --steps=30000 --batch_size=12 \
        --peft.method_type=LORA --peft.r=32 \
        ...

The patch covers SmolVLA, π₀.₅, and π₀ — the three pretrained-VLA classes that ship
`_get_default_peft_targets`.
"""

import os
import sys


def _engage_alpha_patch():
    alpha = os.environ.get("LORA_ALPHA")
    if alpha is None:
        print("[wrap] LORA_ALPHA not set — running unpatched.", file=sys.stderr)
        return
    alpha = int(alpha)
    dropout = os.environ.get("LORA_DROPOUT")
    dropout = float(dropout) if dropout is not None else None

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def _patch(cls):
        orig = cls._get_default_peft_targets

        def patched(self):
            cfg = dict(orig(self) or {})
            cfg["lora_alpha"] = alpha
            if dropout is not None:
                cfg["lora_dropout"] = dropout
            return cfg

        cls._get_default_peft_targets = patched
        print(
            f"[wrap] patched {cls.__name__}._get_default_peft_targets — "
            f"lora_alpha={alpha}" + (f", lora_dropout={dropout}" if dropout is not None else ""),
            file=sys.stderr,
        )

    for cls in (SmolVLAPolicy, PI05Policy, PI0Policy):
        _patch(cls)


def _engage_shape_safe_load_patch():
    """Skip state_dict keys whose shapes don't match the current model.

    Required when bumping max_state_dim / max_action_dim above the pretrained checkpoint's
    values: the state_proj / action_in_proj / action_out_proj layers have new shapes,
    must be reinitialized fresh, but the rest of the checkpoint is fine to load.
    """
    if os.environ.get("SHAPE_SAFE_LOAD") != "1":
        return

    from lerobot.policies import pretrained as _pre

    _ = _pre.PreTrainedPolicy._load_as_safetensor

    def patched(cls_or_self, model, model_file, map_location, strict):
        from safetensors.torch import load_file

        loaded = load_file(model_file, device=map_location)
        own = model.state_dict()
        skipped = []
        kept = {}
        for k, v in loaded.items():
            if k in own and own[k].shape == v.shape:
                kept[k] = v
            elif k in own:
                skipped.append((k, tuple(v.shape), tuple(own[k].shape)))
            else:
                kept[k] = v  # not in model — load_state_dict(strict=False) handles
        if skipped:
            print(
                f"[wrap] shape-safe load: skipped {len(skipped)} mismatched keys:",
                file=sys.stderr,
            )
            for k, src, dst in skipped[:8]:
                print(f"  - {k}: ckpt {src} vs model {dst}", file=sys.stderr)
        model.load_state_dict(kept, strict=False)
        return model

    _pre.PreTrainedPolicy._load_as_safetensor = classmethod(patched)
    print(
        "[wrap] enabled SHAPE_SAFE_LOAD: skipping state_dict keys with shape mismatches",
        file=sys.stderr,
    )


def main():
    _engage_alpha_patch()
    _engage_shape_safe_load_patch()
    from lerobot.scripts.lerobot_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
