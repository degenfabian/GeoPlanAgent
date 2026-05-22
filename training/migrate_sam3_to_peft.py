"""ONE-TIME MIGRATION: convert per-fold ``best.pt`` (~3.34 GB each, full
state-dict + optimizer + scheduler + history) into the PEFT publication
format (``adapter_config.json`` + ``adapter_model.safetensors``, ~74 MB
each) for shipping the trained weights.

This script is **additive**: it writes the PEFT files alongside the
existing ``best.pt`` files without touching or deleting them. You can
verify the resulting PEFT files load equivalently (re-run
``training/eval/eval_sam_kfold.py`` and compare ``cv_summary.json``)
and only then delete ``best.pt`` with::

    rm models/sam3_lora/fold_*/best.pt   # reclaim ~17 GB

After verification, **delete this script** — it's a one-shot tool and
the trainer will produce PEFT format directly from now on.

Side-effect summary per fold:
  +  adapter_config.json         (~1 KB)
  +  adapter_model.safetensors   (~74 MB)
  +  training_meta.json          (~1 KB; preserves epoch / best_val_iou /
                                  config from the original best.pt)
  (best.pt and history.json are not touched)

Usage::

    uv run python training/migrate_sam3_to_peft.py
    uv run python training/migrate_sam3_to_peft.py --folds 0,1      (subset)
    uv run python training/migrate_sam3_to_peft.py --dry-run        (no writes)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
REPO = THIS.parent
sys.path.insert(0, str(REPO))

import torch
from peft import LoraConfig, get_peft_model
from transformers import Sam3Model

# Import the trainer's EXACT constants so a config drift between the
# trainer and this migration script is impossible — both pull from the
# same source of truth.
from training.train_sam3_kfold import (
    HEAD_MODULES, LORA_TARGET_MODULES, MODEL_ID,
)


MODELS_DIR = REPO / "models" / "sam3_lora"


def convert_fold(fold_idx: int, hf_token: str | None,
                 dry_run: bool = False) -> bool:
    fold_dir = MODELS_DIR / f"fold_{fold_idx}"
    src = fold_dir / "best.pt"
    if not src.exists():
        print(f"  fold {fold_idx}: missing {src}, skipping")
        return False
    dst_cfg = fold_dir / "adapter_config.json"
    dst_safe = fold_dir / "adapter_model.safetensors"
    dst_meta = fold_dir / "training_meta.json"
    if dst_cfg.exists() and dst_safe.exists() and dst_meta.exists():
        print(f"  fold {fold_idx}: PEFT files already present, skipping")
        return False

    print(f"  fold {fold_idx}: loading {src.name} ({src.stat().st_size / 1e9:.2f} GB) ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    cfg = ckpt.get("config", {})
    rank = cfg.get("rank", 16)

    # Fresh base + fresh PEFT wrap PER FOLD. We can't re-wrap the same
    # base across folds: get_peft_model on an already-PeftModel attaches
    # a second adapter (PEFT logs a "modify ... for a second time" warning)
    # and the loaded weights become ambiguous between adapter copies.
    # Slower (~3 GB load × 5) but the only correctness-safe pattern.
    print(f"  fold {fold_idx}: building fresh base + PEFT wrapper "
          f"(rank={rank}, target_modules={len(LORA_TARGET_MODULES)} prefixes, "
          f"modules_to_save={HEAD_MODULES})")
    base = Sam3Model.from_pretrained(MODEL_ID, token=hf_token)
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=rank * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05, bias="none",
        modules_to_save=HEAD_MODULES,
    )
    model = get_peft_model(base, lora_cfg)

    # strict=False: best.pt holds ALL of the base model's frozen weights
    # too (under their original key paths), which the freshly-built PEFT
    # wrapper already filled via from_pretrained. A few "missing" keys
    # there are expected and benign.
    #
    # `unexpected` keys, on the other hand, would mean the trainer's
    # config drifted from this script's — fail loud so we don't ship a
    # silently-mismatched adapter.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  fold {fold_idx}: ✗ ABORT — {len(unexpected)} UNEXPECTED keys "
              f"(trainer/migration config drift). First five: {unexpected[:5]}")
        return False
    # Sanity check on what's NOT missing: every key we care about (LoRA
    # matrices + saved heads) must have been loaded. Anything else
    # missing is the frozen base, which is fine.
    really_missing = [
        k for k in missing
        if "lora_" in k or any(h in k for h in HEAD_MODULES)
    ]
    if really_missing:
        print(f"  fold {fold_idx}: ✗ ABORT — {len(really_missing)} trainable "
              f"keys were NOT in best.pt. First five: {really_missing[:5]}")
        return False
    print(f"  fold {fold_idx}: load_state_dict OK ({len(missing)} frozen-base "
          f"keys missing, 0 unexpected, 0 trainable missing)")

    if dry_run:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  fold {fold_idx}: [DRY-RUN] would write {dst_cfg.name} + "
              f"{dst_safe.name} + {dst_meta.name} "
              f"({n_trainable / 1e6:.1f}M trainable params, "
              f"~{n_trainable * 4 / 1e6:.0f} MB fp32 .safetensors)")
        return True

    print(f"  fold {fold_idx}: saving PEFT adapter to {fold_dir} ...")
    model.save_pretrained(str(fold_dir))

    # Verify the files we expected were written.
    if not (dst_cfg.exists() and dst_safe.exists()):
        print(f"  fold {fold_idx}: ✗ save_pretrained didn't write the "
              f"expected files. adapter_config: {dst_cfg.exists()}, "
              f"adapter_model: {dst_safe.exists()}")
        return False

    # Preserve the bits of metadata best.pt was carrying that PEFT
    # save_pretrained drops (epoch, val IoU, training config). The eval
    # script + paper reporting still want these.
    meta = {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "best_val_iou": ckpt.get("best_val_iou"),
        "best_metric": ckpt.get("best_metric"),
        "epochs_since_best": ckpt.get("epochs_since_best"),
        "fold": ckpt.get("fold", fold_idx),
        "config": ckpt.get("config", {}),
        "_source": "migrated from best.pt by training/migrate_sam3_to_peft.py",
    }
    dst_meta.write_text(json.dumps(meta, indent=2))

    size_mb = (dst_cfg.stat().st_size + dst_safe.stat().st_size
               + dst_meta.stat().st_size) / 1e6
    src_gb = src.stat().st_size / 1e9
    print(f"  fold {fold_idx}: ✓ wrote PEFT bundle ({size_mb:.1f} MB; "
          f"vs {src_gb:.2f} GB best.pt — {src_gb * 1024 / size_mb:.0f}× smaller)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folds", default="0,1,2,3,4",
                    help="Comma-separated fold indices to convert.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build the PEFT model and report sizes WITHOUT writing.")
    args = ap.parse_args()

    folds = [int(s) for s in args.folds.split(",") if s.strip()]
    if not MODELS_DIR.exists():
        print(f"ERROR: {MODELS_DIR} not found.", file=sys.stderr)
        return 1

    print(f"{'Dry-run: ' if args.dry_run else ''}"
          f"Converting folds {folds} in {MODELS_DIR}")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")

    ok = sum(convert_fold(k, hf_token, dry_run=args.dry_run) for k in folds)
    print(f"\nDone. {ok}/{len(folds)} folds converted.")
    if not args.dry_run and ok > 0:
        print("\nNext steps:")
        print("  1. Verify equivalence: re-run training/eval/eval_sam_kfold.py "
              "and confirm cv_summary.json per-fold sem_iou values still match.")
        print("  2. Once verified, reclaim disk by removing the old "
              "full-state checkpoints:")
        print(f"       rm models/sam3_lora/fold_*/best.pt   "
              f"# frees ~{3.34 * ok:.1f} GB")
        print("  3. Delete this migration script:")
        print("       rm training/migrate_sam3_to_peft.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
