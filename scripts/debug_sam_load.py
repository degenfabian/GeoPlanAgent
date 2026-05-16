"""Diagnose whether my k-fold eval is actually loading the LoRA correctly."""
import os, sys, json
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import Sam3Model, Sam3Processor

REPO = Path("/Users/fabiandegen/Documents/VSCODE/GeoMapAgent_autonomous")
sys.path.insert(0, str(REPO))

MODEL_ID = "facebook/sam3"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
HEAD_MODULES = ["mask_embedder", "presence_head", "semantic_projection"]

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
print("loading base SAM3 model…")
base = Sam3Model.from_pretrained(MODEL_ID, token=hf_token)
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGET_MODULES,
                lora_dropout=0.05, bias="none",
                modules_to_save=HEAD_MODULES)
model = get_peft_model(base, cfg)
print(f"wrapped model state_dict size: {len(model.state_dict())}")

# Load fold 0 best.pt
ckpt = torch.load(REPO / "models/sam3_lora/fold_0/best.pt",
                    map_location="cpu", weights_only=False)
state = ckpt["state_dict"]
print(f"checkpoint state_dict size: {len(state)}")

# strict=True load — if anything mismatches we see it
try:
    res = model.load_state_dict(state, strict=True)
    print("strict=True load: SUCCESS")
except Exception as e:
    msg = str(e)
    # First 1500 chars
    print(f"strict=True load FAILED:\n{msg[:1500]}")

# Now compare keys
model_keys = set(model.state_dict().keys())
ckpt_keys = set(state.keys())
missing_in_ckpt = model_keys - ckpt_keys  # model expects but ckpt doesn't have
unexpected_in_ckpt = ckpt_keys - model_keys  # ckpt has but model doesn't expect

print(f"\nKey overlap: {len(model_keys & ckpt_keys)}")
print(f"In model but NOT in ckpt: {len(missing_in_ckpt)}")
print(f"In ckpt but NOT in model: {len(unexpected_in_ckpt)}")
print(f"\nFirst 5 in model not in ckpt:")
for k in list(missing_in_ckpt)[:5]:
    print(" ", k)
print(f"\nFirst 5 in ckpt not in model:")
for k in list(unexpected_in_ckpt)[:5]:
    print(" ", k)

# Check what % of lora_A keys actually loaded
ckpt_lora_a = [k for k in ckpt_keys if 'lora_A' in k]
ckpt_mts = [k for k in ckpt_keys if 'modules_to_save' in k]
print(f"\nckpt has {len(ckpt_lora_a)} lora_A and {len(ckpt_mts)} modules_to_save keys")
match_a = sum(1 for k in ckpt_lora_a if k in model_keys)
match_mts = sum(1 for k in ckpt_mts if k in model_keys)
print(f"of those: {match_a} lora_A match, {match_mts} modules_to_save match")
