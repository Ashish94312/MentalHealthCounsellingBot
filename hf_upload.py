#!/usr/bin/env python3
"""
High-level uploader for pushing model artifacts to the Hugging Face Hub.

Features:
- Creates the repo if it doesn't exist (model repo).
- Uploads a local folder (e.g., a checkpoint directory).
- Optionally deletes training-only artifacts (optimizer/scheduler/rng files).
- Optionally writes a README with MPS-only usage instructions.

Environment:
- Uses HF token from env var HF_TOKEN by default; can also be passed via --token.

Example:
  python hf_upload.py \
    --repo-id sudojarvis/tinylama-mental-health-mentalchat16k-ckpt-3620 \
    --local-path /Users/ashishkumar/MentalHealthCounsellingBot/tinylama-mental-health-mentalchat16k/checkpoint-3620 \
    --private \
    --delete-training-artifacts \
    --write-readme
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    CommitOperationAdd,
    CommitOperationDelete,
)


TRAINING_ARTIFACTS = [
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
]


def build_readme(repo_id: str, base_model: str) -> str:
    """Return a README.md content with a compact model card and MPS-only usage."""
    return f"""---
tags:
  - lora
  - tinylama
  - mental-health
  - counseling
license: apache-2.0
library_name: transformers
inference: false
pipeline_tag: text-generation
---

# TinyLlama Mental Health (LoRA Adapter) — 16k checkpoint

This repo hosts a LoRA adapter checkpoint for TinyLlama fine-tuned on mental health counseling style data.

- Base: `{base_model}`
- Domain: Mental health counseling
- Method: PEFT LoRA fine-tuning
- Context length: 16k

## Usage (Apple Silicon, MPS only)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = '{base_model}'
adapter = '{repo_id}'

device = torch.device('mps')  # MPS-only usage

tok = AutoTokenizer.from_pretrained(base)

model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.to(device)

model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.float16)
model.to(device)

prompt = "I feel overwhelmed at work."
inputs = tok(prompt, return_tensors='pt').to(device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Files
- `adapter_model.safetensors`: LoRA weights
- `adapter_config.json`: PEFT configuration
- `tokenizer.*`, `special_tokens_map.json`, `chat_template.jinja`

## License
Apache-2.0
"""


@dataclass
class UploadConfig:
    repo_id: str
    local_path: str
    token: Optional[str]
    private: bool
    base_model: str
    exclude: List[str]
    delete_training_artifacts: bool
    write_readme: bool


def parse_args(argv: List[str]) -> UploadConfig:
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub (model repo)")
    parser.add_argument("--repo-id", required=True, help="Target repo id, e.g. username/repo-name")
    parser.add_argument(
        "--local-path",
        required=True,
        help="Local folder to upload (e.g., checkpoint directory)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (defaults to env HF_TOKEN)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist",
    )
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model id for README usage snippet",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["**/optimizer.pt", "**/scheduler.pt", "**/rng_state.pth"],
        help="Glob patterns to exclude from upload",
    )
    parser.add_argument(
        "--delete-training-artifacts",
        action="store_true",
        help="Delete optimizer/scheduler/rng files from the repo after upload",
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Write a README.md with MPS-only usage instructions",
    )

    args = parser.parse_args(argv)

    if not args.token:
        raise SystemExit("HF token not provided. Set --token or HF_TOKEN env var.")
    if not os.path.isdir(args.local_path):
        raise SystemExit(f"Local path does not exist or is not a directory: {args.local_path}")

    return UploadConfig(
        repo_id=args.repo_id,
        local_path=args.local_path,
        token=args.token,
        private=bool(args.private),
        base_model=args.base_model,
        exclude=list(args.exclude or []),
        delete_training_artifacts=bool(args.delete_training_artifacts),
        write_readme=bool(args.write_readme),
    )


def ensure_repo_exists(api: HfApi, repo_id: str, private: bool) -> None:
    # create_repo is idempotent with exist_ok=True
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=api.token)


def upload_directory(api: HfApi, cfg: UploadConfig) -> None:
    upload_folder(
        repo_id=cfg.repo_id,
        repo_type="model",
        folder_path=cfg.local_path,
        path_in_repo=".",
        ignore_patterns=cfg.exclude,
        token=api.token,
        commit_message="Upload model folder",
    )


def delete_artifacts(api: HfApi, repo_id: str) -> None:
    ops = [CommitOperationDelete(path_in_repo=p) for p in TRAINING_ARTIFACTS]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=ops,
        commit_message="Remove training artifacts (optimizer, scheduler, rng state)",
        token=api.token,
    )


def write_readme(api: HfApi, repo_id: str, base_model: str) -> None:
    content = build_readme(repo_id=repo_id, base_model=base_model)
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=content.encode("utf-8"))]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=ops,
        commit_message="Add model card (README.md)",
        token=api.token,
    )


def main(argv: Optional[List[str]] = None) -> None:
    cfg = parse_args(argv if argv is not None else sys.argv[1:])
    api = HfApi(token=cfg.token)

    ensure_repo_exists(api, cfg.repo_id, cfg.private)
    upload_directory(api, cfg)
    if cfg.delete_training_artifacts:
        delete_artifacts(api, cfg.repo_id)
    if cfg.write_readme:
        write_readme(api, cfg.repo_id, cfg.base_model)

    print(f"Uploaded to https://huggingface.co/{cfg.repo_id}")


if __name__ == "__main__":
    main()


