"""Repository paths and external assets (CausalPFN checkpoint, IHDP data).

Checkpoint resolution order:
  1. explicit path argument
  2. $CAUSALPFN_CKPT
  3. <repo>/checkpoints/causalpfn_v0.pt
  4. download from the Hugging Face hub (vdblm/causalpfn) via huggingface_hub
IHDP csv: $IHDP_CSV, else <repo>/data/IHDP/csv/ihdp_npci_1.csv (see scripts/get_ihdp.sh).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_NAME = "causalpfn_v0.pt"
HF_REPO = "vdblm/causalpfn"


def get_checkpoint(path: str | None = None) -> str:
    for cand in (path, os.environ.get("CAUSALPFN_CKPT"), REPO_ROOT / "checkpoints" / CKPT_NAME):
        if cand and Path(cand).exists():
            return str(cand)
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=HF_REPO, filename=CKPT_NAME)
    except Exception as e:  # pragma: no cover
        raise FileNotFoundError(
            f"CausalPFN checkpoint not found. Download {CKPT_NAME} from https://huggingface.co/{HF_REPO} "
            f"into {REPO_ROOT / 'checkpoints'} or set CAUSALPFN_CKPT. ({e})")


def ihdp_csv() -> str:
    return os.environ.get("IHDP_CSV", str(REPO_ROOT / "data" / "IHDP" / "csv" / "ihdp_npci_1.csv"))
