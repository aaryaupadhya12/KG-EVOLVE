import os
import json
import shutil
from pathlib import Path


HF_USERNAME   = "aaryaupadhya20"   # ← change this
REPO_NAME     = "codex-s-complex-winner"
REPO_ID       = f"{HF_USERNAME}/{REPO_NAME}"

WINNER_PT     = Path(r"C:\Users\aaryaupadhya\Documents\Aarya\NLP\codex\models\link-prediction\codex-s\complex\winner_model.pt")
METRICS_JSON  = Path(r"C:\Users\aaryaupadhya\Documents\Aarya\NLP\codex\models\link-prediction\codex-s\complex\winner_model_metrics.json")
KGE_PATH      = str(Path("kge").resolve())

# ── Sanity check files exist ──────────────────────────
print("Checking files...")
assert WINNER_PT.exists(),    f"winner_model.pt not found at {WINNER_PT}"
assert METRICS_JSON.exists(), f"metrics JSON not found at {METRICS_JSON}"
print(f"winner_model.pt  — {WINNER_PT.stat().st_size / 1e6:.1f} MB")
print(f"metrics JSON     — found")

# ── Print metrics ─────────────────────────────────────
with open(METRICS_JSON) as f:
    metrics = json.load(f)
print("\n── Metrics ──────────────────────────────────────")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# ── Create loader script ──────────────────────────────
loader_code = f'''import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def load_winner_model(kge_path: str = r"{KGE_PATH}", device: str = "cpu"):
    """
    Load the CoDEx-S ComplEx winner model from Hugging Face.

    Args:
        kge_path : absolute path to your local codex/kge directory
        device   : "cpu" or "cuda"

    Returns:
        winner_model : KgeModel ready for inference
    """
    sys.path.insert(0, kge_path)
    from kge.model import KgeModel
    from kge.util.io import load_checkpoint

    print("Downloading winner_model from Hugging Face...")
    path = hf_hub_download(
        repo_id="{REPO_ID}",
        filename="winner_model.pt"
    )

    print("Loading checkpoint...")
    checkpoint   = load_checkpoint(path, device=device)
    winner_model = KgeModel.create_from(checkpoint)
    winner_model.eval()
    print("winner_model loaded and ready!")
    return winner_model


if __name__ == "__main__":
    import torch

    model = load_winner_model()

    # Score a test triple (integer indices from CoDEx-S)
    s = torch.tensor([0])
    p = torch.tensor([1])
    o = torch.tensor([2])

    score = model.score_spo(s, p, o, direction="o")
    print(f"Test triple score: {{score.item():.4f}}")
'''

loader_path = Path(r"C:\Users\aaryaupadhya\Documents\Aarya\NLP\KGE_Evaluation\load_winner.py")
with open(loader_path, "w", encoding="utf-8") as f:
    f.write(loader_code)
print(f"\nLoader script saved → {loader_path}")

# ── Create model card ─────────────────────────────────
readme = f"""---
language: en
tags:
- knowledge-graph
- link-prediction
- ComplEx
- CoDEx
- LibKGE
- wikidata
datasets:
- CoDEx-S
metrics:
- mrr
- hits@1
- hits@10
---

# CoDEx-S ComplEx — Winner Model

Knowledge graph link prediction on **CoDEx-S** using **ComplEx** embeddings,
trained with the [LibKGE](https://github.com/uma-pi1/kge) framework.
Reproduces and slightly improves results from the
[CoDEx paper (EMNLP 2020)](https://arxiv.org/pdf/2009.07810.pdf).

## Results (Validation Set — Filtered with Test)

| Metric  | This Model | Paper |
|---------|-----------|-------|
| MRR     | {metrics.get("MRR", "—")} | 0.465 |
| Hits@1  | {metrics.get("Hits@1", "—")} | 0.372 |
| Hits@3  | {metrics.get("Hits@3", "—")} | 0.504 |
| Hits@10 | {metrics.get("Hits@10", "—")} | 0.646 |

Training stopped early at epoch **{metrics.get("epochs_trained", 345)}** via early stopping.

## Dataset — CoDEx-S

| | Count |
|-|-------|
| Entities | 2,034 |
| Relations | 42 |
| Train triples | 32,888 |
| Valid triples | 1,827 |
| Test triples | 1,828 |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 512 |
| Optimizer | Adam |
| Learning rate | 0.000339 |
| Batch size | 1024 |
| Max epochs | 400 |
| Training type | 1vsAll |
| Loss | KL divergence |
| LR scheduler | ReduceLROnPlateau |
| Entity dropout | 0.079 |
| Relation dropout | 0.056 |

## Load in Your App

```python
import sys
sys.path.insert(0, r"C:/path/to/codex/kge")

from huggingface_hub import hf_hub_download
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import torch

# Download from Hugging Face
path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="winner_model.pt"
)

# Load model
checkpoint   = load_checkpoint(path, device="cpu")
winner_model = KgeModel.create_from(checkpoint)
winner_model.eval()

print("winner_model ready!")

# Score a triple using entity/relation integer indices
s = torch.tensor([0])   # head entity index
p = torch.tensor([1])   # relation index
o = torch.tensor([2])   # tail entity index

score = winner_model.score_spo(s, p, o, direction="o")
print("Score:", score.item())
```

## Citation

```bibtex
@inproceedings{{safavi-koutra-2020-codex,
    title     = "CoDEx: A Comprehensive Knowledge Graph Completion Benchmark",
    author    = "Safavi, Tara and Koutra, Danai",
    booktitle = "Proceedings of EMNLP 2020",
    year      = "2020",
    url       = "https://arxiv.org/pdf/2009.07810.pdf"
}}
```
"""

readme_path = Path("README_hf.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)
print(f"Model card saved → {readme_path}")

# ── Push to Hugging Face ──────────────────────────────
print(f"\n Pushing to Hugging Face repo: {REPO_ID}")

from huggingface_hub import HfApi, create_repo

api = HfApi()

create_repo(REPO_ID, exist_ok=True, repo_type="model")
print(f"Repo ready → https://huggingface.co/{REPO_ID}")

uploads = [
    (str(WINNER_PT),   "winner_model.pt"),
    (str(METRICS_JSON),"winner_model_metrics.json"),
    (str(loader_path), "load_winner.py"),
    (str(readme_path), "README.md"),
]

for local_path, repo_filename in uploads:
    print(f"  Uploading {repo_filename}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_filename,
        repo_id=REPO_ID,
    )
    print(f" {repo_filename} uploaded")


print(f"View at: https://huggingface.co/{REPO_ID}")
print(f"\nLoad anytime with:")
print(f'  hf_hub_download(repo_id="{REPO_ID}", filename="winner_model.pt")')