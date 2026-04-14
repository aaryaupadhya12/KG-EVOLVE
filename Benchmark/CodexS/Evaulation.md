---
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
| MRR     | 0.474 | 0.465 |
| Hits@1  | 0.377 | 0.372 |
| Hits@3  | 0.522 | 0.504 |
| Hits@10 | 0.664 | 0.646 |

Training stopped early at epoch **345** via early stopping.

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
    repo_id="aaryaupadhya20/codex-s-complex-winner",
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
@inproceedings{safavi-koutra-2020-codex,
    title     = "CoDEx: A Comprehensive Knowledge Graph Completion Benchmark",
    author    = "Safavi, Tara and Koutra, Danai",
    booktitle = "Proceedings of EMNLP 2020",
    year      = "2020",
    url       = "https://arxiv.org/pdf/2009.07810.pdf"
}
```

# Reproduced By AaryaUpadhya (aarya.upadhya@gmail.com)