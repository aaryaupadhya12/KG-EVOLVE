---

## System Requirements

| Component | Requirement |
|-----------|------------|
| OS | Windows 10/11 |
| Python | **3.7.0 exactly** |
| RAM | 8GB minimum |
| Disk | 2GB free |
| GPU (optional) | NVIDIA with CUDA 11.0 |

---

## Step 1 — Install Python 3.7.0

Download exactly this version:
👉 https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe

During install:
- Check **Add Python 3.7 to PATH**
- Check **Install for all users**

Verify:
```powershell
python --version
# Must show: Python 3.7.0
```

---

## Step 2 — Install Git

Download from https://git-scm.com/download/win and install with defaults.

---

## Step 3 — Clone This Repository

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

---

## Step 4 — Create Virtual Environment

```powershell
# Allow scripts to run in PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create venv with Python 3.7
python -m venv myenv

# Activate
.\myenv\Scripts\Activate.ps1
```

---

## Step 5 — Install Python Dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

---

## Step 6 — Install LibKGE (exact frozen commit)

```powershell
# Clone LibKGE at the exact commit the paper used
git clone https://github.com/uma-pi1/kge.git
cd kge
git checkout a9ecd249ec2d205df59287f64553a1536add4a43
pip install -e .
cd ..
```

---

## Step 7 — Fix Dependency Conflicts

Python 3.7 requires older versions of these packages:

```powershell
pip install sqlalchemy==1.3.23
pip install ax-platform==0.1.10
pip install urllib3==1.26.18
pip install requests==2.28.2
```

Verify no conflicts:
```powershell
pip check
# Should say: No broken requirements found
```

---

## Step 8 — Prepare CoDEx-S Dataset

```powershell
# Create data folder inside LibKGE
cd kge\data
New-Item -ItemType Directory -Name "codex-s" -Force

# Copy triple files
Copy-Item "..\..\data\triples\codex-s\train.txt" ".\codex-s\train.txt"
Copy-Item "..\..\data\triples\codex-s\valid.txt" ".\codex-s\valid.txt"
Copy-Item "..\..\data\triples\codex-s\test.txt"  ".\codex-s\test.txt"

# Preprocess into LibKGE format
python preprocess.py codex-s

# Go back to repo root
cd ..\..
```

Verify preprocessing worked:
```powershell
Get-ChildItem "kge\data\codex-s"
# Should show: train.del, valid.del, test.del, entity_ids.del, relation_ids.del, dataset.yaml
```

---

## Step 9 — Create Training Config

Run this in PowerShell from the repo root to create the config
without BOM encoding issues:

```powershell
New-Item -ItemType Directory -Force -Path "models\link-prediction\codex-s\complex"

$config = @"
import:
- reciprocal_relations_model
- complex

model: reciprocal_relations_model

dataset:
  name: codex-s
  num_entities: 2034
  num_relations: 42

reciprocal_relations_model:
  base_model:
    type: complex

lookup_embedder:
  dim: 512
  initialize: xavier_normal_
  initialize_args:
    xavier_normal_:
      gain: 1.0
    xavier_uniform_:
      gain: 1.0
    normal_:
      mean: 0.0
      std: 0.0003706650808601012
    uniform_:
      a: -0.2092997908254619
  regularize: ''
  regularize_args:
    weighted: true

complex:
  entity_embedder:
    dropout: 0.07931799348443747
    regularize_weight: 9.584175626202284e-13
  relation_embedder:
    dropout: 0.05643956921994686
    regularize_weight: 0.022858621200283015

train:
  type: 1vsAll
  auto_correct: true
  batch_size: 1024
  max_epochs: 400
  optimizer: Adam
  optimizer_args:
    lr: 0.00033858206813454155
  loss_arg: .nan
  lr_scheduler: ReduceLROnPlateau
  lr_scheduler_args:
    factor: 0.95
    mode: max
    patience: 7
    threshold: 0.0001

valid:
  early_stopping:
    patience: 10
    threshold:
      epochs: 50
      metric_value: 0.05

eval:
  batch_size: 256

entity_ranking:
  metrics_per:
    relation_type: true
"@

[System.IO.File]::WriteAllText(
  "$PWD\models\link-prediction\codex-s\complex\config.yaml",
  $config,
  [System.Text.UTF8Encoding]::new($false)
)

Write-Host "Config saved"
```

---

## Step 10 — Train ComplEx

### CPU (slower — ~20 hours full run)
```powershell
cd kge

python -m kge start `
  ..\models\link-prediction\codex-s\complex\config.yaml `
  --job.device cpu

cd ..
```

### GPU — CUDA 11.0 (faster — ~4-6 hours)

First install CUDA 11.0:
👉 https://developer.nvidia.com/cuda-11.0-download-archive

Then reinstall PyTorch with CUDA support:
```powershell
pip uninstall torch -y
pip install torch==1.7.1+cu110 `
  --find-links https://download.pytorch.org/whl/torch_stable.html
```

Then train on GPU:
```powershell
cd kge

python -m kge start `
  ..\models\link-prediction\codex-s\complex\config.yaml `
  --job.device cuda

cd ..
```

### What to Expect

| | CPU | GTX 1650 GPU |
|-|-----|-------------|
| Per epoch | ~3-5 min | ~30-60 sec |
| Validation every | 5 epochs | 5 epochs |
| Early stopping | ~200-345 epochs | same |
| Total time | ~15-20 hrs | ~4-6 hrs |

## Citation

```bibtex
@inproceedings{safavi-koutra-2020-codex,
    title     = "CoDEx: A Comprehensive Knowledge Graph Completion Benchmark",
    author    = "Safavi, Tara and Koutra, Danai",
    booktitle = "Proceedings of EMNLP 2020",
    year      = "2020",
    url       = "https://arxiv.org/pdf/2009.07810.pdf"
}