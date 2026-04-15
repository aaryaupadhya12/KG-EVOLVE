# Reproducing CODEX-S Results with LibKGE

This section documents the configuration and methodology used to reproduce results on the **CODEX-S** dataset using **LibKGE**.  

The goal is to:
- Understand the YAML-based configuration pipeline  
- Reproduce benchmark metrics (MRR, Hits@K)  
- Analyze key architectural and training choices  

> Note: While additional datasets were explored using PyKEEN, the primary experiments focus on **closed-world knowledge graph settings**, making CODEX-S a more suitable benchmark.

---

## 1. Model Architecture

We use the **ComplEX** model wrapped inside a **Reciprocal Relations Model**.

### Reciprocal Learning
Instead of learning only: (France, capital, Paris) -> Model also learns -> (Paris, capital_inverse, France)


### Why this matters:
- Doubles the number of training triples  
- Improves bidirectional reasoning  
- Leads to **better Mean Reciprocal Rank (MRR)**  

---

## 📐 2. Embedding Configuration

- **Embedding Type:** Complex-valued vectors  
- **Dimension:** 512 (for both entities and relations)  
- **Initialization:** Xavier Normal  

### Insight:
Complex embeddings allow modeling **asymmetric relations**, which are common in knowledge graphs.

---

## ⚙️ 3. Regularization & Dropout (ComplEX-specific)

- **Entity Dropout:** ~0  
- **Relation Dropout:** ~0  
- **Regularization:** Minimal / near-zero  

These values are **not arbitrary** — they are derived from:
> Automated hyperparameter optimization using **Ax (Adaptive Experimentation Platform)**

### Key takeaway:
- Over-regularization hurts performance on CODEX-S  
- Best results come from **letting embeddings fully express structure**

---

## 4. Training Strategy: 1-vs-All

Instead of sampling negatives, we use:

### **1-vs-All Training**
For each triple `(s, p, o)`:
- The model scores `o` against **all entities (~2034)** simultaneously  

### Advantages:
- More stable gradients  
- Eliminates sampling bias  
- Faster convergence in closed-world settings  

---

## 5. Evaluation Metrics

We evaluate using standard link prediction metrics:

### Hits@K
- Hits@1, Hits@3, Hits@10, Hits@50  
- Measures how often the correct entity appears in top-K predictions  

### Mean Rank (MR) Variants
- **Rounded Mean Rank:**  
  Handles ties by assigning the **average rank**  

### Relation-Type Breakdown
Metrics are reported separately for:
- 1-to-1  
- 1-to-many  
- many-to-1  
- many-to-many  

### Why this matters:
Different relation types behave very differently — this gives deeper insight beyond aggregate scores.

---

##  6. Reproducibility & Random Seeds

By default: seed: -1

This means:
- Each run uses a different random seed  
- Results will vary slightly across runs  

### For reproducibility:
Set all seeds explicitly:


## Citation

```bibtex
@inproceedings{safavi-koutra-2020-codex,
    title     = "CoDEx: A Comprehensive Knowledge Graph Completion Benchmark",
    author    = "Safavi, Tara and Koutra, Danai",
    booktitle = "Proceedings of EMNLP 2020",
    year      = "2020",
    url       = "https://arxiv.org/pdf/2009.07810.pdf"
}

