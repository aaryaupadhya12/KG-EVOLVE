# ════════════════════════════════════════════════════════
# UNIFIED DATASET BUILDER + BERT DATASET
# Merges: convert_record pipeline (doc 5)
#       + format_candidate_structured (doc 4)
#       + KGRerankerDataset (doc 4)
# ════════════════════════════════════════════════════════

import json, numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict

# ── config ────────────────────────────────────────────────
MAX_K        = 10
NEG_PRIORITY = ["model", "structural", "similarity", "random"]
W_KGE        = 0.5
W_AGENT      = 0.3
W_OVERLAP    = 0.2
MAX_LENGTH   = 256    # increased from 128 — structured format is longer

FEATURE_KEYS = [
    "kge_score", "sim_to_head", "sim_to_true_tail",
    "shared_rels_count", "only_cand_count", "only_true_count",
    "jaccard_overlap", "direct_edge", "hop_count",
]

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")



def load_agent_confidence(path="val_hard_results.json"):
    try:
        with open(path) as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] {path} not found — defaulting conf=0.5")
        return {}

    lookup = {}
    for r in results:
        agg    = r.get("aggregator", {})
        chosen = agg.get("chosen_agent", "B")
        key_out = r.get("agent_a" if chosen == "A" else "agent_b", {})
        conf   = float(key_out.get("confidence", 0.5))

        t = r.get("triple", "")
        if t.startswith("(") and t.endswith(")"):
            parts = t[1:-1].split(", ")
            if len(parts) == 3:
                lookup[f"{parts[0]}|{parts[1]}|{parts[2]}"] = conf

    print(f"Agent confidence lookup: {len(lookup)} triples")
    return lookup


agent_conf_lookup = load_agent_confidence()


# ════════════════════════════════════════════════════════
# STEP 2 — FORMAT (replaces build_bert_text entirely)
# This is the structured format BERT actually trains on
# ════════════════════════════════════════════════════════

def format_candidate_structured(head:          str,
                                 relation:      str,
                                 candidate:     str,
                                 features:      dict,
                                 only_tail_has: list) -> str:
    """
    Replaces build_bert_text() from doc 5.
    
    Key difference: structured blocks instead of prose.
    Every field is labeled — no ambiguity for BERT.
    
    [QUERY]
    brazil | blockpositionindex | ?

    [CANDIDATE]
    china

    [KEY SIGNALS]
    kge_score = -5.327
    sim_to_head = 0.90
    shared_rels = 5
    only_true = 1
    only_cand = 0
    direct_edge = 1

    [DIFFERENCE]
    only_true_has = negativebehavior
    """
    f    = features
    diff = " | ".join(only_tail_has) if only_tail_has else "none"

    query_block     = f"[QUERY]\n{head} | {relation} | ?"
    candidate_block = f"[CANDIDATE]\n{candidate}"
    signals = (
        f"[KEY SIGNALS]\n"
        f"kge_score = {f.get('kge_score', 0.0):.3f}\n"
        f"sim_to_head = {f.get('sim_to_head', 0.0):.2f}\n"
        f"shared_rels = {f.get('shared_rels_count', 0)}\n"
        f"only_true = {f.get('only_true_count', 0)}\n"
        f"only_cand = {f.get('only_cand_count', 0)}\n"
        f"direct_edge = {f.get('direct_edge', 0)}"
    )
    diff_block = f"[DIFFERENCE]\nonly_true_has = {diff}"

    return "\n\n".join([query_block, candidate_block,
                        signals, diff_block])


# ════════════════════════════════════════════════════════
# STEP 3 — SOFT LABELS (unchanged from doc 5)
# ════════════════════════════════════════════════════════

def compute_soft_labels(candidates: list, agent_conf: float) -> list:
    neg_kge = [c["features"].get("kge_score", 0.0)
               for c in candidates if c["label"] == 0]
    neg_ov  = [c["features"].get("jaccard_overlap", 0.0)
               for c in candidates if c["label"] == 0]

    kge_min, kge_max = (min(neg_kge), max(neg_kge)) if neg_kge else (0.0, 1.0)
    ov_min,  ov_max  = (min(neg_ov),  max(neg_ov))  if neg_ov  else (0.0, 1.0)
    kge_r = max(kge_max - kge_min, 1e-6)
    ov_r  = max(ov_max  - ov_min,  1e-6)

    for c in candidates:
        if c["label"] == 1:
            c["soft_label"] = 1.0
        elif c.get("neg_type") == "random":
            c["soft_label"] = 0.0
        else:
            nk      = (c["features"].get("kge_score", 0.0)
                       - kge_min) / kge_r
            no      = (c["features"].get("jaccard_overlap", 0.0)
                       - ov_min) / ov_r
            penalty = (1.0 - agent_conf) * W_AGENT
            soft    = W_KGE * nk + penalty + W_OVERLAP * no
            c["soft_label"] = round(min(max(soft, 0.0), 0.89), 4)
    return candidates


# ════════════════════════════════════════════════════════
# STEP 4 — CANDIDATE SELECTION (unchanged from doc 5)
# ════════════════════════════════════════════════════════

def select_candidates(positive: dict,
                      negatives: list,
                      k: int = MAX_K) -> list:
    selected = [positive]
    seen     = {positive["entity"]}
    pool     = defaultdict(list)

    for neg in negatives:
        pool[neg.get("neg_type", "random")].append(neg)

    for neg_type in NEG_PRIORITY:
        for neg in pool[neg_type]:
            if neg["entity"] not in seen:
                selected.append(neg)
                seen.add(neg["entity"])
                if len(selected) == k:
                    return selected
    return selected


# ════════════════════════════════════════════════════════
# STEP 5 — CONVERT ONE RECORD
# Key change: text_input now uses format_candidate_structured
# instead of build_bert_text, and [DIFFERENCE] is recomputed
# per candidate (not shared from record level)
# ════════════════════════════════════════════════════════

def convert_record(record: dict) -> dict:
    head, relation, true_tail = record["triple"]
    agent_conf = agent_conf_lookup.get(
        f"{head}|{relation}|{true_tail}", 0.5
    )

    raw           = select_candidates(record["positive"],
                                      record["negatives"])
    only_tail_has = record.get("only_tail_has", [])

    for c in raw:
        if c["label"] == 1:
            # positive: full only_tail_has signal
            tail_has = only_tail_has
        else:
            # negative: recompute relative to THIS candidate
            # so [DIFFERENCE] reflects what THIS candidate lacks
            cand_shared = set(c.get("shared_relations", []))
            tail_has = [
                r for r in only_tail_has
                if r not in cand_shared
            ]

        # structured format replaces build_bert_text
        c["text_input"] = format_candidate_structured(
            head, relation, c["entity"],
            features      = c.get("features", {}),
            only_tail_has = tail_has,
        )

    candidates = compute_soft_labels(raw, agent_conf)
    candidates.sort(key=lambda c: -c["soft_label"])

    return {
        "triple":           [head, relation, true_tail],
        "true_rank":        record["true_rank"],
        "hop_type":         record.get("hop_type", "multi"),
        "agent_confidence": agent_conf,
        "candidates": [
            {
                "entity":     c["entity"],
                "label":      c["label"],
                "soft_label": c["soft_label"],
                "neg_type":   c.get("neg_type", "positive"),
                "text_input": c["text_input"],
                "features":   c.get("features", {}),
            }
            for c in candidates
        ],
        "context": {
            "subgraph_str":  record.get("subgraph_str", ""),
            "only_tail_has": only_tail_has,
            "fail_summary":  record.get("fail_summary", ""),
        }
    }


# ════════════════════════════════════════════════════════
# STEP 6 — BUILD AND SAVE bert_reranker_train.json
# ════════════════════════════════════════════════════════

def build_reranker_json(input_path:  str = "bert_training_data.json",
                        output_path: str = "bert_reranker_train.json"):

    with open(input_path) as f:
        bert_data = json.load(f)

    print(f"Input records: {len(bert_data)}")

    reranker_data, errors = [], 0
    for i, record in enumerate(bert_data):
        try:
            reranker_data.append(convert_record(record))
        except Exception as e:
            print(f"[ERROR] {i}: {e}")
            errors += 1

    with open(output_path, "w") as f:
        json.dump(reranker_data, f, indent=2)

    # ── stats ──────────────────────────────────────────────
    all_soft = [
        c["soft_label"]
        for r in reranker_data
        for c in r["candidates"]
    ]
    print(f"\nOutput:  {len(reranker_data)} records  ({errors} errors)")
    print(f"Avg candidates per triple: "
          f"{sum(len(r['candidates']) for r in reranker_data) / max(len(reranker_data),1):.1f}")
    print(f"Soft label — min={min(all_soft):.3f}  "
          f"max={max(all_soft):.3f}  "
          f"mean={np.mean(all_soft):.3f}")
    print(f"\nSample text_input (positive):")
    print(reranker_data[0]["candidates"][0]["text_input"])
    print(f"\nSample text_input (first negative):")
    print(reranker_data[0]["candidates"][1]["text_input"])

    return reranker_data



class KGRerankerDataset(Dataset):
    """
    Reads from bert_reranker_train.json.
    Each __getitem__ = one (positive, negative) pair.

    text_input is already structured — just tokenize it.
    soft_label is used as pair weight during loss computation.
    """

    def __init__(self, data: list, max_length: int = MAX_LENGTH):
        self.pairs      = []
        self.max_length = max_length

        for ex in data:
            # find the positive candidate
            pos = next(
                (c for c in ex["candidates"] if c["label"] == 1),
                None
            )
            if pos is None:
                continue

            # pair positive against every negative
            for neg in ex["candidates"]:
                if neg["label"] == 1:
                    continue

                self.pairs.append({
                    "pos_text":   pos["text_input"],
                    "neg_text":   neg["text_input"],
                    "pos_feats":  self._get_features(pos),
                    "neg_feats":  self._get_features(neg),
                    "neg_type":   neg["neg_type"],
                    "soft_label": neg["soft_label"],  # pair weight
                    "true_rank":  ex["true_rank"],
                })

        print(f"Dataset: {len(self.pairs)} pairs  "
              f"({len(data)} triples)")

    def _get_features(self, candidate: dict) -> torch.Tensor:
        f = candidate.get("features", {})
        return torch.tensor(
            [float(f.get(k, 0.0)) for k in FEATURE_KEYS],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        pos_enc = tokenizer(
            pair["pos_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        neg_enc = tokenizer(
            pair["neg_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pos_input_ids":      pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids":      neg_enc["input_ids"].squeeze(0),
            "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
            "pos_features":       pair["pos_feats"],
            "neg_features":       pair["neg_feats"],
            "neg_type":           pair["neg_type"],
            "soft_label":         torch.tensor(pair["soft_label"],
                                               dtype=torch.float32),
        }


# ════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════

reranker_data = build_reranker_json(
    input_path  = r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\Failure_Aware_KG_reasoning\Nations\Data\bert_training_data.json",
    output_path = "bert_reranker_train.json",
)

dataset = KGRerankerDataset(reranker_data)

pair = dataset[0]
print("\nPos tokens decoded:")
print(tokenizer.decode(pair["pos_input_ids"], skip_special_tokens=True)[:300])
print("\nNeg tokens decoded:")
print(tokenizer.decode(pair["neg_input_ids"], skip_special_tokens=True)[:300])
print("\nSoft label:", pair["soft_label"].item())
print("Neg type:",   pair["neg_type"])