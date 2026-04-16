"""
DATA LOADER MODULE
==================

Handles all CODEX-S dataset loading, entity/relation ID mapping, graph construction,
and preprocessing of triples with model scores.

Key Responsibilities:
  1. Load CODEX-S dataset (train/test/valid splits)
  2. Resolve entity/relation IDs to English labels
  3. Build entity_to_id and relation_to_id mappings
  4. Construct knowledge graph from dataframe
  5. Preprocess all triples with model scores (done once, cached to JSON)
  6. Build type constraints from training data

Design:
  - Preprocessing is SLOW (must score every triple with model)
  - Results cached to JSON immediately
  - Subsequent runs load JSON instantly (no model calls needed)
  - This enables quick iteration on Kaggle or GPU-limited environments
"""

import json
import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F

from codex.codex import Codex


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD CODEX-S DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_codex_data(size="s", code="en"):
    """
    Load CODEX dataset and resolve all entity/relation IDs to English labels.
    
    Args:
        size: "s" for small, "m" for medium, "l" for large
        code: Language code, "en" for English labels
    
    Returns:
        Tuple of (df_train, df_test, df_valid, df_all, entity_to_id, relation_to_id, 
                  id_to_entity, id_to_relation)
    
    Data Flow:
        1. Load raw CODEX using Codex(size, code)
        2. For each triple (head_raw, relation_raw, tail_raw):
           - Resolve to English label using entity_label() and relation_label()
           - Build list of (label, label, label) tuples
        3. Convert to pandas DataFrames for easy manipulation
        4. Build ID mappings from CODEX entity/relation lists
        5. Return all mappings for use throughout pipeline
    
    Note:
        - ID mappings MUST match exactly what the pretrained model was trained with
        - score_hrt expects integer IDs built from these sorted lists
        - If ID order changes, model scores become meaningless
    """
    codex = Codex(size=size, code=code)

    def _resolve(row):
        """Helper: resolve raw IDs to English labels for one triple."""
        return (
            codex.entity_label(row["head"]) or row["head"],
            codex.relation_label(row["relation"]) or row["relation"],
            codex.entity_label(row["tail"]) or row["tail"],
        )

    # Load splits and resolve to English labels
    df_train = pd.DataFrame(
        [_resolve(r) for _, r in codex.split("train").iterrows()],
        columns=["head", "relation", "tail"]
    )
    df_test = pd.DataFrame(
        [_resolve(r) for _, r in codex.split("test").iterrows()],
        columns=["head", "relation", "tail"]
    )
    df_valid = pd.DataFrame(
        [_resolve(r) for _, r in codex.split("valid").iterrows()],
        columns=["head", "relation", "tail"]
    )
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    print(f"[Loader] Train: {len(df_train)}  Valid: {len(df_valid)}  Test: {len(df_test)}")

    # Build ID mappings from sorted entity/relation lists
    # CRITICAL: Order must match pretrained model!
    entities_raw = sorted(codex.entities())
    relations_raw = sorted(codex.relations())

    entity_to_id = {e: i for i, e in enumerate(entities_raw)}
    relation_to_id = {r: i for i, r in enumerate(relations_raw)}
    id_to_entity = {i: e for e, i in entity_to_id.items()}
    id_to_relation = {i: r for r, i in relation_to_id.items()}

    print(f"[Loader] Entities: {len(entity_to_id)}   Relations: {len(relation_to_id)}")

    return (
        df_train, df_test, df_valid, df_all,
        entity_to_id, relation_to_id,
        id_to_entity, id_to_relation
    )


def build_graph_from_df(df):
    """
    Build adjacency list representation of knowledge graph.
    
    Args:
        df: DataFrame with columns ["head", "relation", "tail"]
    
    Returns:
        dict: graph[entity] = [(relation, tail_entity), ...]
        
    Example:
        graph["Alice"] = [("hasChild", "Bob"), ("livesIn", "NewYork")]
        
    Used for:
        - Subgraph extraction (BFS from entity)
        - Hop classification (count hops between head and tail)
        - Type constraint building (aggregate triple patterns)
    """
    graph = defaultdict(list)
    for _, row in df.iterrows():
        graph[row["head"]].append((row["relation"], row["tail"]))
    return graph


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TYPE CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────

def build_type_constraints(df):
    """
    Build statistical type constraints from training data.
    
    Analyzes which entities appear as tails/heads of each relation.
    Creates a learned prior over valid argument types.
    
    Args:
        df: Training dataframe with ["head", "relation", "tail"]
    
    Returns:
        dict with keys:
            - "rel_head_to_ranked_tails": (relation, head) -> {tail: probability}
              For each (relation, specific_head), what are the likely tail entities?
              Example: ("hasChild", "Alice") -> {"Bob": 0.5, "Carol": 0.5}
              
            - "rel_to_tail_dist": relation -> {tail: probability}
              For each relation, what's the general distribution of tails?
              Example: "hasChild" -> {"Bob": 0.3, "Carol": 0.2, "David": 0.5}
              
            - "rel_to_tail_counts": relation -> {tail: count}
              Raw counts for debugging/analysis
              
            - "rel_to_head_counts": relation -> {head: count}
              Raw counts for head argument types
    
    Purpose:
        - Compute type_fit_true: P(true_tail | relation, head)
        - Compute type_fit_pred: P(predicted_tail | relation, head)
        - Score candidates by how "normal" they are for this relation
        - Used by both Agent A and Agent B in reasoning
    
    Example Usage:
        constraints = build_type_constraints(df_train)
        type_fit_score = constraints["rel_head_to_ranked_tails"].get(
            ("hasChild", "Alice"), {}
        ).get("Bob", 0.0)
        # Result: 0.5 = Bob appears as tail in 50% of (Alice, hasChild) triples
    """
    rel_to_tail_counts = defaultdict(lambda: defaultdict(int))
    rel_to_head_counts = defaultdict(lambda: defaultdict(int))
    rel_to_pair_counts = defaultdict(lambda: defaultdict(int))

    # Count occurrences
    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]
        rel_to_tail_counts[r][t] += 1
        rel_to_head_counts[r][h] += 1
        rel_to_pair_counts[r][(h, t)] += 1

    # Build specific (relation, head) -> ranked tails
    rel_head_to_ranked_tails = defaultdict(dict)
    for r, pair_counts in rel_to_pair_counts.items():
        head_to_tails = defaultdict(dict)
        for (h, t), count in pair_counts.items():
            head_to_tails[h][t] = count
        for h, tail_counts in head_to_tails.items():
            total = sum(tail_counts.values())
            rel_head_to_ranked_tails[(r, h)] = {
                t: round(c / total, 4)
                for t, c in sorted(tail_counts.items(), key=lambda x: -x[1])
            }

    # Build general relation -> tail distribution
    rel_to_tail_dist = {}
    for r, tail_counts in rel_to_tail_counts.items():
        total = sum(tail_counts.values())
        rel_to_tail_dist[r] = {
            t: round(c / total, 4)
            for t, c in sorted(tail_counts.items(), key=lambda x: -x[1])
        }

    return {
        "rel_head_to_ranked_tails": dict(rel_head_to_ranked_tails),
        "rel_to_tail_dist": rel_to_tail_dist,
        "rel_to_tail_counts": dict(rel_to_tail_counts),
        "rel_to_head_counts": dict(rel_to_head_counts),
    }


def get_type_constraint_signal(head, relation, true_tail, predicted, constraints):
    """
    Analyze type fit signal for a (head, relation) query with two candidate tails.
    
    Args:
        head (str): Head entity
        relation (str): Query relation
        true_tail (str): Ground truth tail
        predicted (str): Model's predicted tail (likely wrong)
        constraints (dict): Output from build_type_constraints()
    
    Returns:
        dict with keys:
            - "type_fit_true": P(true_tail | relation, head) — specific
              Falls back to P(true_tail | relation) — general if no specific examples
              
            - "type_fit_pred": P(predicted_tail | relation, head) — same logic
            
            - "type_gap": type_fit_true - type_fit_pred
              Positive = true_tail is more "normal" than predicted
              Negative = predicted is more "normal" (model surprised, or true is rare)
              
            - "expected_tails": List of top 5 tail entities for this relation
              What the relation "usually" expects
              
            - "only_true_has": Relations that true_tail participates in but predicted does NOT
              These are discriminating relations — strong signal
              
            - "only_pred_has": Relations that predicted participates in but true_tail does NOT
              These are "noise" relations — should avoid citing
              
            - "shared_rels": Relations BOTH entities share
              Can be cited by Agent A
    
    Purpose:
        - Diagnose why model failed
        - Provide agents with grounded signals to override model
        - Tell agents which relations are "safe" to cite
    """
    rh_tails = constraints["rel_head_to_ranked_tails"]
    tail_dist = constraints["rel_to_tail_dist"]

    # Specific probability (exact head + relation)
    specific = rh_tails.get((relation, head), {})
    # Fallback to general probability (any head + relation)
    general = tail_dist.get(relation, {})

    type_fit_true = specific.get(true_tail) or general.get(true_tail, 0.0)
    type_fit_pred = specific.get(predicted) or general.get(predicted, 0.0)

    expected_tails = list(specific.keys())[:5] or list(general.keys())[:5]

    tail_counts = constraints["rel_to_tail_counts"]
    true_rels = {r for r, tails in tail_counts.items() if true_tail in tails}
    pred_rels = {r for r, tails in tail_counts.items() if predicted in tails}

    return {
        "type_fit_true": type_fit_true,
        "type_fit_pred": type_fit_pred,
        "type_gap": round(type_fit_true - type_fit_pred, 4),
        "expected_tails": expected_tails,
        "only_true_has": sorted(true_rels - pred_rels),
        "only_pred_has": sorted(pred_rels - true_rels),
        "shared_rels": sorted(true_rels & pred_rels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PREPROCESSING (Heavy — run once!)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_all_triples(
    df,
    split_name,
    entity_to_id,
    relation_to_id,
    id_to_entity,
    id_to_relation,
    model,
    graph,
    df_train,
    embedding_fn,
    failure_summary_fn,
    focused_subgraph_fn,
    hop_classifier_fn,
    similarity_summary_fn,
):
    """
    Compute and cache all model scores, subgraphs, and analysis for every triple.
    
    This is the HEAVY function — calls model.score_hrt for every triple!
    Run ONCE, save to JSON, then reload forever.
    
    Args:
        df: DataFrame with triples to preprocess
        split_name: "train", "test", "valid" — for reference
        model: Pretrained ComplEx model (loaded from HuggingFace)
        graph: Knowledge graph object
        df_train: Training set (to build type constraints)
        *_fn: Helper functions from other modules (avoid circular imports)
    
    Returns:
        List of dicts, one per triple:
        {
            "split": "train|test|valid",
            "head": "entity1",
            "relation": "hasChild",
            "tail": "entity2",
            "true_rank": 42,               # Where ground truth ranks in all candidates
            "predicted": "entity3",        # Model's top prediction (usually wrong)
            "score_true": 0.85,            # Model score for correct triple
            "score_predicted": 0.92,       # Model score for top prediction
            "top5": [                      # Top 5 predictions from model
                ("entity3", 0.92),
                ("entity4", 0.88),
                ...
            ],
            "hop_type": "single|multi",    # 1-hop or 2+ hops in graph?
            "hop_count": 2,                # Number of hops
            "sim_head": "head summary...", # Embedding similarity neighbors
            "sim_tail": "tail summary...",
            "fail_summary": "Model predicted...because...",
            "subgraph": [                  # BFS subgraph around head/tail
                ["entity1", "rel", "entity2"],
                ...
            ],
            "shared_relations": [...],     # Relations true_tail and predicted share
            "only_tail_has": [...],        # DISCRIMINATING SIGNAL
            "only_pred_has": [...],
            "hard_failure": true,          # true_rank > 15% of entities?
        }
    
    Performance:
        - ~1-2 seconds per triple (model.score_hrt is the bottleneck)
        - For 9k triples: ~3-5 hours on GPU
        - Parallelization possible but limits GPU memory
    
    Caching:
        Save immediately to JSON:
            json.dump(train_records, open("CODEX_S_preprocessed_train.json"), indent=2)
        Subsequent runs load from JSON (100x faster):
            json.load(open("CODEX_S_preprocessed_train.json"))
    """
    records = []
    n_entities = len(entity_to_id)
    hard_threshold = max(10, int(n_entities * 0.15))
    constraints = build_type_constraints(df_train)
    total = len(df)

    print(f"[Preprocess] {split_name}: {total} triples")
    print(f"[Preprocess] hard_threshold = rank > {hard_threshold} (top 15%)")

    for i, row in df.iterrows():
        try:
            head = row["head"]
            relation = row["relation"]
            tail = row["tail"]

            # Skip if not in ID mappings
            if relation not in relation_to_id:
                continue
            if head not in entity_to_id:
                continue
            if tail not in entity_to_id:
                continue

            # HEAVY: Score this triple against all candidates
            ranking = model.get_full_ranking_filtered_batched(head, relation, tail)

            # Similarity neighbors using embeddings
            sim_head, _ = similarity_summary_fn(head, k=5)
            sim_tail, _ = similarity_summary_fn(tail, k=5)

            # BFS subgraph extraction
            subgraph = focused_subgraph_fn(
                [head, tail, ranking["predicted"]],
                graph,
                hops=3,
                max_triples=150,
                query_relation=relation,
            )

            # Count hops in graph between head and tail
            hop_type, hops, _ = hop_classifier_fn(head, tail, graph, target_relation=relation)

            # Analyze failure signal
            fail_sum, fail_raw = failure_summary_fn(
                head, relation, tail, ranking["predicted"], constraints
            )

            records.append({
                "split": split_name,
                "head": head,
                "relation": relation,
                "tail": tail,
                "true_rank": int(ranking["true_rank"]),
                "predicted": ranking["predicted"],
                "score_true": float(ranking["true_score"]),
                "score_predicted": float(ranking["predicted_score"]),
                "top5": ranking["full_ranking"][:5],
                "hop_type": hop_type,
                "hop_count": int(hops),
                "sim_head": sim_head,
                "sim_tail": sim_tail,
                "fail_summary": fail_sum,
                "subgraph": subgraph,
                "shared_relations": list(fail_raw["shared"]),
                "only_tail_has": list(fail_raw["only_true"]),
                "only_pred_has": list(fail_raw["only_pred"]),
                "hard_failure": bool(ranking["true_rank"] > hard_threshold),
            })

            if (i + 1) % 50 == 0:
                print(f"  {split_name}: {i+1}/{total}")

        except Exception as e:
            print(f"[Preprocess] ERROR at row {i}: {e}")
            continue

    return records


def load_or_preprocess_triples(
    df_split,
    split_name,
    output_file,
    force=False,
    **preprocess_kwargs
):
    """
    Load preprocessed triples from JSON, or compute if not cached.
    
    Args:
        df_split: DataFrame to preprocess
        split_name: "train", "test", "valid"
        output_file: Path to cache JSON file
        force: If True, recompute even if file exists
        **preprocess_kwargs: Arguments for preprocess_all_triples()
    
    Returns:
        List of preprocessed record dicts
    
    Logic:
        if file exists and not force:
            load from JSON (instant)
        else:
            preprocess (slow)
            save to JSON
            return
    """
    if os.path.exists(output_file) and not force:
        print(f"[Loader] Found {output_file} — loading from disk")
        with open(output_file) as f:
            return json.load(f)

    print(f"[Loader] Computing {output_file} (this will take several hours)...")
    records = preprocess_all_triples(df_split, split_name, **preprocess_kwargs)

    with open(output_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[Loader] Saved {output_file}")

    return records


def load_preprocessed_records(filepath):
    """
    Quick load of cached preprocessed records from JSON.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        List of record dicts (instant load)
    """
    with open(filepath) as f:
        return json.load(f)
