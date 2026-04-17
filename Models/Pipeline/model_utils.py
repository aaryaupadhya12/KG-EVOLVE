"""
MODEL UTILITIES MODULE
======================

Handles ComplEx model loading and inference.
ComplEx is a tensor factorization model for knowledge graph completion.

Key Responsibilities:
  1. Load pretrained ComplEx from HuggingFace
  2. Score triples using model's score_hrt() method
  3. Rank candidates for a given (head, relation) query
  4. Extract entity embeddings for similarity computations

Design:
  - Model is loaded ONCE and cached globally
  - score_hrt() expects integer IDs matching entity_to_id, relation_to_id
  - ID mapping MUST NOT change between preprocessing and inference
  - All scores use the same pretrained model (no fallback training)
"""

import pickle
from typing import Dict, List, Tuple

import torch
from huggingface_hub import hf_hub_download


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

# Global model cache
_LOADED_MODEL = None
_MODEL_REPO = "aaryaupadhya20/codex-s-complex-winner"


def load_pretrained_model(repo_id=None, filename="trained_pipeline.pkl"):
    """
    Load pretrained ComplEx model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID. Defaults to CODEX-S ComplEx winner.
        filename: Model filename in repo. Should be "trained_pipeline.pkl"
    
    Returns:
        model: Pretrained ComplEx model object with score_hrt() method
    
    Details:
        - Downloads model from HuggingFace to local cache
        - Subsequent calls return cached version (instant)
        - Model is PyTorch ComplEx from PyKEEN library
        - Supports score_hrt(tensor) for batch scoring
    
    Example:
        model = load_pretrained_model()
        device = next(model.parameters()).device
        print(f"Model loaded on {device}")
    """
    global _LOADED_MODEL
    
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    if repo_id is None:
        repo_id = _MODEL_REPO

    print(f"[Model] Loading from {repo_id}/{filename}...")

    # hf_hub_download returns path to cached model
    # First call downloads, subsequent calls return cached path instantly
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    with open(model_path, "rb") as f:
        pipeline_result = pickle.load(f)

    _LOADED_MODEL = pipeline_result.model
    _LOADED_MODEL.eval()  # Set to eval mode (no dropout, frozen batch norm)

    device = next(_LOADED_MODEL.parameters()).device
    print(f"[Model] ComplEx loaded on {device}")

    return _LOADED_MODEL


def get_model():
    """
    Retrieve already-loaded model or load if not cached.
    
    Returns:
        Cached model or loads on first call
    """
    if _LOADED_MODEL is None:
        load_pretrained_model()
    return _LOADED_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TRIPLE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_triple(head_id, relation_id, tail_id, model=None):
    """
    Score a single triple using the pretrained model.
    
    Args:
        head_id (int): Head entity ID
        relation_id (int): Relation ID
        tail_id (int): Tail entity ID
        model: Model object. If None, uses loaded model.
    
    Returns:
        float: Score (higher = more likely to be true)
    
    Details:
        - ComplEx scores using tensor factorization
        - Higher scores indicate triples more likely to exist in KG
        - Score ranges vary; no fixed bounds (can be negative)
        - Used to rank candidate tails
    
    Example:
        # Score (Alice, hasChild, Bob)
        score = score_triple(
            head_id=entity_to_id["Alice"],
            relation_id=relation_to_id["hasChild"],
            tail_id=entity_to_id["Bob"]
        )
        # Result: 0.85 (plausible)
        
        # Score (Alice, hasChild, London)
        score = score_triple(
            head_id=entity_to_id["Alice"],
            relation_id=relation_to_id["hasChild"],
            tail_id=entity_to_id["London"]
        )
        # Result: -0.45 (implausible)
    """
    if model is None:
        model = get_model()

    hrs = torch.tensor([[head_id, relation_id, tail_id]])
    with torch.no_grad():
        score = model.score_hrt(hrs).squeeze().item()
    return float(score)


def score_batch(triples, model=None):
    """
    Score multiple triples efficiently using batch processing.
    
    Args:
        triples: List of (head_id, relation_id, tail_id) tuples
        model: Model object. If None, uses loaded model.
    
    Returns:
        List of float scores corresponding to input triples
    
    Example:
        triples = [
            (entity_to_id["Alice"], rel_id, entity_to_id["Bob"]),
            (entity_to_id["Alice"], rel_id, entity_to_id["Carol"]),
        ]
        scores = score_batch(triples)
        # [0.85, 0.72]
    """
    if model is None:
        model = get_model()

    hrs = torch.tensor(triples)
    with torch.no_grad():
        scores = model.score_hrt(hrs).squeeze(-1).cpu().tolist()
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RANKING AND FULL CANDIDATE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def get_full_ranking_filtered_batched(
    head,
    relation,
    true_tail,
    entity_to_id,
    relation_to_id,
    id_to_entity,
    batch_size=512,
    model=None,
):
    """
    Rank ALL candidate entities for a (head, relation) query.
    This is the core model inference function!
    
    Args:
        head (str): Head entity label
        relation (str): Relation label
        true_tail (str): Ground truth tail (to find its rank)
        entity_to_id (dict): Label -> ID mapping
        relation_to_id (dict): Label -> ID mapping
        id_to_entity (dict): ID -> Label mapping
        batch_size (int): Number of candidates per batch (tune for GPU memory)
        model: Model object. If None, uses loaded model.
    
    Returns:
        dict with keys:
            - "predicted": Top-ranked entity label
            - "predicted_score": Score of top entity
            - "true_tail": Ground truth tail label
            - "true_score": Score of ground truth
            - "true_rank": Rank of ground truth (1-indexed)
              Example: true_rank=1 means model ranked correctly
                       true_rank=100 means model ranked it 100th
            - "full_ranking": Full list of (entity, score) tuples, sorted by score desc
    
    Algorithm:
        1. Convert head, relation to IDs
        2. Get all entity IDs (excluding head, to avoid self-loops)
        3. Score all (head, relation, candidate_tail) triples
        4. Sort candidates by score descending
        5. Find position of true_tail in ranking
        6. Return all results
    
    Performance:
        - Batching avoids GPU memory errors
        - ~1-2 seconds for 9000 candidates on GPU
        - Bottleneck of preprocessing pipeline
    
    Example:
        ranking = get_full_ranking_filtered_batched(
            head="Alice",
            relation="hasChild",
            true_tail="Bob",
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            id_to_entity=id_to_entity
        )
        
        print(ranking["predicted"])      # "Carol" (top prediction)
        print(ranking["true_rank"])      # 42 (Bob ranked 42nd)
        print(ranking["full_ranking"][:5])  # Top 5 candidates
    """
    if model is None:
        model = get_model()

    h_id = entity_to_id[head]
    r_id = relation_to_id[relation]

    # All candidate tail IDs except head (avoid self-loops)
    all_ids = [i for i in range(len(entity_to_id)) if id_to_entity[i] != head]

    all_scores = []

    # Score in batches to manage GPU memory
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        # Create tensor of [h_id, r_id, candidate_tail_id] for each candidate
        hrs = torch.tensor([[h_id, r_id, t_id] for t_id in batch_ids])
        
        with torch.no_grad():
            scores = model.score_hrt(hrs).squeeze(-1).cpu().tolist()
        
        # Pair each candidate with its score
        all_scores.extend(zip([id_to_entity[j] for j in batch_ids], scores))

    # Sort by score descending (highest score first)
    all_scores.sort(key=lambda x: -x[1])
    ranked_entities = [e for e, s in all_scores]

    # Find rank of true tail (1-indexed)
    true_rank = ranked_entities.index(true_tail) + 1

    # Extract true tail's score from all_scores list
    true_score = next(s for e, s in all_scores if e == true_tail)

    return {
        "predicted": all_scores[0][0],          # Top ranked entity
        "predicted_score": all_scores[0][1],    # Top entity's score
        "true_tail": true_tail,
        "true_score": true_score,               # True tail's score
        "true_rank": true_rank,                 # Position in ranking
        "full_ranking": [(e, round(s, 3)) for e, s in all_scores],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: EMBEDDINGS FOR SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def get_entity_embeddings(model=None):
    """
    Extract entity embeddings from the pretrained model.
    
    Args:
        model: Model object. If None, uses loaded model.
    
    Returns:
        tensor: Shape [num_entities, embedding_dim]
        Each row is the embedding vector for one entity (in ID order)
    
    Details:
        - ComplEx has complex-valued embeddings
        - Returns absolute value (magnitude) to get real-valued embeddings
        - Used for entity similarity computations
        - Should be cached globally and reused (expensive to compute repeatedly)
    
    Memory:
        - For 9000 entities with 256-dim embeddings: ~9MB (float32)
        - Cached once at start of preprocessing, reused for all entities
    
    Example:
        embeddings = get_entity_embeddings()
        # Shape: [9000, 256]
        # embeddings[entity_id] = embedding vector for that entity
    """
    if model is None:
        model = get_model()

    # model.entity_representations[0] is the embedding layer
    embs = model.entity_representations[0](indices=None).detach().cpu()
    
    # ComplEx has complex embeddings; convert to real by taking absolute value
    if embs.is_complex():
        embs = embs.abs()
    
    return embs


def get_relation_embeddings(model=None):
    """
    Extract relation embeddings from the pretrained model.
    
    Args:
        model: Model object. If None, uses loaded model.
    
    Returns:
        tensor: Shape [num_relations, embedding_dim]
    
    Note:
        Not commonly used in this pipeline, but available for extensions.
    """
    if model is None:
        model = get_model()

    embs = model.relation_representations[0](indices=None).detach().cpu()
    
    if embs.is_complex():
        embs = embs.abs()
    
    return embs
