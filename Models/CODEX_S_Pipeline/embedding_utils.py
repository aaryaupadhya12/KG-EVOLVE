"""
EMBEDDING UTILITIES MODULE
==========================

Computes entity similarities using pretrained embeddings.
Used to provide agents with "analogous entities" context.

Key Responsibilities:
  1. Cache entity embeddings globally (expensive to compute repeatedly)
  2. Compute cosine similarity between embeddings
  3. Rank similar entities (by embedding + relational similarity)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


# Global cache
_EMBEDDINGS_CACHE = None


def get_entity_embeddings_cached(model):
    """
    Extract and cache entity embeddings from model.
    
    Args:
        model: Pretrained ComplEx model
    
    Returns:
        tensor: Shape [num_entities, embedding_dim]
    
    Design:
        - Called ONCE during preprocessing
        - Result cached globally
        - Reused for all similarity computations
        - Saves ~seconds per call (avoids re-extracting embeddings)
    
    Note:
        ComplEx has complex embeddings; we take absolute value for real-valued similarity.
    """
    global _EMBEDDINGS_CACHE
    
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    embs = model.entity_representations[0](indices=None).detach().cpu()
    if embs.is_complex():
        embs = embs.abs()
    
    _EMBEDDINGS_CACHE = embs
    return _EMBEDDINGS_CACHE


def similarity_summary(
    entity: str,
    entity_to_id: Dict[str, int],
    id_to_entity: Dict[int, str],
    embeddings: torch.Tensor,
    df_train=None,
    k: int = 5,
) -> Tuple[str, List[Tuple[str, float, int, float]]]:
    """
    Find most similar entities by embedding + relational profile.
    
    Args:
        entity (str): Entity to find neighbors for
        entity_to_id (dict): Label -> ID mapping
        id_to_entity (dict): ID -> Label mapping
        embeddings (tensor): Shape [num_entities, dim] from get_entity_embeddings_cached()
        df_train (df): Training dataframe to compute relation overlap
        k (int): Number of neighbors to return
    
    Returns:
        Tuple of (summary_string, results_list)
        
        Summary String Example:
            "Alice most similar to: Bob(sim=0.89,shared=3,rel=0.67), 
                                    Carol(sim=0.85,shared=2,rel=0.50), ..."
        
        Results List:
            List of (entity_name, embedding_score, num_shared_relations, rel_score)
    
    Algorithm:
        1. Get embedding for query entity
        2. Compute cosine similarity to all other embeddings
        3. Sort by embedding similarity descending
        4. For top candidates:
           - Count overlapping relations (entities they both appear in)
           - Compute relational Jaccard: overlap / union
           - Combine embedding + relational signal
        5. Return top k
    
    Purpose:
        - Give agents "analogous entities" context
        - Example: "Alice is similar to Carol (more successful examples with Carol)"
        - Helps agents by learning from similar reasoning patterns
    
    Example:
        summary, results = similarity_summary(
            "Alice",
            entity_to_id=entity_to_id,
            id_to_entity=id_to_entity,
            embeddings=embeddings,
            df_train=df_train,
            k=5
        )
        
        print(summary)
        # "Alice most similar to: Bob(sim=0.89,shared=3,rel=0.67), 
        #                         Carol(sim=0.85,shared=2,rel=0.50), ..."
        
        print(results)
        # [("Bob", 0.89, 3, 0.67), ("Carol", 0.85, 2, 0.50), ...]
    """
    if entity not in entity_to_id:
        return f"{entity} not in entity_to_id", []

    e_id = entity_to_id[entity]
    e_vec = embeddings[e_id]

    # Cosine similarity to all entities
    sims = F.cosine_similarity(
        e_vec.unsqueeze(0), embeddings
    ).detach().cpu().numpy()

    # Sort by similarity descending (most similar first)
    ranked = np.argsort(sims)[::-1]

    # Get relations this entity participates in
    if df_train is not None:
        entity_rels = get_entity_relations(entity, df_train)
    else:
        entity_rels = set()

    results = []

    for idx in ranked:
        name = id_to_entity[idx]
        
        # Skip self
        if name == entity:
            continue

        # Embedding similarity score [0, 1]
        score = float(sims[idx])

        # Relational overlap
        if df_train is not None:
            neighbor_rels = get_entity_relations(name, df_train)
            shared = len(entity_rels & neighbor_rels)
            total = len(entity_rels | neighbor_rels)
            rel_score = shared / total if total > 0 else 0.0
        else:
            shared = 0
            rel_score = 0.0

        results.append((name, score, shared, rel_score))

        if len(results) == k:
            break

    # Format as string for agents
    parts = [
        f"{n}(sim={s:.2f},shared={sh},rel={rs:.2f})"
        for n, s, sh, rs in results
    ]
    summary = f"{entity} most similar to: {', '.join(parts)}"

    return summary, results


def get_entity_relations(entity: str, df):
    """
    Find all relations an entity participates in.
    
    Args:
        entity (str): Entity to analyze
        df: DataFrame with columns ["head", "relation", "tail"]
    
    Returns:
        set: All unique relations where entity is head or tail
    
    Example:
        relations = get_entity_relations("Alice", df_train)
        # {"hasChild", "spouse", "livesIn"}
    """
    head_rels = set(df[df["head"] == entity]["relation"])
    tail_rels = set(df[df["tail"] == entity]["relation"])
    return head_rels | tail_rels


def embedding_distance(
    entity1: str,
    entity2: str,
    entity_to_id: Dict[str, int],
    embeddings: torch.Tensor,
) -> float:
    """
    Cosine similarity between two entities' embeddings.
    
    Args:
        entity1, entity2 (str): Entity labels
        entity_to_id (dict): ID mapping
        embeddings (tensor): Precomputed embeddings
    
    Returns:
        float: Cosine similarity in [0, 1]
    
    Example:
        sim = embedding_distance("Alice", "Bob", entity_to_id, embeddings)
        # 0.87 (fairly similar)
    """
    if entity1 not in entity_to_id or entity2 not in entity_to_id:
        return 0.0

    id1, id2 = entity_to_id[entity1], entity_to_id[entity2]
    vec1 = embeddings[id1].unsqueeze(0)
    vec2 = embeddings[id2].unsqueeze(0)

    sim = F.cosine_similarity(vec1, vec2).item()
    return max(0.0, float(sim))  # Clamp to [0, 1]


def find_similar_cluster(
    entities: List[str],
    entity_to_id: Dict[str, int],
    id_to_entity: Dict[int, str],
    embeddings: torch.Tensor,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Find entity cluster similar to a set of entities.
    
    Useful for finding "backup" entities that are similar to all three:
    head, tail, predicted entities.
    
    Args:
        entities: List of entity labels
        k: Number of neighbors to return
    
    Returns:
        List of (entity, avg_similarity) sorted by similarity descending
    
    Algorithm:
        1. Average embeddings of input entities
        2. Find most similar entities to average
        3. Return top k
    
    Example:
        # Find entities similar to head, tail, and predicted
        cluster = find_similar_cluster(
            [record["head"], record["tail"], record["predicted"]],
            entity_to_id,
            id_to_entity,
            embeddings,
            k=5
        )
        # [("Carol", 0.82), ("David", 0.79), ...]
    """
    valid_entities = [e for e in entities if e in entity_to_id]
    
    if not valid_entities:
        return []

    # Get IDs and average embedding
    ids = [entity_to_id[e] for e in valid_entities]
    avg_embedding = embeddings[ids].mean(dim=0).unsqueeze(0)

    # Similarity to average
    sims = F.cosine_similarity(avg_embedding, embeddings).detach().cpu().numpy()

    # Sort descending
    ranked = np.argsort(sims)[::-1]

    results = []
    for idx in ranked:
        name = id_to_entity[idx]
        
        # Skip input entities
        if name in valid_entities:
            continue

        score = float(sims[idx])
        results.append((name, score))

        if len(results) == k:
            break

    return results
