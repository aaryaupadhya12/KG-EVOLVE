"""
KNOWLEDGE GRAPH UTILITIES MODULE
=================================

Graph operations for knowledge graph reasoning.

Key Responsibilities:
  1. Extract subgraphs via BFS (multi-hop neighborhoods)
  2. Focus subgraphs to relevant entities (tier-based ranking)
  3. Classify hops between entities (1-hop, 2-hop, etc.)
  4. Manage type constraint signals
"""

from collections import deque, defaultdict
from typing import List, Set, Tuple, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SUBGRAPH EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_subgraph(
    entity: str,
    graph: Dict[str, List[Tuple[str, str]]],
    hops: int = 2,
    max_triples: int = 150,
) -> List[Tuple[str, str, str]]:
    """
    Extract k-hop subgraph around an entity using BFS.
    
    Args:
        entity (str): Starting entity
        graph (dict): Adjacency list {entity: [(relation, tail), ...]}
        hops (int): Maximum hops from entity
        max_triples (int): Stop extraction after this many triples
    
    Returns:
        List of (head, relation, tail) triples
    
    Algorithm:
        1. Start BFS from entity at depth 0
        2. For each node at current depth:
           - Add all outgoing edges to subgraph
           - Queue unexplored tail entities at next depth
        3. Stop at max_hops or max_triples
    
    Purpose:
        - Get local neighborhood around an entity
        - Used to extract subgraph around head, tail, and predicted entities
        - Provides ground truth paths for agents to reason with
    
    Example:
        graph = {
            "Alice": [("hasChild", "Bob"), ("livesIn", "NYC")],
            "Bob": [("hasAge", "10"), ("livesIn", "NYC")],
            "NYC": [("hasCountry", "USA")],
        }
        
        subgraph = extract_subgraph("Alice", graph, hops=2)
        # Returns:
        # [
        #   ("Alice", "hasChild", "Bob"),
        #   ("Alice", "livesIn", "NYC"),
        #   ("Bob", "hasAge", "10"),
        #   ("Bob", "livesIn", "NYC"),
        #   ("NYC", "hasCountry", "USA"),
        # ]
    
    Notes:
        - Nodes visited multiple times at different depths are handled correctly
        - Each edge appears at most once in output
        - BFS ensures shallowest paths are prioritized
    """
    subgraph = []
    visited = {entity}  # Track visited nodes to avoid cycles
    queue = deque([(entity, 0)])  # (node, depth)

    while queue and len(subgraph) < max_triples:
        node, depth = queue.popleft()
        
        if depth >= hops:
            continue

        # Add all outgoing edges from this node
        for rel, tail in graph.get(node, []):
            if len(subgraph) >= max_triples:
                break
            subgraph.append((node, rel, tail))

            # Queue unvisited neighbors for next hop
            if tail not in visited:
                visited.add(tail)
                queue.append((tail, depth + 1))

    return subgraph


def focused_subgraph(
    entities: List[str],
    graph: Dict[str, List[Tuple[str, str]]],
    hops: int = 2,
    max_triples: int = 100,
    query_relation: str = None,
) -> List[Tuple[str, str, str]]:
    """
    Extract and prioritize subgraph containing multiple entities.
    
    Used to build context containing head, tail, and predicted entities.
    Ranks triples by relevance using a tier system.
    
    Args:
        entities (list): Core entities of interest
        graph (dict): Knowledge graph
        hops (int): Hops per entity
        max_triples (int): Total triples to return
        query_relation (str): If provided, prioritize triples with this relation
    
    Returns:
        List of (head, relation, tail) triples, tier-ranked
    
    Tier System:
        TIER 1: Triples with query_relation (highest priority)
                "Gold signal" — directly matches the query
                Example: If query is "hasChild", rank these first
        
        TIER 2: Triples where both head and tail are in entities
                High confidence — both endpoints are important
        
        TIER 3: Triples where one endpoint is in entities
                Medium relevance — connects to important entity
        
        TIER 4: All other triples
                Low relevance — background knowledge
    
    Example:
        record = {
            "head": "Alice",
            "tail": "Bob",
            "predicted": "Carol",
            "relation": "hasChild"
        }
        
        subgraph = focused_subgraph(
            entities=[record["head"], record["tail"], record["predicted"]],
            graph=graph,
            hops=3,
            max_triples=150,
            query_relation=record["relation"]
        )
        
        # Returns triples, prioritized:
        # - Triples with "hasChild" relation (query_relation)
        # - Triples between Alice-Bob, Alice-Carol, or Bob-Carol
        # - Triples touching any of these 3 entities
        # - All other triples (least relevant)
    
    Implementation:
        1. Extract subgraph around each entity
        2. Remove duplicates
        3. Classify each triple into tier
        4. Concatenate tiers: tier1 + tier2 + tier3 + tier4
        5. Return first max_triples (tier-sorted)
    """
    entity_set = set(entities)
    all_triples = []
    seen = set()

    # Extract subgraph around each entity
    for ent in entities:
        for triple in extract_subgraph(ent, graph, hops, max_triples):
            if triple not in seen:
                seen.add(triple)
                all_triples.append(triple)

    # If we're under budget, return all
    if len(all_triples) <= max_triples:
        return all_triples

    # Otherwise, tier-sort and truncate
    tier1, tier2, tier3, tier4 = [], [], [], []

    for h, r, t in all_triples:
        if query_relation and r == query_relation:
            tier1.append((h, r, t))
        elif h in entity_set and t in entity_set:
            tier2.append((h, r, t))
        elif h in entity_set or t in entity_set:
            tier3.append((h, r, t))
        else:
            tier4.append((h, r, t))

    # Concatenate tiers and truncate
    return (tier1 + tier2 + tier3 + tier4)[:max_triples]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: HOP CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def hop_classifier(
    head: str,
    tail: str,
    graph: Dict[str, List[Tuple[str, str]]],
    target_relation: str = None,
) -> Tuple[str, float, Any]:
    """
    Classify path type and distance between head and tail entities.
    
    Returns:
        Tuple of (hop_type, hop_distance, evidence)
        - hop_type: "single" (1-hop), "multi" (2+ hops), or "none" (disconnected)
        - hop_distance: 1.0, 1.5, 2.0, or 99.0
        - evidence: Explanation of classification
    
    Algorithm:
        1. Check if direct edge (head -r-> tail) exists for target_relation
           → single hop (distance 1.0)
        2. Check if any direct edge (head -any_r-> tail) exists
           → single hop but "multi" since it's not the target relation (distance 1.5)
        3. BFS to find 2-hop paths (head -r1-> mid -r2-> tail)
           → multi hop (distance 2.0)
        4. If no path found → "none" (distance 99.0)
    
    Purpose:
        - Classify query difficulty (single-hop vs multi-hop)
        - Provide context to agents (affects confidence calibration)
        - Route logic (example: if single-hop, prefer Agent A)
    
    Example:
        graph = {
            "Alice": [("hasChild", "Bob"), ("spouse", "David")],
            "Bob": [("livesIn", "NYC")],
            "David": [("livesIn", "NYC")],
        }
        
        # Single hop with correct relation
        hop_type, distance, evidence = hop_classifier(
            "Alice", "Bob", graph, target_relation="hasChild"
        )
        # ("single", 1, "hasChild")
        
        # Single hop but wrong relation
        hop_type, distance, evidence = hop_classifier(
            "Alice", "Bob", graph, target_relation="friend"
        )
        # ("single", 1.5, "direct but via hasChild")
        
        # Double hop
        hop_type, distance, evidence = hop_classifier(
            "Alice", "NYC", graph
        )
        # ("multi", 2, "Alice-hasChild->Bob-livesIn->NYC")
        
        # Disconnected
        hop_type, distance, evidence = hop_classifier(
            "Alice", "Unknown", graph
        )
        # ("none", 99, [])
    """
    # Check for direct edge with target relation
    for relation, t in graph.get(head, []):
        if t == tail:
            if target_relation is None or relation == target_relation:
                return "single", 1, relation

    # Check for direct edge with any relation
    direct_wrong = [r for r, t in graph.get(head, []) if t == tail]
    if direct_wrong:
        return "multi", 1.5, f"direct but via {direct_wrong[0]}"

    # BFS for 2-hop paths
    paths_found = []
    for r1, mid in graph.get(head, []):
        for r2, t2 in graph.get(mid, []):
            if t2 == tail:
                paths_found.append(f"{head}-{r1}->{mid}-{r2}->{tail}")

    if paths_found:
        return "multi", 2, paths_found[0]

    # No path found
    return "none", 99, []


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FAILURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def failure_summary(
    head: str,
    relation: str,
    true_tail: str,
    predicted_tail: str,
    constraints: Dict[str, Any],
    score_fn=None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Diagnose why the model predicted wrong using type constraints and scores.
    
    Args:
        head (str): Head entity
        relation (str): Relation
        true_tail (str): Ground truth tail
        predicted_tail (str): Model's prediction (wrong)
        constraints (dict): Output from build_type_constraints()
        score_fn: Function to get model scores (optional)
    
    Returns:
        Tuple of (summary_text, analysis_dict)
    
    Summary Text Example:
        "Model predicted 'Carol' (score=0.92) over 'Bob' (score=0.85). 
         Type fit: 'Bob'=0.3 vs 'Carol'=0.1 for 'hasChild'. 
         Expected tails: Alice, Bob, David. 
         Only 'Bob' in: [family]. 
         Only 'Carol' in: [colleague]."
    
    Analysis Dict:
        {
            "score_true": 0.85,
            "score_pred": 0.92,
            "shared": set of relations both entities appear in,
            "only_true": relations true_tail has but predicted doesn't,
            "only_pred": relations predicted has but true_tail doesn't,
            "type_fit_true": 0.3,
            "type_fit_pred": 0.1,
            "type_gap": 0.2,  # positive = true_tail more likely
            "expected_tails": [list of likely tails],
        }
    
    Purpose:
        - Understand why model failed
        - Provide agents with diagnostic information
        - Identify which relations are "discriminating signals"
    """
    # Get type constraint signal
    sig = get_type_constraint_signal(head, relation, true_tail, predicted_tail, constraints)

    # Get model scores if function provided
    score_true, score_pred = None, None
    if score_fn:
        score_true = score_fn(head, relation, true_tail)
        score_pred = score_fn(head, relation, predicted_tail)

    summary = (
        f"Model predicted '{predicted_tail}' (score={score_pred:.3f}) "
        f"over '{true_tail}' (score={score_true:.3f}). "
        f"Type fit: '{true_tail}'={sig['type_fit_true']:.3f} vs "
        f"'{predicted_tail}'={sig['type_fit_pred']:.3f} for '{relation}'. "
        f"Expected tails: {', '.join(sig['expected_tails'][:3])}. "
        f"Only '{true_tail}' in: {', '.join(sig['only_true_has'][:5]) or 'none'}. "
        f"Only '{predicted_tail}' in: {', '.join(sig['only_pred_has'][:5]) or 'none'}."
    ) if score_true and score_pred else ""

    return summary, {
        "score_true": score_true,
        "score_pred": score_pred,
        "shared": set(sig["shared_rels"]),
        "only_true": set(sig["only_true_has"]),
        "only_pred": set(sig["only_pred_has"]),
        "type_fit_true": sig["type_fit_true"],
        "type_fit_pred": sig["type_fit_pred"],
        "type_gap": sig["type_gap"],
        "expected_tails": sig["expected_tails"],
    }


def get_type_constraint_signal(
    head: str,
    relation: str,
    true_tail: str,
    predicted: str,
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute type fit signals for true and predicted tails.
    
    Type fit = "How often does this entity appear as tail of this relation?"
    
    Looks up in two levels:
    1. Specific: (relation, head) -> {tail: probability}
       E.g., for specific "Alice", what are common children?
    2. General: (relation) -> {tail: probability}
       E.g., for any parent, what are common children?
    
    Uses specific if available, falls back to general.
    
    Returns:
        dict with:
        - "type_fit_true": P(true_tail | relation, head)
        - "type_fit_pred": P(predicted | relation, head)
        - "type_gap": type_fit_true - type_fit_pred
        - "expected_tails": Top candidates for this relation
        - "only_true_has": Relations true_tail participates in (not predicted)
        - "only_pred_has": Relations predicted participates in (not true_tail)
        - "shared_rels": Relations both participate in
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

    # Find relations each entity participates in
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
