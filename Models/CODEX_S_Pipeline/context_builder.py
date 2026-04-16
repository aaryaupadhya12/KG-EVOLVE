"""
CONTEXT BUILDER MODULE
======================

Constructs rich context strings from records for agent reasoning.

Agents (A and B) need:
  1. Clear problem statement (head, relation, wrong prediction)
  2. Grounded signals (subgraph, type constraints)
  3. Separating relations (which entities don't share relations)
  4. Historical hints (episodic memory)
  
This module creates human-readable prose from all signals.
"""

from typing import Dict, List, Optional, Any


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: SUBGRAPH FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def trim_subgraph(
    subgraph: List,
    head: str,
    true_tail: str,
    predicted: str,
    only_tail_has: List[str],
    max_triples: int = 12,
) -> List:
    """
    Tier and truncate subgraph for agent context.
    
    Prioritizes triples with "separating relations"
    (relations only true_tail has, not predicted).
    
    Args:
        subgraph: List of triple tuples/dicts
        head: Head entity
        true_tail: Ground truth tail
        predicted: Model's prediction
        only_tail_has: Relations true_tail has that predicted doesn't
        max_triples: Max triples to return
    
    Returns:
        List of triples, tier-ranked
    
    Tier System:
        TIER 1: Triples with "separating relations"
                Highest value — discriminates between true and predicted
        
        TIER 2: Triples where one endpoint is true_tail or predicted
                Medium value — relevant to our choice
        
        TIER 3: Triples where one endpoint is head
                Lower value — context about query
        
        TIER 4: All other triples
                Lowest value — background
    
    Example:
        trimmed = trim_subgraph(
            record["subgraph"],
            head="Alice",
            true_tail="Bob",
            predicted="Carol",
            only_tail_has=["hasChild", "livesIn"],
            max_triples=12
        )
        # Returns subgraph prioritizing hasChild and livesIn triples first
    """
    oth_set = set(only_tail_has)
    tier1, tier2, tier3, tier4 = [], [], [], []

    for triple in subgraph:
        # Parse triple (can be list, tuple, or dict)
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            h, r, t = triple
        elif isinstance(triple, dict):
            h = triple.get("head", "")
            r = triple.get("relation", "")
            t = triple.get("tail", "")
        else:
            continue

        # Classify by tier
        if r in oth_set:
            tier1.append(triple)
        elif t in (true_tail, predicted) or h in (true_tail, predicted):
            tier2.append(triple)
        elif h == head or t == head:
            tier3.append(triple)
        else:
            tier4.append(triple)

    return (tier1 + tier2 + tier3 + tier4)[:max_triples]


def build_subgraph_str(
    record: Dict[str, Any],
    max_triples: int = 12,
) -> str:
    """
    Convert subgraph to readable prose with visual markers.
    
    Args:
        record: Preprocessed record dict
        max_triples: Max triples to format
    
    Returns:
        str: Multi-line graph visualization
    
    Output Format:
        Alice --hasChild--> Bob ◆
        Alice --spouse--> David
        Bob --livesIn--> NYC
        
        ◆ marks "separating relations" (special signal)
    
    Example:
        context = build_subgraph_str(record)
        print(context)
        # Alice --hasChild--> Bob ◆
        # Alice --spouse--> David
        # Bob --livesIn--> NYC ◆
    """
    true_tail = record.get("true_tail") or record.get("tail", "")
    trimmed = trim_subgraph(
        record.get("subgraph", []),
        record["head"],
        true_tail,
        record.get("predicted", ""),
        record.get("only_tail_has", []),
        max_triples,
    )

    if not trimmed:
        return "  (no subgraph available)"

    only_set = set(record.get("only_tail_has", []))
    lines = []

    for triple in trimmed:
        # Parse triple format
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            h, r, t = triple
        elif isinstance(triple, dict):
            h = triple.get("head", "?")
            r = triple.get("relation", "?")
            t = triple.get("tail", "?")
        else:
            continue

        # Mark separating relations
        marker = " ◆" if r in only_set else ""
        lines.append(f"  {h} --{r}--> {t}{marker}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FULL CONTEXT BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_context(
    record: Dict[str, Any],
    tsv_memory: Optional[Dict] = None,
    episodic_hint: str = "",
) -> str:
    """
    Build comprehensive context string for agent reasoning.
    
    Args:
        record: Preprocessed record dict with all fields
        tsv_memory: Optional TSV memory dict
        episodic_hint: Optional episodic memory query result
    
    Returns:
        str: Full formatted context for agent prompt
    
    Context Sections:
        1. Query statement (head, relation, rank)
        2. Separating relations (prime signal)
        3. Episodic memory (past examples)
        4. Subgraph (local neighborhood)
        5. Similarity (analogous entities)
        6. Metadata (hop type, failure analysis)
    
    Example:
        context = build_agent_context(record, tsv_memory=tsv_memory)
        
        print(context)
        # Triple: (Alice, hasChild, ?)
        # Predicted wrong: Carol  |  rank of correct answer: 42
        # 
        # ⚠ SEPARATING RELATIONS — cite these or state you cannot:
        #   ◆ spouse
        #   ◆ livesIn
        # 
        # Subgraph (◆ = gold signal triple — use these first):
        #   Alice --hasChild--> Bob ◆
        #   Alice --spouse--> David ◆
        #   ...
    """
    true_tail = record.get("true_tail") or record.get("tail", "")
    predicted = record.get("predicted", "unknown")
    only_tail = record.get("only_tail_has", [])
    only_pred = record.get("only_pred_has", [])

    # 1. MEMORY BLOCK
    memory_block = ""
    if tsv_memory:
        from memory_manager import get_memory_hint
        hint = get_memory_hint(record["head"], tsv_memory)
        if hint:
            memory_block = f"\nMemory prior for {record['head']}:\n{hint}\n"

    # 2. SEPARATING RELATIONS BLOCK
    if only_tail:
        separator_block = (
            "⚠ SEPARATING RELATIONS — cite these or state you cannot:\n"
            + "\n".join(f"  ◆ {r}" for r in only_tail)
            + f"\nThese connect [{record['head']}] → [{true_tail}] "
            f"but NOT → [{predicted}].\n"
        )
    else:
        separator_block = (
            "⚠ NO SEPARATING RELATIONS in subgraph.\n"
            "Forbidden (connect head to WRONG entity — do not cite):\n"
            + ("\n".join(f"  ✗ {r}" for r in only_pred) or "  none")
            + "\nYou must find a multi-hop path OR output confidence 0.3.\n"
        )

    # 3. SIMILARITY BLOCK
    sim_head = record.get("sim_head", "")
    if sim_head:
        # Trim to first 3 for brevity
        sim_head = ", ".join(sim_head.split(", ")[:3])

    # 4. ASSEMBLE FULL CONTEXT
    return (
        f"Triple: ({record['head']}, {record['relation']}, ?)\n"
        f"Predicted wrong: {predicted}  |  rank of correct answer: {record['true_rank']}\n\n"
        f"{separator_block}\n"
        f"{memory_block}\n"
        f"Subgraph (◆ = gold signal triple — use these first):\n"
        f"{build_subgraph_str(record)}\n\n"
        f"Similarity (top-3 embedding neighbours of head):\n"
        f"  {sim_head}\n\n"
        f"Hop type: {record.get('hop_type','multi')}\n"
        f"Failure: {record.get('fail_summary','')}"
    )


def build_context_minimal(record: Dict[str, Any]) -> str:
    """
    Build minimal context (for testing or low-resource environments).
    
    Args:
        record: Preprocessed record dict
    
    Returns:
        str: Compact context string
    
    Compared to build_agent_context(), this:
        - Omits similarity block
        - Omits memory blocks
        - Uses compact subgraph format
    
    Useful when:
        - GPU memory is limited
        - LLM context window is tight
        - Just testing agent logic
    """
    true_tail = record.get("true_tail") or record.get("tail", "")
    predicted = record.get("predicted", "unknown")
    only_tail = record.get("only_tail_has", [])

    separator_block = (
        "Discriminating relations (cite these):\n"
        + "\n".join(f"  {r}" for r in only_tail[:3])
        if only_tail
        else "No discriminating relations."
    )

    return (
        f"Query: ({record['head']}, {record['relation']}, ?)\n"
        f"Model gave: {predicted} (rank {record['true_rank']})\n\n"
        f"{separator_block}\n\n"
        f"Subgraph:\n"
        f"{build_subgraph_str(record, max_triples=5)}"
    )


def build_agent_context_with_memory(
    record: Dict[str, Any],
    memory_manager,
) -> str:
    """
    Build context using unified memory manager.
    
    Args:
        record: Preprocessed record dict
        memory_manager: MemoryManager instance
    
    Returns:
        str: Full context with memory hints
    
    This is preferred over build_agent_context()
    if using the new MemoryManager class.
    """
    # Get memory context
    mem_context = memory_manager.get_context_for_query(
        record["head"],
        record["relation"],
    )

    true_tail = record.get("true_tail") or record.get("tail", "")
    predicted = record.get("predicted", "unknown")
    only_tail = record.get("only_tail_has", [])

    separator_block = (
        "⚠ SEPARATING RELATIONS:\n"
        + "\n".join(f"  ◆ {r}" for r in only_tail)
        if only_tail
        else "⚠ NO SEPARATING RELATIONS."
    )

    memory_block = ""
    if mem_context["episodic"]:
        memory_block += f"\nEpisodic memory:\n{mem_context['episodic']}\n"
    if mem_context["tsv"]:
        memory_block += f"\nPast examples:\n{mem_context['tsv']}\n"

    return (
        f"Triple: ({record['head']}, {record['relation']}, ?)\n"
        f"Model predicted: {predicted} (rank {record['true_rank']})\n\n"
        f"{separator_block}\n"
        f"{memory_block}\n"
        f"Subgraph:\n"
        f"{build_subgraph_str(record)}\n\n"
        f"Hop type: {record.get('hop_type','multi')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FAILURE CLASSIFICATION CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def classify_failure_type(record: Dict[str, Any]) -> str:
    """
    Classify the type of KG reasoning failure based on record signals.
    
    Args:
        record: Preprocessed record dict
    
    Returns:
        str: Failure type classification
    
    Failure Types:
        - "similarity_confusion": Model confused by embedding similarity
          (true_tail and predicted both seem plausible)
        
        - "structural_gap": True tail requires multi-hop reasoning
          and model failed to rank correctly
        
        - "type_fit_gap": Type constraint suggests predicted is wrong
          (e.g., predicted entity never appears as tail of this relation)
        
        - "both_failed": Both type fit and structure fail
        
        - "complex": Unclear cause, requires investigation
    
    Example:
        ftype = classify_failure_type(record)
        # "type_fit_gap"
    """
    only_tail = set(record.get("only_tail_has", []))
    only_pred = set(record.get("only_pred_has", []))
    hop_type = record.get("hop_type", "multi")
    rank = record.get("true_rank", 99)

    # Type fit gap: low probability for true_tail
    type_fit_true = record.get("type_fit_true", 0.0) if "type_fit_true" in record else None
    type_fit_pred = record.get("type_fit_pred", 0.0) if "type_fit_pred" in record else None

    has_type_gap = type_fit_true is not None and type_fit_pred is not None and (
        type_fit_true < 0.1 or (type_fit_true - type_fit_pred) > 0.2
    )

    # Structural gap: requires multi-hop, model misranked
    has_structural_gap = hop_type == "multi" and rank > 50

    # Similarity confusion: both entities seem plausible
    has_similarity_confusion = len(only_tail) == 0 and len(only_pred) == 0

    # Classify
    if has_type_gap and has_structural_gap:
        return "both_failed"
    elif has_type_gap:
        return "type_fit_gap"
    elif has_structural_gap:
        return "structural_gap"
    elif has_similarity_confusion:
        return "similarity_confusion"
    else:
        return "complex"
