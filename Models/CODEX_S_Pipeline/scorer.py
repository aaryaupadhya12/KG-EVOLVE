"""
SCORER & AGGREGATOR MODULE
==========================

Responsible for:
  1. Verifying agent claims (hallucination detection)
  2. Computing quality scores for agent outputs
  3. Routing decisions to Agent A or B
  4. Aggregating final answers with reasoning
"""

import json
import time
from typing import Dict, List, Optional, Tuple
from transformers import pipeline as hf_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_type_fit(agent_out: Dict, constraints: Dict) -> Dict:
    """
    Verify that Agent B's cited relations actually exist.
    
    Agent B claims: "I'm confident because these key_relations support it"
    Verification: "Do these relations actually exist in training data?"
    
    Args:
        agent_out (dict): Agent B output with "key_relations" field
        constraints (dict): Type constraints from build_type_constraints()
    
    Returns:
        dict with:
        - "verified": bool, all relations confirmed?
        - "hallucinated": list of relations not in training data
        - "confirmed": list of relations that exist
    
    Purpose:
        Detect when Agent B invents relations.
        Remove hallucinations before using for routing/reasoning.
    
    Example:
        verify = verify_type_fit(agent_b_output, constraints)
        if not verify["verified"]:
            print(f"Agent B hallucinated: {verify['hallucinated']}")
            # Remove from key_relations before scoring
    """
    key_rels = agent_out.get("key_relations", [])
    
    if not key_rels:
        return {"verified": True, "hallucinated": [], "confirmed": []}

    all_rels = set(constraints["rel_to_tail_counts"].keys())
    confirmed = [r for r in key_rels if r in all_rels]
    hallucinated = [r for r in key_rels if r not in all_rels]

    if hallucinated:
        print(f"  [Verify B] ✗ Hallucinated: {hallucinated}")
        agent_out["key_relations"] = confirmed
    else:
        print(f"  [Verify B] ✓ All relations confirmed")

    return {
        "verified": len(hallucinated) == 0,
        "confirmed": confirmed,
        "hallucinated": hallucinated,
    }


def verify_relations(
    claimed_rels: List[str],
    head: str,
    tail: str,
    df_ref,
) -> Dict:
    """
    Verify that Agent A's cited shared relations are real.
    
    Agent A claims: "These relations are shared by true_tail and predicted"
    Verification:
        Query df_ref: (head, relation1, true_tail)?
        Query df_ref: (head, relation1, predicted)?
        Do both exist?
    
    Args:
        claimed_rels (list): Relations Agent A claimed as "shared"
        head (str): Head entity
        tail (str): Tail entity we're checking
        df_ref: Training dataframe
    
    Returns:
        dict with:
        - "verified": list of confirmed relations
        - "hallucinated": list of claimed but non-existent triples
        - "rate": hallucination_rate = n_hallucinated / total_claimed
    
    Purpose:
        Verify Agent A didn't invent shared relations.
    
    Example:
        verify = verify_relations(
            ["hasChild", "livesIn"],
            head="Alice",
            tail="Bob",
            df_ref=df_train
        )
        
        If (Alice, hasChild, Bob) exists but (Alice, livesIn, Bob) doesn't:
        # verified: ["hasChild"]
        # hallucinated: ["livesIn"]
        # rate: 0.5
    """
    if not claimed_rels:
        return {"verified": [], "hallucinated": [], "rate": 0.0}

    # Check which claimed relations actually connect (head, tail) in data
    actual = set(df_ref[(df_ref["head"] == head) & (df_ref["tail"] == tail)]["relation"])
    
    verified = [r for r in claimed_rels if r in actual]
    hallucinated = [r for r in claimed_rels if r not in actual]

    return {
        "verified": verified,
        "hallucinated": hallucinated,
        "rate": len(hallucinated) / max(len(claimed_rels), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: GROUNDED SCORING
# ─────────────────────────────────────────────────────────────────────────────

def compute_quality_score(
    agent_out: Dict,
    record: Dict,
    agent_name: str,
    df_ref=None,
    constraints=None,
) -> Dict:
    """
    Compute quality_score for an agent's output.
    
    Quality Score measures:
        1. COVERAGE: Did agent cite "only_tail_has" relations?
           coverage = (overlapping "only_tail_has" relations) / total "only_tail_has"
        2. CONTAMINATION: Did agent cite "only_pred_has" relations (noise)?
           contamination = (overlapping "only_pred_has" relations) / agent's citations
        3. QUALITY: quality = coverage * (1 - 0.5 * contamination)
           Heavily penalizes contamination (false positive citations)
    
    Args:
        agent_out (dict): Agent A or B output
        record (dict): Preprocessed record with "only_tail_has", "only_pred_has"
        agent_name (str): "A" or "B" for logging
        df_ref (df): Reference data for verification
        constraints (dict): For Agent B verification
    
    Returns:
        dict with:
        - "agent": agent name
        - "relation_score": coverage metric
        - "quality_score": final score [0, 1]
        - "contamination": false positive rate
        - "overlap_tail": cited relations in "only_tail_has"
        - "overlap_pred": cited relations in "only_pred_has" (bad)
        - "agent_relations": all relations cited by agent
        - "prediction_correct": Did agent predict correctly?
        - "confidence": Agent's confidence
    
    Example:
        score = compute_quality_score(agent_a_output, record, "A", df_train)
        
        If "only_tail_has" = ["hasChild", "spouse"]
        and agent cited ["hasChild", "livesIn"]
        
        overlap_tail = ["hasChild"] → coverage = 0.5
        if agent has no contamination: quality = 0.5
    
    Algorithm:
        1. Get cited relations from agent (shared_relations for A, key_relations for B)
        2. Verify claims (remove hallucinations)
        3. Overlap with "only_tail_has" (good)
        4. Overlap with "only_pred_has" (bad)
        5. Compute quality = coverage * (1 - 0.5 * contamination)
        6. Check if prediction is correct
    """
    # Verification step: remove hallucinated claims
    if agent_name == "B" and constraints:
        pv = verify_type_fit(agent_out, constraints)
        if not pv["verified"]:
            path_verified = False
        else:
            path_verified = True
    elif agent_name == "A" and df_ref is not None:
        true_tail = record.get("true_tail") or record.get("tail", "")
        rv = verify_relations(
            agent_out.get("shared_relations", []),
            record["head"],
            true_tail,
            df_ref
        )
        if rv["rate"] > 0:
            print(f"  [Verify A] ✗ Hallucinated: {rv['hallucinated']}")

    # Get cited relations (A uses shared_relations, B uses key_relations)
    agent_rels = set(agent_out.get("shared_relations", []) if agent_name == "A"
                     else agent_out.get("key_relations", []))

    # Gold signals
    only_tail = set(record.get("only_tail_has", []))
    only_pred = set(record.get("only_pred_has", []))

    # Compute overlap
    overlap_tail = agent_rels & only_tail  # Good: agent cited separating relations
    overlap_pred = agent_rels & only_pred  # Bad: agent cited wrong-entity relations

    # Compute metrics
    coverage_score = len(overlap_tail) / max(len(only_tail), 1)
    contamination = len(overlap_pred) / max(len(agent_rels), 1)

    # Quality = coverage with contamination penalty
    quality_score = round(coverage_score * (1 - 0.5 * contamination), 3)

    # Check correctness
    true_tail = record.get("true_tail") or record.get("tail", "")
    prediction = (agent_out.get("prediction", "").strip().lower()).strip()
    correct = prediction == true_tail.strip().lower()

    print(f"  [Score {agent_name}] coverage={coverage_score:.2f}  "
          f"contam={contamination:.2f}  quality={quality_score}  correct={correct}")

    return {
        "agent": agent_name,
        "relation_score": round(coverage_score, 3),
        "quality_score": quality_score,
        "contamination": round(contamination, 3),
        "overlap_tail": sorted(overlap_tail),
        "overlap_pred": sorted(overlap_pred),
        "agent_relations": sorted(agent_rels),
        "only_tail_has": sorted(only_tail),
        "prediction_correct": correct,
        "confidence": agent_out.get("confidence", 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def route_decision(score_a: Dict, score_b: Dict, record: Dict) -> str:
    """
    Route decision to Agent A or B based on quality scores.
    
    Routing Rules (in priority order):
        1. If one has quality > 0.5 and other ≤ 0.5 → Use the one > 0.5
        2. If both have positive quality → Use lower contamination
        3. Tie-break by hop type:
           - single-hop → prefer A (type constraints more reliable)
           - multi-hop → prefer B (structural reasoning needed)
    
    Args:
        score_a (dict): Score output for Agent A
        score_b (dict): Score output for Agent B
        record (dict): Record with "hop_type"
    
    Returns:
        str: "A" or "B"
    
    Purpose:
        Select most reliable agent without trusting second-best.
    
    Example:
        chosen = route_decision(score_a, score_b, record)
        # "B" (better quality and lower contamination)
    """
    qa, qb = score_a["quality_score"], score_b["quality_score"]
    ca, cb = score_a["contamination"], score_b["contamination"]

    # Quality threshold
    if qa > 0.5 and qb <= 0.5:
        return "A"
    if qb > 0.5 and qa <= 0.5:
        return "B"

    # Both positive quality: use lower contamination
    if qa > 0 and qb > 0:
        if ca < cb:
            return "A"
        if cb < ca:
            return "B"

    # Tie-break by hop type
    return "A" if record.get("hop_type") == "single" else "B"


def get_routing_rationale(score_a, score_b, a_out, b_out, chosen, record):
    """
    Generate human-readable explanation of routing decision.
    
    Returns:
        str: One sentence explaining why the chosen agent was trusted
    
    Example:
        "Agent A quality (0.65) > threshold, B quality (0.30) did not — "
        "A cited [hasChild, spouse] with low contamination (0.1)"
    """
    qa, qb = score_a["quality_score"], score_b["quality_score"]
    ca, cb = score_a["contamination"], score_b["contamination"]
    a_rels = score_a.get("agent_relations", [])
    b_rels = score_b.get("agent_relations", [])
    b_path = b_out.get("path_found", "none")

    if chosen == "A":
        if qa > 0.5 and qb <= 0.5:
            return (f"A quality ({qa:.2f}) > threshold, B quality ({qb:.2f}) did not — "
                    f"A cited {a_rels} with contam={ca:.2f}")
        if ca < cb:
            return (f"A contam ({ca:.2f}) < B ({cb:.2f}) — "
                    f"A cited {score_a.get('overlap_tail',[])} vs B noise {score_b.get('overlap_pred',[])}")
        return f"Equal quality, single-hop favours A — cited {a_rels}"
    else:
        if qb > 0.5 and qa <= 0.5:
            return (f"B quality ({qb:.2f}) > threshold, A quality ({qa:.2f}) did not — "
                    f"B identified {score_b.get('overlap_tail',[])}")
        if cb < ca:
            return f"B contam ({cb:.2f}) < A ({ca:.2f}) — B path {b_path} cited {b_rels}"
        path_str = f"path {b_path} found" if b_path not in ["none", "null", None, ""] else "no path"
        return (f"Equal quality, multi-hop — B {path_str}, "
                f"cited {len(b_rels)} rels vs A {len(a_rels)} rels")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

AGGREGATOR_SYSTEM = """You are documenting a Knowledge Graph routing decision.
The routing decision has already been made. Your job:

1. Write one sentence explaining WHY the chosen agent was trusted.
   Use the deciding signal.
2. Classify the failure type based on the pattern observed.

Output ONLY valid JSON:
{
  "reason": "<one specific, informative sentence>",
  "failure_type": "similarity_confusion | type_fit_gap | structural_gap | both_failed | resolved"
}"""


AGGREGATOR_USER = """Triple: ({head}, {relation}, ?)
Hop type: {hop_type}
Chosen agent: {chosen}
Deciding signal: {deciding_signal}

Agent A — prediction: {a_pred}  confidence: {a_conf}
  shared_relations: {a_relations}
  failure_diagnosis: {a_diagnosis}

Agent B — prediction: {b_pred}  confidence: {b_conf}
  key_relations: {b_relations}
  path_found: {b_path}  path_relation_matches_query: {b_rel_match}
  reasoning: {b_reasoning}
  failure_diagnosis: {b_diagnosis}

Write ONE sentence explaining why Agent {chosen} was trusted.
Focus on the deciding signal. Do not argue for the other agent."""


def aggregate_results(
    a_out: Dict,
    b_out: Dict,
    score_a: Dict,
    score_b: Dict,
    record: Dict,
    llm_backend = None,
    use_llm_aggregation: bool = False,
) -> Dict:
    """
    Aggregate Agent A and B outputs into final answer.
    
    Args:
        a_out, b_out: Agent outputs
        score_a, score_b: Quality scores
        record: Original record
        llm_backend: Optional LLM (if use_llm_aggregation=True)
        use_llm_aggregation: Call LLM to document decision?
    
    Returns:
        dict with:
        - "final_answer": Predicted entity
        - "chosen_agent": "A" or "B"
        - "confidence": Chosen agent's confidence
        - "reason": Why this agent was trusted
        - "selected_relations": Relations cited
        - "failure_type": Classification
    
    Implementation:
        1. Route decision
        2. Use chosen agent's prediction
        3. If use_llm_aggregation: Call LLM to classify failure type
        4. Otherwise: Use heuristic classification
    """
    chosen = route_decision(score_a, score_b, record)
    chosen_out = a_out if chosen == "A" else b_out
    chosen_score = score_a if chosen == "A" else score_b

    final_answer = chosen_out.get("prediction", "")
    rationale = get_routing_rationale(score_a, score_b, a_out, b_out, chosen, record)

    print(f"  [Route] Agent {chosen}  quality={chosen_score['quality_score']}  "
          f"contam={chosen_score['contamination']}")
    print(f"  [Route] {rationale}")

    if use_llm_aggregation and llm_backend:
        print("  [Agg] LLM labelling...")
        user = AGGREGATOR_USER.format(
            head=record["head"],
            relation=record["relation"],
            hop_type=record.get("hop_type", "multi"),
            chosen=chosen,
            deciding_signal=rationale,
            a_pred=a_out.get("prediction", "unknown"),
            a_conf=a_out.get("confidence", 0.0),
            a_relations=a_out.get("shared_relations", []),
            a_diagnosis=a_out.get("failure_diagnosis", "none"),
            b_pred=b_out.get("prediction", "unknown"),
            b_conf=b_out.get("confidence", 0.0),
            b_relations=b_out.get("key_relations", []),
            b_path=b_out.get("path_found", "none"),
            b_rel_match=b_out.get("path_relation_matches_query", False),
            b_reasoning=b_out.get("reasoning", "none"),
            b_diagnosis=b_out.get("failure_diagnosis", "none"),
        )
        raw = llm_backend.call(AGGREGATOR_SYSTEM, user)
        llm_out = json.loads(raw.strip())
        failure_type = llm_out.get("failure_type", "resolved")
    else:
        # Heuristic classification
        only_tail = set(record.get("only_tail_has", []))
        only_pred = set(record.get("only_pred_has", []))
        
        if len(only_tail) == 0 and len(only_pred) == 0:
            failure_type = "similarity_confusion"
        elif score_a["quality_score"] < 0.3 and score_b["quality_score"] < 0.3:
            failure_type = "both_failed"
        else:
            failure_type = "resolved"

    result = {
        "final_answer": final_answer,
        "chosen_agent": chosen,
        "confidence": chosen_out.get("confidence", 0.0),
        "reason": rationale,
        "selected_relations": chosen_score.get("overlap_tail", []),
        "failure_type": failure_type,
    }

    print(f"  [Agg] final={result['final_answer']}  agent={chosen}  type={failure_type}")

    return result
