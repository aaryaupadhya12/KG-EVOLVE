import json
from pathlib import Path

# ── load your results files ───────────────────────────────

def load_results(path: str) -> list:
    """
    Handles both your training results format
    (val_hard_results.json) and inference format
    (held_out_results.json).
    
    Both have the same schema so this works for either.
    """
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records from {path}")
    return data


def extract_fields(record: dict) -> dict:
    """
    Pulls every field needed for pass@k from one record.
    Works with both your training and inference JSON schemas.
    
    From your actual JSON structure:
    - record["true_tail"]           → ground truth
    - record["agent_a"]["prediction"] → Agent A answer
    - record["agent_b"]["prediction"] → Agent B answer  
    - record["score_a"]["quality_score"] → A quality
    - record["score_b"]["quality_score"] → B quality
    - record["aggregator"]["final_answer"] → routed answer
    - record["aggregator"]["failure_type"] → resolved/etc
    - record["final_correct"]       → did routing work?
    - record["model_rank"]          → RotatE rank
    """
    true_tail = record.get("true_tail", "").strip().lower()
    
    # agent predictions
    a_pred = (
        record.get("agent_a", {})
               .get("prediction", "") or ""
    ).strip().lower()
    
    b_pred = (
        record.get("agent_b", {})
               .get("prediction", "") or ""
    ).strip().lower()
    
    # final routed answer
    final = (
        record.get("aggregator", {})
               .get("final_answer", "") or ""
    ).strip().lower()
    
    # quality scores — handle None explicitly
    qa = record.get("score_a", {}).get("quality_score")
    qb = record.get("score_b", {}).get("quality_score")
    qa = float(qa) if qa is not None else 0.0
    qb = float(qb) if qb is not None else 0.0
    
    # correctness flags
    a_correct   = (a_pred == true_tail) and (a_pred != "")
    b_correct   = (b_pred == true_tail) and (b_pred != "")
    final_correct = (final == true_tail) and (final != "")
    
    # grounded = correct AND quality > 0
    a_grounded  = a_correct and (qa > 0)
    b_grounded  = b_correct and (qb > 0)
    
    # lucky = correct AND quality == 0
    a_lucky     = a_correct and (qa == 0)
    b_lucky     = b_correct and (qb == 0)
    
    # path verification from your schema
    b_path_verified = (
        record.get("score_b", {}).get("path_verified") 
        is True
    )
    
    # failure type
    failure_type = (
        record.get("aggregator", {})
               .get("failure_type", "unknown")
    )
    
    return {
        "triple":          record.get("triple", ""),
        "true_tail":       true_tail,
        "model_rank":      record.get("model_rank", 0),
        "a_correct":       a_correct,
        "b_correct":       b_correct,
        "final_correct":   final_correct,
        "a_grounded":      a_grounded,
        "b_grounded":      b_grounded,
        "a_lucky":         a_lucky,
        "b_lucky":         b_lucky,
        "qa":              qa,
        "qb":              qb,
        "b_path_verified": b_path_verified,
        "failure_type":    failure_type,
        "chosen_agent":    record.get(
            "aggregator", {}
        ).get("chosen_agent", "?"),
    }


# ── core pass@k computations ─────────────────────────────

def compute_pass_at_k(records: list, split_name: str) -> dict:
    """
    Computes the full pass@k suite for your paper.
    
    Reads your JSON records directly and computes:
    
    pass@1_routed   = Hits@1 after routing (your current metric)
    pass@2_total    = at least one agent correct (capability ceiling)
    pass@2_grounded = at least one agent correct AND quality > 0
    pass@2_lucky    = correct but only via ungrounded prediction
    
    routing_efficiency = how close routing gets to the ceiling
    routing_loss       = cases where answer existed but routing missed
    grounded_ceiling   = max achievable with grounded routing only
    """
    fields = [extract_fields(r) for r in records]
    n = len(fields)
    
    if n == 0:
        print(f"[{split_name}] No records found")
        return {}
    
    # ── pass@1: routed answer ─────────────────────────────
    pass1_routed = sum(
        1 for f in fields if f["final_correct"]
    ) / n

    # ── pass@2 total: did EITHER agent get it? ────────────
    pass2_total = sum(
        1 for f in fields 
        if f["a_correct"] or f["b_correct"]
    ) / n

    # ── pass@2 grounded: correct + quality > 0 ───────────
    pass2_grounded = sum(
        1 for f in fields
        if f["a_grounded"] or f["b_grounded"]
    ) / n

    # ── pass@2 lucky: correct only via quality=0 ─────────
    # these cases are in pass2_total but NOT pass2_grounded
    pass2_lucky = sum(
        1 for f in fields
        if (f["a_correct"] or f["b_correct"])
        and not (f["a_grounded"] or f["b_grounded"])
    ) / n

    # ── both correct simultaneously ───────────────────────
    both_correct = sum(
        1 for f in fields
        if f["a_correct"] and f["b_correct"]
    ) / n

    # ── routing gap analysis ──────────────────────────────
    # routing_loss: answer existed in pool but routing missed
    routing_loss = pass2_total - pass1_routed
    
    # routing_efficiency: how close to ceiling?
    routing_efficiency = (
        pass1_routed / pass2_total 
        if pass2_total > 0 else 0.0
    )

    # ── per-agent lucky rates ─────────────────────────────
    a_correct_total = sum(1 for f in fields if f["a_correct"])
    b_correct_total = sum(1 for f in fields if f["b_correct"])
    
    a_lucky_rate = (
        sum(1 for f in fields if f["a_lucky"]) 
        / a_correct_total
        if a_correct_total > 0 else 0.0
    )
    b_lucky_rate = (
        sum(1 for f in fields if f["b_lucky"]) 
        / b_correct_total
        if b_correct_total > 0 else 0.0
    )

    # ── only one agent correct (disagreement cases) ───────
    # these are the cases where routing actually matters
    only_a_correct = sum(
        1 for f in fields
        if f["a_correct"] and not f["b_correct"]
    )
    only_b_correct = sum(
        1 for f in fields
        if f["b_correct"] and not f["a_correct"]
    )
    neither_correct = sum(
        1 for f in fields
        if not f["a_correct"] and not f["b_correct"]
    )
    agreement = sum(
        1 for f in fields
        if (f["a_correct"] and f["b_correct"])
        or (not f["a_correct"] and not f["b_correct"])
    )

    # ── path verification contribution ───────────────────
    # cases where B's path was verified and B was correct
    b_verified_correct = sum(
        1 for f in fields
        if f["b_correct"] and f["b_path_verified"]
    )

    results = {
        "split":               split_name,
        "n":                   n,

        # headline metrics
        "pass@1_routed":       round(pass1_routed,    4),
        "pass@2_total":        round(pass2_total,     4),
        "pass@2_grounded":     round(pass2_grounded,  4),
        "pass@2_lucky":        round(pass2_lucky,     4),
        "both_correct":        round(both_correct,    4),

        # routing analysis
        "routing_loss":        round(routing_loss,    4),
        "routing_efficiency":  round(routing_efficiency, 4),
        "grounded_ceiling":    round(pass2_grounded,  4),

        # disagreement breakdown
        "only_A_correct":      only_a_correct,
        "only_B_correct":      only_b_correct,
        "neither_correct":     neither_correct,
        "agreement_rate":      round(agreement / n,  4),

        # lucky rate per agent
        "A_lucky_rate":        round(a_lucky_rate,   4),
        "B_lucky_rate":        round(b_lucky_rate,   4),

        # path verification
        "B_verified_correct":  b_verified_correct,
    }

    return results


def print_pass_at_k_report(results: dict):
    """
    Prints the paper-ready summary from compute_pass_at_k.
    """
    if not results:
        return

    print(f"\n{'='*55}")
    print(f"PASS@K ANALYSIS — {results['split']} (n={results['n']})")
    print(f"{'='*55}")

    print(f"\n── Capability vs Reliability ──────────────────────")
    print(f"  pass@1 (routed Hits@1):     {results['pass@1_routed']:.4f}")
    print(f"  pass@2 total (ceiling):     {results['pass@2_total']:.4f}")
    print(f"  pass@2 grounded (ceiling):  {results['pass@2_grounded']:.4f}")
    print(f"  pass@2 lucky (ungrounded):  {results['pass@2_lucky']:.4f}")

    print(f"\n── Routing Analysis ───────────────────────────────")
    print(f"  Routing loss:               {results['routing_loss']:.4f}")
    print(f"  Routing efficiency:         {results['routing_efficiency']:.4f}")
    print(f"  Grounded ceiling:           {results['grounded_ceiling']:.4f}")

    print(f"\n── Agent Disagreement (where routing matters) ─────")
    print(f"  Only A correct:             {results['only_A_correct']}")
    print(f"  Only B correct:             {results['only_B_correct']}")
    print(f"  Neither correct:            {results['neither_correct']}")
    print(f"  Agreement rate:             {results['agreement_rate']:.4f}")

    print(f"\n── Lucky Rate per Agent ───────────────────────────")
    print(f"  Agent A lucky rate:         {results['A_lucky_rate']:.4f}")
    print(f"  Agent B lucky rate:         {results['B_lucky_rate']:.4f}")
    print(f"  B verified+correct:         {results['B_verified_correct']}")

    print(f"\n── Paper Interpretation ───────────────────────────")
    gap = results['pass@2_total'] - results['pass@1_routed']
    grounded_gap = results['pass@2_grounded'] - results['pass@1_routed']
    lucky_contamination = results['pass@2_lucky']

    print(f"  Routing improvement budget: {gap:.4f}")
    print(f"    (max gain from better routing over total pool)")
    print(f"  Grounded improvement budget: {grounded_gap:.4f}")
    print(f"    (max gain from grounded routing only)")
    print(f"  Lucky contamination:        {lucky_contamination:.4f}")
    print(f"    (fraction correct only via parametric recall)")

    if results['agreement_rate'] > 0.90:
        print(f"\n  ⚠ Agreement rate {results['agreement_rate']:.1%} > 90%")
        print(f"    Routing is degenerate on this dataset.")
        print(f"    CoDEx-S will show larger disagreement.")


# ── run on your actual files ──────────────────────────────

if __name__ == "__main__":

    # load both result files
    val_records  = load_results(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\Nations_minimal_Run\Without_hallucination\val_hard_results.json")
    held_records = load_results(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\Nations_minimal_Run\Without_hallucination\held_out_results (11).json")

    # compute pass@k for each split
    val_results    = compute_pass_at_k(val_records,  "Nations VAL")
    held_results   = compute_pass_at_k(held_records, "Nations HELD")

    # print reports
    print_pass_at_k_report(val_results)
    print_pass_at_k_report(held_results)

    # ── cross-split comparison for paper ─────────────────
    print(f"\n{'='*55}")
    print("CROSS-SPLIT COMPARISON")
    print(f"{'='*55}")
    print(f"{'Metric':<30} {'VAL':>10} {'HELD':>10}")
    print("-" * 52)

    metrics = [
        "pass@1_routed",
        "pass@2_total",
        "pass@2_grounded",
        "pass@2_lucky",
        "routing_efficiency",
        "routing_loss",
        "A_lucky_rate",
        "B_lucky_rate",
        "agreement_rate",
    ]

    for m in metrics:
        v = val_results.get(m, 0)
        h = held_results.get(m, 0)
        print(f"  {m:<28} {v:>10.4f} {h:>10.4f}")

    # ── save for paper ────────────────────────────────────
    combined = {
        "val":  val_results,
        "held": held_results,
    }
    with open("pass_at_k_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved: pass_at_k_results.json")