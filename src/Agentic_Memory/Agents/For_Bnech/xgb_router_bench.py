import json, os, csv
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from collections import Counter

HOP_MAP = {"single": 0, "multi": 1, "none": 2}

def extract_honest(record: dict):
    sa = record.get("score_a", {})
    sb = record.get("score_b", {})

    a_conf  = float(record.get("agent_a", {}).get("confidence", 0.0))
    b_conf  = float(record.get("agent_b", {}).get("confidence", 0.0))
    a_cited = len(sa.get("agent_relations", []))
    b_cited = len(sb.get("agent_relations", []))

    b_path     = int(bool(
        record.get("agent_b", {}).get("path_found") and
        record.get("agent_b", {}).get("path_found") != "none"
    ))
    b_rel_match = int(bool(
        record.get("agent_b", {}).get("path_relation_matches_query", False)
    ))
    true_rank = int(record.get("model_rank", record.get("true_rank", 13)))
    hop_raw   = record.get("hop_type",
                record.get("aggregator", {}).get("failure_type", "multi"))
    hop_type  = HOP_MAP.get(str(hop_raw), 1)

    return {
        "a_conf":       a_conf,
        "b_conf":       b_conf,
        "conf_delta":   a_conf - b_conf,
        "a_cited":      a_cited,
        "b_cited":      b_cited,
        "cited_delta":  a_cited - b_cited,
        "b_path_found": b_path,
        "b_rel_match":  b_rel_match,
        "b_hub_risk":   int(b_conf > 0.5 and not b_rel_match),
        "true_rank":    true_rank,
        "hop_type":     hop_type,
    }

def get_label_routing(record):
    raw = (
        record.get("aggregator", {})
              .get("chosen_agent", "B")
    )
    # handle "Agent A", "Agent B", "A", "B"
    raw = str(raw).replace("Agent", "").strip()
    if raw not in ["A", "B"]:
        raw = "B"  # default
    return raw

def get_label_correct(record):
    return int(record.get("final_correct", False))

def load_safe(path):
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        return []
    with open(path) as f:
        raw = json.load(f)
    records = list(raw.values()) if isinstance(raw, dict) else raw
    return [
        r for r in records
        if isinstance(r, dict)
        and "aggregator" in r
        and "score_a" in r
        and "score_b" in r
    ]

# ── LOAD ─────────────────────────────────────
val_results  = load_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\Bench\with_writeback\val_hard_results (11).json")
held_results = load_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\Bench\with_writeback\held_out_results (4).json")

print(f"Val records:  {len(val_results)}")
print(f"Held records: {len(held_results)}")

# ── DEBUG: check what chosen_agent actually looks like ──
print("\nSample chosen_agent values (val):")
for r in val_results[:5]:
    raw = r.get("aggregator", {}).get("chosen_agent", "MISSING")
    print(f"  raw='{raw}'  parsed='{get_label_routing(r)}'")

X_val  = pd.DataFrame([extract_honest(r) for r in val_results])
X_held = pd.DataFrame([extract_honest(r) for r in held_results])

print(f"\nFeature shapes:")
print(f"X_val : {X_val.shape}")
print(f"X_held: {X_held.shape}")

# ── LABELS ───────────────────────────────────
y_val_route    = np.array([get_label_routing(r) for r in val_results])
y_held_route   = np.array([get_label_routing(r) for r in held_results])
y_val_correct  = np.array([get_label_correct(r) for r in val_results])
y_held_correct = np.array([get_label_correct(r) for r in held_results])

le = LabelEncoder()
all_labels = list(y_val_route) + list(y_held_route)
le.fit(all_labels)
y_val_route_enc  = le.transform(y_val_route)
y_held_route_enc = le.transform(y_held_route)

print(f"\nRouting label distribution (val):")
for cls, count in zip(*np.unique(y_val_route, return_counts=True)):
    print(f"  {cls}: {count}")

print(f"\nRouting label distribution (held):")
for cls, count in zip(*np.unique(y_held_route, return_counts=True)):
    print(f"  {cls}: {count}")

print(f"\nCorrectness distribution (val):")
for v, c in zip(*np.unique(y_val_correct, return_counts=True)):
    print(f"  {v}: {c}")

print(f"\nCorrectness distribution (held):")
for v, c in zip(*np.unique(y_held_correct, return_counts=True)):
    print(f"  {v}: {c}")

# ── ROUTER ───────────────────────────────────
n_classes = len(np.unique(y_val_route_enc))

if n_classes < 2:
    print(f"\nOnly one routing class in val ({le.classes_}) — XGBoost skipped")
    print(f"This means all val records were routed to the same agent.")
    print(f"Check chosen_agent field above.")
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    router = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    cv_scores = cross_val_score(
        router, X_val, y_val_route_enc,
        cv=cv, scoring="accuracy"
    )
    print(f"\nRouter CV: {cv_scores.round(3)}  mean={cv_scores.mean():.3f}")
    router.fit(X_val, y_val_route_enc)

    held_acc = router.score(X_held, y_held_route_enc)
    naive    = np.mean(y_held_route_enc == np.bincount(y_held_route_enc).argmax())

    print(f"\n{'='*45}")
    print(f"HONEST ROUTER — held-out")
    print(f"{'='*45}")
    print(f"Naive majority:    {naive:.3f}")
    print(f"XGBoost held-out:  {held_acc:.3f}")
    print(f"Improvement:       {held_acc - naive:+.3f}")

    importance = pd.Series(
        router.feature_importances_,
        index=X_val.columns
    ).sort_values(ascending=False)
    print(f"\nTop features (routing):")
    for feat, imp in importance.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<20} {imp:.3f}  {bar}")

    router.save_model("xgb_router_honest.json")
    print("\nSaved: xgb_router_honest.json")

    y_pred = router.predict(X_held)
    print(classification_report(
        y_held_route_enc, y_pred,
        target_names=le.classes_
    ))

# ── CORRECTNESS MODEL ────────────────────────
if len(np.unique(y_val_correct)) > 1:
    corrector = xgb.XGBClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, eval_metric="logloss",
        random_state=42,
    )
    corrector.fit(X_val, y_val_correct)
    held_corr = corrector.score(X_held, y_held_correct)
    print(f"\nCorrectness held-out: {held_corr:.3f}")
    corrector.save_model("xgb_correct_honest.json")
else:
    print("\nAll val cases correct — correctness model skipped")

# ── END-TO-END ───────────────────────────────
end_to_end_correct = 0
for record in held_results:
    true_tail    = record.get("true_tail")
    chosen_agent = get_label_routing(record)
    if chosen_agent == "A":
        prediction = record["agent_a"].get("prediction")
    else:
        prediction = record["agent_b"].get("prediction")
    if prediction == true_tail:
        end_to_end_correct += 1

e2e = end_to_end_correct / len(held_results)
print(f"\nEnd-to-end answer accuracy: {e2e:.3f}")

both_agree = sum(
    1 for r in held_results
    if r["agent_a"].get("prediction") == r["agent_b"].get("prediction")
)
quality_ones = sum(
    1 for r in held_results
    if r["score_b"].get("quality_score", 0) == 1.0
)
always_b = sum(
    1 for r in held_results
    if get_label_routing(r) == "B"
)
print(f"Both agents agree: {both_agree}/{len(held_results)} ({both_agree/len(held_results):.1%})")
print(f"Perfect quality scores: {quality_ones}/{len(held_results)}")
print(f"Aggregator picks B: {always_b}/{len(held_results)}")

# ── HYPOTHESIS VALIDATION ────────────────────
print(f"\n{'='*45}")
print(f"HYPOTHESIS VALIDATION")
print(f"{'='*45}")

correct = sum(1 for r in held_results if r.get("final_correct"))
print(f"\n[1] Hits@1 (hard held set)")
print(f"    {correct}/{len(held_results)} = {correct/len(held_results):.3f}")
print(f"    RotatE baseline: 0.000 (all rank >= 4 by construction)")

qa = [r["score_a"].get("quality_score", 0) or 0 for r in val_results]
qb = [r["score_b"].get("quality_score", 0) or 0 for r in val_results]
a_gt_b    = sum(1 for a, b in zip(qa, qb) if a > b)
b_gt_a    = sum(1 for a, b in zip(qa, qb) if b > a)
both_zero = sum(1 for a, b in zip(qa, qb) if a == 0 and b == 0)
both_pos  = sum(1 for a, b in zip(qa, qb) if a > 0 and b > 0)

print(f"\n[2] Quality score distribution (val, n={len(val_results)})")
print(f"    Agent A mean: {sum(qa)/len(qa):.3f}  zeros: {sum(1 for q in qa if q==0)}")
print(f"    Agent B mean: {sum(qb)/len(qb):.3f}  zeros: {sum(1 for q in qb if q==0)}")
print(f"    A > B: {a_gt_b}  |  B > A: {b_gt_a}  |  tied-zero: {both_zero}  |  both>0: {both_pos}")
print(f"    Scorer discriminates: {'YES' if (a_gt_b+b_gt_a)>0 else 'NO'}")

lucky_a = sum(1 for r in val_results
              if r["score_a"].get("prediction_correct")
              and (r["score_a"].get("quality_score") or 0) == 0.0)
lucky_b = sum(1 for r in val_results
              if r["score_b"].get("prediction_correct")
              and (r["score_b"].get("quality_score") or 0) == 0.0)
print(f"\n[3] Lucky predictions (correct answer, quality=0.0)")
print(f"    Agent A lucky: {lucky_a}/{len(val_results)} = {lucky_a/len(val_results):.1%}")
print(f"    Agent B lucky: {lucky_b}/{len(val_results)} = {lucky_b/len(val_results):.1%}")

ftypes = Counter(r["aggregator"].get("failure_type") for r in held_results)
mislabelled = sum(1 for r in held_results
                  if not r.get("final_correct")
                  and r["aggregator"].get("failure_type") == "resolved")
print(f"\n[4] Failure type distribution (held)")
for ft, count in ftypes.most_common():
    print(f"    {ft}: {count}")
print(f"    Mislabelled (wrong but resolved): {mislabelled}")

print(f"\n[5] Memory ablation")
tsv_path = "episodic_memory.tsv"
if not os.path.exists(tsv_path):
    print(f"    episodic_memory.tsv not found")
else:
    memory_heads = set()
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            memory_heads.add(row["head"])
    seen, unseen = [], []
    for r in held_results:
        triple = r.get("triple", "")
        head = triple.replace("(","").split(",")[0].strip()
        (seen if head in memory_heads else unseen).append(r)
    acc_seen   = sum(1 for r in seen   if r.get("final_correct")) / max(len(seen), 1)
    acc_unseen = sum(1 for r in unseen if r.get("final_correct")) / max(len(unseen), 1)
    print(f"    Memory entities: {len(memory_heads)}")
    print(f"    Seen in memory:   {len(seen)}  Hits@1={acc_seen:.3f}")
    print(f"    Unseen in memory: {len(unseen)}  Hits@1={acc_unseen:.3f}")
    print(f"    Memory delta: {acc_seen - acc_unseen:+.3f}")

print(f"\n{'='*45}")
print(f"SUMMARY FOR PAPER")
print(f"{'='*45}")
print(f"  Recovery:    {correct/len(held_results):.1%} Hits@1 (RotatE=0%)")
print(f"  Scorer:      {'discriminates' if (a_gt_b+b_gt_a)>0 else 'DEGENERATE'}")
print(f"  Lucky rate:  A={lucky_a/len(val_results):.1%}  B={lucky_b/len(val_results):.1%}")
print(f"  Agreement:   {both_agree/len(held_results):.1%}")
print(f"  Mislabelled: {mislabelled}")