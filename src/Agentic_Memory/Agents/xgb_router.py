import json, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

HOP_MAP = {"single": 0, "multi": 1, "none": 2}

CLEAN_FEATURES = [
    "a_conf", "b_conf", "conf_delta",
    "a_cited", "b_cited", "cited_delta",
    "b_path_found", "b_rel_match", "b_hub_risk",
    "true_rank", "hop_type",
]

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
    return (
        record.get("aggregator", {})
              .get("chosen_agent", "A")
              .replace("Agent", "")
              .strip()
    )

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

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
val_results  = load_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\src\Agentic_Memory\Agents\json\Val_Hard\val_hard_results (6).json")
held_results = load_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\src\Agentic_Memory\Agents\json\Held_out\held_out_results (1).json")

print(f"Val records:  {len(val_results)}")
print(f"Held records: {len(held_results)}")

X_val  = pd.DataFrame([extract_honest(r) for r in val_results])
X_held = pd.DataFrame([extract_honest(r) for r in held_results])

print(f"\nFeature shapes:")
print(f"X_val : {X_val.shape}")
print(f"X_held: {X_held.shape}")

le = LabelEncoder()
y_val_route  = le.fit_transform(
    [get_label_routing(r) for r in val_results]
)
y_held_route = le.transform(
    [get_label_routing(r) for r in held_results]
)
y_val_correct  = np.array([get_label_correct(r) for r in val_results])
y_held_correct = np.array([get_label_correct(r) for r in held_results])

print(f"\nRouting label distribution (val):")
vals, counts = np.unique(y_val_route, return_counts=True)
for v, c in zip(le.classes_, counts):
    print(f"  {v}: {c}")

print(f"\nCorrectness distribution (val):")
for v, c in zip(*np.unique(y_val_correct, return_counts=True)):
    print(f"  {v}: {c}")

print(f"\nCorrectness distribution (held):")
for v, c in zip(*np.unique(y_held_correct, return_counts=True)):
    print(f"  {v}: {c}")

# ─────────────────────────────────────────────
# ROUTER — train here before importance
# ─────────────────────────────────────────────
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
    router, X_val, y_val_route,
    cv=cv, scoring="accuracy"
)

print(f"\nRouter CV: {cv_scores.round(3)}  "
      f"mean={cv_scores.mean():.3f}")

router.fit(X_val, y_val_route)

held_acc = router.score(X_held, y_held_route)
majority = np.bincount(y_held_route).argmax()
naive    = np.mean(y_held_route == majority)

print(f"\n{'='*45}")
print(f"HONEST ROUTER — held-out")
print(f"{'='*45}")
print(f"Naive majority:    {naive:.3f}")
print(f"XGBoost held-out:  {held_acc:.3f}")
print(f"Improvement:       {held_acc - naive:+.3f}")

# ─────────────────────────────────────────────
# CORRECTNESS MODEL
# ─────────────────────────────────────────────
if len(np.unique(y_val_correct)) > 1:
    corrector = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    corrector.fit(X_val, y_val_correct)
    held_corr = corrector.score(X_held, y_held_correct)
    print(f"\nCorrectness held-out: {held_corr:.3f}")
    corrector.save_model("xgb_correct_honest.json")
else:
    print("\nAll val cases correct — correctness model skipped")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importance = pd.Series(
    router.feature_importances_,
    index=X_val.columns
).sort_values(ascending=False)

print(f"\nTop features (routing):")
for feat, imp in importance.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<20} {imp:.3f}  {bar}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
router.save_model("xgb_router_honest.json")
print("\nSaved: xgb_router_honest.json")

from sklearn.metrics import classification_report

y_pred = router.predict(X_held)
print(classification_report(
    y_held_route, y_pred,
    target_names=le.classes_
))

end_to_end_correct = 0
total = 0

for record in held_results:
    true_tail = record.get("true_tail")
    
    # Fix: wrap in a single-row DataFrame to match training format
    features = pd.DataFrame([extract_honest(record)])
    route = router.predict(features)[0]
    chosen_agent = le.inverse_transform([route])[0]
    
    if chosen_agent == "A":
        prediction = record["agent_a"].get("prediction")
    else:
        prediction = record["agent_b"].get("prediction")
    
    if prediction == true_tail:
        end_to_end_correct += 1
    total += 1

e2e_accuracy = end_to_end_correct / total
print(f"End-to-end answer accuracy: {e2e_accuracy:.3f}")
print(f"Routing accuracy:           0.833")
print(f"Gap:                        {0.833 - e2e_accuracy:+.3f}")


# Run this on your held-out data
both_agree = sum(
    1 for r in held_results
    if r["agent_a"].get("prediction") == r["agent_b"].get("prediction")
)
print(f"Both agents agree: {both_agree}/{len(held_results)} "
      f"({both_agree/len(held_results):.1%})")

quality_ones = sum(
    1 for r in held_results
    if r["score_b"].get("quality_score", 0) == 1.0
)
print(f"Perfect quality scores: {quality_ones}/{len(held_results)}")

always_b = sum(
    1 for r in held_results
    if r["aggregator"].get("chosen_agent", "").replace("Agent","").strip() == "B"
)
print(f"Aggregator always picks B: {always_b}/{len(held_results)}")