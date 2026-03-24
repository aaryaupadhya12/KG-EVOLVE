import json, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HOP_MAP = {"single": 0, "multi": 1, "none": 2}

# ─────────────────────────────────────────────
# FEATURE EXTRACTOR (UPGRADED)
# ─────────────────────────────────────────────
def extract_lower_r(record: dict):

    if "score_a" in record:
        sa = record["score_a"]

        only_tail = sa.get("only_tail_has", [])
        only_pred = sa.get("overlap_pred", [])
        shared    = []

        hop_type  = record.get("aggregator", {}).get("failure_type", "multi")
        hop_count = 1

        true_rank  = int(record.get("model_rank", 13))
        score_true = 0.0
        score_pred = 0.0

    else:
        only_tail = record.get("only_tail_has", [])
        only_pred = record.get("only_pred_has", [])
        shared    = record.get("shared_relations", [])

        hop_type  = record.get("hop_type", "multi")
        hop_count = float(record.get("hop_count", 1))

        true_rank  = int(record.get("true_rank", 13))
        score_true = float(record.get("score_true", 0.0))
        score_pred = float(record.get("score_predicted", 0.0))

    # ── base counts ───────────────────────────
    only_tail_count = len(only_tail)
    only_pred_count = len(only_pred)
    shared_count    = len(shared)

    total = only_tail_count + only_pred_count + shared_count + 1

    # ── derived features (CRITICAL) ───────────
    signal_ratio = only_tail_count / max(only_tail_count + only_pred_count, 1)

    return {
        # original
        "true_rank":       true_rank,
        "score_gap":       score_pred - score_true,
        "only_tail_count": only_tail_count,
        "only_pred_count": only_pred_count,
        "shared_count":    shared_count,
        "signal_exists":   int(only_tail_count > 0),
        "signal_ratio":    signal_ratio,
        "hop_type":        HOP_MAP.get(str(hop_type), 1),
        "hop_count":       hop_count,

        # ── NEW FEATURES ───────────────────────
        "pred_density": only_pred_count / total,
        "tail_density": only_tail_count / total,

        "imbalance": only_tail_count - only_pred_count,
        "abs_imbalance": abs(only_tail_count - only_pred_count),

        "log_pred": np.log1p(only_pred_count),
        "log_tail": np.log1p(only_tail_count),

        "rank_x_signal": true_rank * only_tail_count,
        "gap_x_pred": (score_pred - score_true) * only_pred_count,

        "noise_flag": int(only_pred_count > only_tail_count),
        "clean_flag": int(only_tail_count > only_pred_count),

        "confidence_proxy": only_tail_count / (only_pred_count + 1),
    }


# ─────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────
def get_label_routing(record: dict):
    return (
        record["aggregator"]
        .get("chosen_agent", "A")
        .replace("Agent", "")
        .strip()
    )

def get_label_tier(record: dict):
    if "score_a" in record:
        sa = record["score_a"]
        if len(sa.get("only_tail_has", [])) == 0:
            return 0
        if len(sa.get("overlap_tail", [])) == 0:
            return 1
        if sa.get("contamination", 0) > 0:
            return 2
        return 3
    else:
        only_tail = record.get("only_tail_has", [])
        if len(only_tail) == 0:
            return 0
        return 1 if record.get("hard_failure", True) else 3


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_safe(path):
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        return []
    with open(path) as f:
        raw = json.load(f)
    records = list(raw.values()) if isinstance(raw, dict) else raw
    return [r for r in records if isinstance(r, dict)]


val_results  = load_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\src\Agentic_Memory\val_hard_results.json")
held_results = load_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\src\Agentic_Memory\held_out_results.json")

print(f"Val records:  {len(val_results)}")
print(f"Held records: {len(held_results)}")

# ─────────────────────────────────────────────
# BUILD MATRICES
# ─────────────────────────────────────────────
X_val  = pd.DataFrame([extract_lower_r(r) for r in val_results])
X_held = pd.DataFrame([extract_lower_r(r) for r in held_results])

# stronger ablation (remove dominant signals)
DROP_COLS = ["signal_ratio", "only_pred_count"]
X_val_abl  = X_val.drop(columns=DROP_COLS)
X_held_abl = X_held.drop(columns=DROP_COLS)

le = LabelEncoder()
y_val_route = le.fit_transform([get_label_routing(r) for r in val_results])

y_val_tier  = np.array([get_label_tier(r) for r in val_results])
y_held_tier = np.array([get_label_tier(r) for r in held_results])

print("\nFeature shapes:")
print("X_val :", X_val.shape)
print("X_held:", X_held.shape)

# ─────────────────────────────────────────────
# MODEL — ROUTER
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

cv_full = cross_val_score(model, X_val, y_val_route, cv=cv, scoring="accuracy")
cv_abl  = cross_val_score(model, X_val_abl, y_val_route, cv=cv, scoring="accuracy")

model.fit(X_val, y_val_route)

print(f"\nRouter FULL: {cv_full.mean():.3f}")
print(f"Router ABL : {cv_abl.mean():.3f}")
print(f"Drop       : {cv_full.mean() - cv_abl.mean():.3f}")

# ─────────────────────────────────────────────
# STRONG BASELINE (IMPORTANT)
# ─────────────────────────────────────────────
rule = (
    (X_val["only_tail_count"] > X_val["only_pred_count"]) &
    (X_val["true_rank"] <= 5)
).astype(int)

rule_acc = np.mean(rule == y_val_route)
model_acc = np.mean(model.predict(X_val) == y_val_route)

print("\nStrong baseline vs model:")
print(f"Rule:  {rule_acc:.3f}")
print(f"Model: {model_acc:.3f}")

# ─────────────────────────────────────────────
# TIER MODEL (TRANSFER)
# ─────────────────────────────────────────────
tier_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="mlogloss",
    random_state=42,
)

tier_model.fit(X_val_abl, y_val_tier)
tier_pred = tier_model.predict(X_held_abl)
tier_acc  = np.mean(tier_pred == y_held_tier)

print(f"\nTier held-out accuracy: {tier_acc:.3f}")

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importance = pd.Series(
    model.feature_importances_,
    index=X_val.columns
).sort_values(ascending=False)

print("\nTop features:")
for k, v in importance.items():
    print(f"{k:<20} {v:.3f}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
model.save_model("xgb_router.json")
tier_model.save_model("xgb_tier.json")

print("\nSaved models: xgb_router.json, xgb_tier.json")
