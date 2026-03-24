import json
import numpy as np

# ── load correctly depending on file structure ────────────
def load_results_safe(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)

    # checkpoint format: {"head|rel|tail": {result dict}, ...}
    if isinstance(raw, dict):
        records = list(raw.values())
    # results format: [{result dict}, ...]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(f"Unexpected format in {path}")

    # filter errors and string entries
    clean = [
        r for r in records
        if isinstance(r, dict)
        and "error" not in r
        and "score_a" in r
        and "score_b" in r
    ]
    print(f"Loaded {len(clean)} clean records from {path}")
    return clean


results = load_results_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\src\Agentic_Memory\val_hard_results.json")

# ── if that's empty try the checkpoint ───────────────────
if not results:
    print("Trying checkpoint file...")
    results = load_results_safe("held_out_checkpoint.json")

# ── verify first record looks right ──────────────────────
if results:
    r0 = results[0]
    print(f"\nFirst record keys: {list(r0.keys())}")
    print(f"score_a keys:      {list(r0['score_a'].keys())}")
    print(f"score_a sample:    quality={r0['score_a'].get('quality_score')}  "
          f"contam={r0['score_a'].get('contamination')}")
else:
    print("No clean records found — check file path and format")

def quality_report(results, label):
    a_q = [r["score_a"]["quality_score"]          for r in results]
    b_q = [r["score_b"]["quality_score"]          for r in results]
    a_c = [r["score_a"]["contamination"]           for r in results]
    b_c = [r["score_b"]["contamination"]           for r in results]
    a_n = [len(r["score_a"]["agent_relations"])    for r in results]
    b_n = [len(r["score_b"]["agent_relations"])    for r in results]

    print(f"\n{'='*52}")
    print(f"{label}")
    print(f"{'='*52}")
    print(f"{'Metric':<32} {'Agent A':>8} {'Agent B':>8}")
    print(f"{'-'*52}")
    print(f"{'quality_score mean':<32} {np.mean(a_q):>8.3f} {np.mean(b_q):>8.3f}")
    print(f"{'quality_score median':<32} {np.median(a_q):>8.3f} {np.median(b_q):>8.3f}")
    print(f"{'contamination mean':<32} {np.mean(a_c):>8.3f} {np.mean(b_c):>8.3f}")
    print(f"{'contamination median':<32} {np.median(a_c):>8.3f} {np.median(b_c):>8.3f}")
    print(f"{'relations cited mean':<32} {np.mean(a_n):>8.1f} {np.mean(b_n):>8.1f}")
    print(f"{'quality == 1.0':<32} {sum(q==1.0 for q in a_q):>8} {sum(q==1.0 for q in b_q):>8}")
    print(f"{'quality == 0.0':<32} {sum(q==0.0 for q in a_q):>8} {sum(q==0.0 for q in b_q):>8}")
    print(f"{'contamination == 0.0':<32} {sum(c==0.0 for c in a_c):>8} {sum(c==0.0 for c in b_c):>8}")
    print(f"{'contamination > 0.5':<32} {sum(c>0.5 for c in a_c):>8} {sum(c>0.5 for c in b_c):>8}")

    b_beats_a = sum(1 for r in results
                    if r["score_b"]["quality_score"] > r["score_a"]["quality_score"])
    a_beats_b = sum(1 for r in results
                    if r["score_a"]["quality_score"] > r["score_b"]["quality_score"])
    tied      = len(results) - b_beats_a - a_beats_b

    print(f"\n{'B quality > A quality':<32} {b_beats_a:>8}")
    print(f"{'A quality > B quality':<32} {a_beats_b:>8}")
    print(f"{'Tied':<32} {tied:>8}")

    # worst contamination cases — who is dumping relations
    print(f"\nTop 5 highest contamination (Agent A):")
    worst_a = sorted(results, key=lambda r: -r["score_a"]["contamination"])[:5]
    for r in worst_a:
        sa = r["score_a"]
        print(f"  {r['triple']}")
        print(f"    cited={len(sa['agent_relations'])}  "
              f"contam={sa['contamination']:.2f}  "
              f"quality={sa['quality_score']:.2f}")

    print(f"\nTop 5 highest contamination (Agent B):")
    worst_b = sorted(results, key=lambda r: -r["score_b"]["contamination"])[:5]
    for r in worst_b:
        sb = r["score_b"]
        print(f"  {r['triple']}")
        print(f"    cited={len(sb['agent_relations'])}  "
              f"contam={sb['contamination']:.2f}  "
              f"quality={sb['quality_score']:.2f}")

quality_report(results, "BASELINE — held_out reasoning quality")

# verify the hypothesis before XGBoost
print("only_tail_has counts for quality=0 cases:")
zero_quality = [r for r in results if r["score_a"]["quality_score"] == 0.0]
for r in zero_quality:
    tail_count = len(r["score_a"]["only_tail_has"])
    print(f"  {r['triple']:<45} only_tail_has={tail_count}")