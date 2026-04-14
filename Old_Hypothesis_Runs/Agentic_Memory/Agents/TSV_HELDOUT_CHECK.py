import csv
import json
from collections import defaultdict
import numpy as np

# ── LOAD TSV ─────────────────────────────────────────────
tsv_path = r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\Nations_minimal_Run\Without_hallucination\episodic_memory.tsv"

tsv_set = set()
tsv_hr_map = defaultdict(set)

with open(tsv_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        h = row["head"].strip()
        r = row["relation"].strip()
        t = row["tail"].strip()

        tsv_set.add((h, r, t))
        tsv_hr_map[(h, r)].add(t)

print(f"Loaded TSV triples: {len(tsv_set)}")
print(f"Unique (head, relation): {len(tsv_hr_map)}")


# ── LOAD HELD-OUT ────────────────────────────────────────
held_path = r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\src\Agentic_Memory\nations_held_out.json"

with open(held_path) as f:
    held_out_records = json.load(f)

print(f"Held-out records: {len(held_out_records)}")


# ── ANALYSIS ─────────────────────────────────────────────
exact_match = 0
hr_match = 0
unique_candidate = 0
candidate_sizes = []

for r in held_out_records:
    h = r["head"]
    rel = r["relation"]
    t = r["tail"]

    # exact triple match
    if (h, rel, t) in tsv_set:
        exact_match += 1

    # (head, relation) exists
    if (h, rel) in tsv_hr_map:
        hr_match += 1

        size = len(tsv_hr_map[(h, rel)])
        candidate_sizes.append(size)

        if size == 1:
            unique_candidate += 1
    else:
        candidate_sizes.append(0)


n = len(held_out_records)

print("\n===== OVERLAP ANALYSIS =====")
print(f"Exact triple overlap:      {exact_match}/{n} = {exact_match/n:.3f}")
print(f"(head, relation) overlap:  {hr_match}/{n} = {hr_match/n:.3f}")
print(f"Unique candidate cases:    {unique_candidate}/{n} = {unique_candidate/n:.3f}")

print("\n===== CANDIDATE STATS =====")
print(f"Avg candidates:    {np.mean(candidate_sizes):.2f}")
print(f"Median candidates: {np.median(candidate_sizes):.2f}")
print(f"Max candidates:    {np.max(candidate_sizes)}")
