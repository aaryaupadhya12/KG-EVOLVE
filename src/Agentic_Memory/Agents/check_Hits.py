import json
import numpy as np

def load_safe(path):
    with open(path) as f:
        raw = json.load(f)
    records = list(raw.values()) if isinstance(raw, dict) else raw
    return [r for r in records if isinstance(r, dict) and "agent_a" in r and "agent_b" in r]

held_results = load_safe(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\dump\json\Held_out\held_out_results (1).json")

total = len(held_results)
a_correct, b_correct, either_correct, both_correct = 0, 0, 0, 0
rotate_beats_agents = 0  # cases where model_rank==1 but agents wrong

rank_improvements = []  # how much did agent push rank vs RotatE

# Check what model_rank actually looks like in your data
ranks = []
for r in held_results:
    rank = r.get("model_rank", r.get("true_rank", None))
    ranks.append(rank)

import collections
print("model_rank value distribution:")
print(dict(collections.Counter(ranks).most_common(20)))

print(f"\nMin: {min(r for r in ranks if r is not None)}")
print(f"Max: {max(r for r in ranks if r is not None)}")
print(f"Null/missing: {sum(1 for r in ranks if r is None)}")

# Also check one full record to see what fields are present
import json
print("\nSample record keys:", list(held_results[0].keys()))
print("Sample model_rank field:", held_results[0].get("model_rank"))
print("Sample true_rank field:", held_results[0].get("true_rank"))

for r in held_results:
    true_tail = r.get("true_tail")
    a_pred = r["agent_a"].get("prediction")
    b_pred = r["agent_b"].get("prediction")
    model_rank = int(r.get("model_rank", r.get("true_rank", 99)))

    a_hit = (a_pred == true_tail)
    b_hit = (b_pred == true_tail)

    a_correct    += a_hit
    b_correct    += b_hit
    either_correct += (a_hit or b_hit)
    both_correct   += (a_hit and b_hit)

    # RotatE got it (rank 1) but agents missed
    if model_rank == 1 and not a_hit and not b_hit:
        rotate_beats_agents += 1

    # Treat agent hit as rank=1, else keep model_rank
    best_agent_rank = 1 if (a_hit or b_hit) else model_rank
    rank_improvements.append(model_rank - best_agent_rank)

print(f"Total held triples:       {total}")
print(f"")
print(f"--- Hits@1 ---")
print(f"RotatE alone (rank==1):   {sum(1 for r in held_results if int(r.get('model_rank', r.get('true_rank',99)))==1)/total:.3f}")
print(f"Agent A:                  {a_correct/total:.3f}")
print(f"Agent B:                  {b_correct/total:.3f}")
print(f"Oracle (best of A or B):  {either_correct/total:.3f}")
print(f"Both agree and correct:   {both_correct/total:.3f}")
print(f"")
print(f"--- Where agents help ---")
print(f"RotatE rank>1 but agent correct: {either_correct - sum(1 for r in held_results if int(r.get('model_rank',r.get('true_rank',99)))==1 and (r['agent_a'].get('prediction')==r.get('true_tail') or r['agent_b'].get('prediction')==r.get('true_tail')))}/{total}")
print(f"RotatE rank==1 but agents wrong:  {rotate_beats_agents}/{total}")
print(f"")
print(f"--- Rank improvement (avg) ---")
print(f"Mean rank gain from best agent:  {np.mean(rank_improvements):.2f}")
print(f"Cases improved:  {sum(1 for x in rank_improvements if x > 0)}")
print(f"Cases hurt:      {sum(1 for x in rank_improvements if x < 0)}")
print(f"Cases neutral:   {sum(1 for x in rank_improvements if x == 0)}")