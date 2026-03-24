import json

with open(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\dump\val_hard_checkpoint (3).json") as f:
    results = json.load(f)

# compute aggregator accuracy
total         = 0
correct       = 0
agent_a_wins  = 0
agent_b_wins  = 0
both_correct  = 0
neither       = 0
failure_types = {}

for key, record in results.items():
    if not isinstance(record, dict):
        continue
    if "aggregator" not in record:
        continue
    
    total += 1
    
    # did aggregator get it right?
    if record.get("final_correct"):
        correct += 1
    
    # which agent was chosen
    chosen = record["aggregator"].get("chosen_agent")
    if chosen == "A":
        agent_a_wins += 1
    elif chosen == "B":
        agent_b_wins += 1
    
    # both correct?
    a_right = record.get("agent_a", {}).get(
        "prediction") == record.get("true_tail")
    b_right = record.get("agent_b", {}).get(
        "prediction") == record.get("true_tail")
    
    if a_right and b_right:
        both_correct += 1
    elif not a_right and not b_right:
        neither += 1
    
    # failure type
    ft = record["aggregator"].get("failure_type", "unknown")
    failure_types[ft] = failure_types.get(ft, 0) + 1

print(f"{'='*45}")
print(f"AGGREGATOR RESULTS — YOUR REAL METRIC")
print(f"{'='*45}")
print(f"Total cases:          {total}")
print(f"Aggregator correct:   {correct} / {total} "
      f"({100*correct/total:.1f}%)")
print(f"")
print(f"Agent A chosen:       {agent_a_wins}")
print(f"Agent B chosen:       {agent_b_wins}")
print(f"Both correct:         {both_correct}")
print(f"Neither correct:      {neither}")
print(f"")
print(f"Failure type breakdown:")
for ft, count in sorted(
    failure_types.items(), key=lambda x: -x[1]
):
    print(f"  {ft:<20} {count}")

# now compare: aggregator vs single agent baseline
print(f"\n{'='*45}")
print(f"BASELINE COMPARISON")
print(f"{'='*45}")

a_correct = 0
b_correct = 0

for key, record in results.items():
    if not isinstance(record, dict):
        continue
    true_tail = record.get("true_tail")
    if record.get("agent_a", {}).get("prediction") == true_tail:
        a_correct += 1
    if record.get("agent_b", {}).get("prediction") == true_tail:
        b_correct += 1

print(f"Agent A alone:        {a_correct}/{total} "
      f"({100*a_correct/total:.1f}%)")
print(f"Agent B alone:        {b_correct}/{total} "
      f"({100*b_correct/total:.1f}%)")
print(f"Aggregator:           {correct}/{total} "
      f"({100*correct/total:.1f}%)")
print(f"")
print(f"Aggregator vs best single agent: "
      f"{100*correct/total - max(100*a_correct/total, 100*b_correct/total):+.1f}%")