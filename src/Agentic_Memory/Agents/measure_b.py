# analyse Agent B's failures specifically

import json

# load from checkpoint instead
with open(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\dump\val_hard_checkpoint (3).json") as f:
    results = json.load(f)

print(f"Total records: {len(results)}")
print(f"Sample key: {list(results.keys())[0]}")
print(f"Sample record keys: {list(results[list(results.keys())[0]].keys())}")


b_failures = []
b_successes = []

for key, record in results.items():
    if "aggregator" not in record:
        continue
    
    true_tail = record.get("true_tail")
    b_pred    = record.get("agent_b",{}).get("prediction")
    b_path    = record.get("agent_b",{}).get("path_found")
    b_conf    = record.get("agent_b",{}).get("confidence",0)
    b_diag    = record.get("agent_b",{}).get("failure_diagnosis","")
    
    entry = {
        "triple":     key,
        "true_tail":  true_tail,
        "b_pred":     b_pred,
        "path_found": b_path,
        "confidence": b_conf,
        "diagnosis":  b_diag,
        "correct":    b_pred == true_tail
    }
    
    if b_pred == true_tail:
        b_successes.append(entry)
    else:
        b_failures.append(entry)

print(f"Agent B correct:   {len(b_successes)}")
print(f"Agent B wrong:     {len(b_failures)}")

print(f"\n=== AGENT B FAILURE PATTERNS ===")
for f in b_failures[:5]:
    print(f"\nTriple:     {f['triple']}")
    print(f"True tail:  {f['true_tail']}")
    print(f"B pred:     {f['b_pred']}")
    print(f"Path:       {f['path_found']}")
    print(f"Confidence: {f['confidence']}")
    print(f"Diagnosis:  {f['diagnosis'][:100]}")


usa_predictions = [
    f for f in b_failures 
    if f["b_pred"] == "usa"
]
other_failures = [
    f for f in b_failures 
    if f["b_pred"] != "usa"
]

print(f"Agent B wrong predictions:")
print(f"  Predicted 'usa' wrongly:  {len(usa_predictions)}")
print(f"  Predicted other wrongly:  {len(other_failures)}")
print(f"")
print(f"If usa bias fixed — Agent B accuracy would be:")
potential = (len(b_successes) + len(usa_predictions))
print(f"  {potential}/40 = {100*potential/42:.1f}%")
print(f"")
print(f"Other failure predictions:")
for f in other_failures[:5]:
    print(f"  true={f['true_tail']:<12} "
          f"pred={f['b_pred']:<12} "
          f"conf={f['confidence']:.2f}")