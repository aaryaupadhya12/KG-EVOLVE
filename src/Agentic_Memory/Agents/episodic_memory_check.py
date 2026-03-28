import pandas as pd
from collections import Counter

df = pd.read_csv(r"C:\Users\Aarya-2\Documents\ADOG\PESU\3rd Year --PESU\6th Sem\NLP\Agentic_AI\KG-schema-evolution-agents\KG-schema-evolution-agents\json\episodic_memory (10).tsv", sep="\t")

print(f"Total rows:        {len(df)}")
print(f"Unique entities:   {df['head'].nunique()}")
print(f"Relation types:    {df['relation'].value_counts().to_dict()}")

# show convergence — how many times each entity was corrected
corrections = df[df['relation'] == 'corrects_failure']
print(f"\nCorrections per entity:")
print(corrections['head'].value_counts())