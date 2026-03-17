# !pip install pykeen -q

from pykeen.datasets import Nations
dataset = Nations()
print(dataset.summary_str())

"""
The output is that we habe 14 entities and 55 relations a total of 1992 triples in the Nations dataset
"""

import pandas as pd
triples = dataset.training.mapped_triples
factory = dataset.training

df = pd.DataFrame(
    factory.triples,
    columns = ["head" ,"relation","tail"]

)

print(df.head())

print(df["relation"].nunique)
print(pd.concat([df["head"], df["tail"]]).unique())

from pykeen.pipeline import pipeline

result = pipeline(dataset ="Nations",
                 model = "TransH",
                 training_kwargs = dict(num_epochs = 50),
                 random_seed = 42)

import torch
import torch.nn.functional as F

model = result.model
entity_embeddings = model.entity_representations[0](indices =  None).detach()
entity_labels = dataset.training.entity_id_to_label

target = "india"
target_id = dataset.training.entity_to_id[target]
target_vec = entity_embeddings[target_id]

simillarity = F.cosine_similarity(target_vec.unsqueeze(0),entity_embeddings)
top5 = torch.topk(simillarity , k = 10)

print(f"Nearest neighbors to '{target}':")
for idx, score in zip(top5.indices.tolist(), top5.values.tolist()):
    label = entity_labels[idx]
    print(f"  {label:20s}  sim={score:.4f}")

# pick two countries you expect to be similar
a = "usa"
b = "uk"
c = "china"

def get_vec(name):
    idx = dataset.training.entity_to_id[name]
    return entity_embeddings[idx]

sim_ab = F.cosine_similarity(get_vec(a).unsqueeze(0), get_vec(b).unsqueeze(0))
sim_ac = F.cosine_similarity(get_vec(a).unsqueeze(0), get_vec(c).unsqueeze(0))

print(f"sim({a}, {b}) = {sim_ab.item():.4f}")
print(f"sim({a}, {c}) = {sim_ac.item():.4f}")