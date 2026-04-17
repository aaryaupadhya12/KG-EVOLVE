import warnings
warnings.filterwarnings("ignore")

import os, json, csv, time, pickle, torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

from codex.codex import Codex

codex = Codex(size="s", code="en")

def _resolve(row):
    return (
        codex.entity_label(row["head"])       or row["head"],
        codex.relation_label(row["relation"])  or row["relation"],
        codex.entity_label(row["tail"])        or row["tail"],
    )

df_train = pd.DataFrame(
    [_resolve(r) for _, r in codex.split("train").iterrows()],
    columns=["head", "relation", "tail"]
)
df_test = pd.DataFrame(
    [_resolve(r) for _, r in codex.split("test").iterrows()],
    columns=["head", "relation", "tail"]
)
df_valid = pd.DataFrame(
    [_resolve(r) for _, r in codex.split("valid").iterrows()],
    columns=["head", "relation", "tail"]
)
df_all = pd.concat([df_train, df_test], ignore_index=True)

print(f"Train: {len(df_train)}  Valid: {len(df_valid)}  Test: {len(df_test)}")

# ── ID mappings built from the SAME English-label entity/relation lists ──
# These must match exactly what the pretrained model was trained with.
# score_hrt(tensor([[h_id, r_id, t_id]])) expects these integer IDs.
entities_raw  = sorted(codex.entities())
relations_raw = sorted(codex.relations())

entity_to_id   = {e: i for i, e in enumerate(entities_raw)}
relation_to_id = {r: i for i, r in enumerate(relations_raw)}
id_to_entity   = {i: e for e, i in entity_to_id.items()}
id_to_relation = {i: r for r, i in relation_to_id.items()}

print(f"Entities: {len(entity_to_id)}   Relations: {len(relation_to_id)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Load pretrained ComplEx from HuggingFace
# https://huggingface.co/aaryaupadhya20/codex-s-complex-winner
#
# score_hrt(tensor([[h_id, r_id, t_id]])) uses the integer IDs built above.
# No fallback training — only this model is used throughout.
# ─────────────────────────────────────────────────────────────────────────────

from huggingface_hub import hf_hub_download

MODEL_REPO = "aaryaupadhya20/codex-s-complex-winner"

# hf_hub_download caches locally — subsequent calls are instant.
# Adjust filename below if your repo uses a different name.
model_path = hf_hub_download(repo_id=MODEL_REPO, filename="trained_pipeline.pkl")

with open(model_path, "rb") as f:
    pipeline_result = pickle.load(f)

winner_model = pipeline_result.model
winner_model.eval()
print(f"Loaded: {type(winner_model).__name__}  "
      f"device={next(winner_model.parameters()).device}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Graph + helper functions
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_from_df(df):
    graph = defaultdict(list)
    for _, row in df.iterrows():
        graph[row["head"]].append((row["relation"], row["tail"]))
    return graph

graph = build_graph_from_df(df_train)
print(f"Graph nodes: {len(graph)}")


# ── Subgraph extraction ──────────────────────────────────────────────────────

def extract_subgraph(entity, graph, hops=2, max_triples=150):
    subgraph = []
    visited  = {entity}
    queue    = deque([(entity, 0)])
    while queue and len(subgraph) < max_triples:
        node, depth = queue.popleft()
        if depth >= hops:
            continue
        for rel, tail in graph.get(node, []):
            if len(subgraph) >= max_triples:
                break
            subgraph.append((node, rel, tail))
            if tail not in visited:
                visited.add(tail)
                queue.append((tail, depth + 1))
    return subgraph


def focused_subgraph(entities, graph, hops=2, max_triples=100, query_relation=None):
    entity_set  = set(entities)
    all_triples = []
    seen        = set()
    for ent in entities:
        for triple in extract_subgraph(ent, graph, hops, max_triples):
            if triple not in seen:
                seen.add(triple)
                all_triples.append(triple)
    if len(all_triples) <= max_triples:
        return all_triples
    tier1, tier2, tier3, tier4 = [], [], [], []
    for h, r, t in all_triples:
        if query_relation and r == query_relation:
            tier1.append((h, r, t))
        elif h in entity_set and t in entity_set:
            tier2.append((h, r, t))
        elif h in entity_set or t in entity_set:
            tier3.append((h, r, t))
        else:
            tier4.append((h, r, t))
    return (tier1 + tier2 + tier3 + tier4)[:max_triples]


def hop_classifier(head, tail, graph, target_relation=None):
    for relation, t in graph.get(head, []):
        if t == tail:
            if target_relation is None or relation == target_relation:
                return "single", 1, relation
    direct_wrong = [r for r, t in graph.get(head, []) if t == tail]
    if direct_wrong:
        return "multi", 1.5, f"direct but via {direct_wrong[0]}"
    paths_found = []
    for r1, mid in graph.get(head, []):
        for r2, t2 in graph.get(mid, []):
            if t2 == tail:
                paths_found.append(f"{head}-{r1}->{mid}-{r2}->{tail}")
    if paths_found:
        return "multi", 2, paths_found[0]
    return "none", 99, []


# ── Embedding similarity — uses winner_model exclusively ────────────────────

EMBS_CACHE = None   # populated once before preprocessing (Section 4)

def get_entity_embeddings():
    """Extract entity embeddings from winner_model."""
    embs = winner_model.entity_representations[0](indices=None).detach().cpu()
    if embs.is_complex():
        embs = embs.abs()
    return embs


def get_entity_relations(entity, df):
    head_rels = set(df[df["head"] == entity]["relation"])
    tail_rels = set(df[df["tail"] == entity]["relation"])
    return head_rels | tail_rels


def similarity_summary(entity, k=5):
    """Uses EMBS_CACHE (from winner_model). Must call after cache is set."""
    e_id  = entity_to_id[entity]
    e_vec = EMBS_CACHE[e_id]
    sims  = F.cosine_similarity(e_vec.unsqueeze(0), EMBS_CACHE).detach().cpu().numpy()
    ranked = np.argsort(sims)[::-1]
    entity_rels = get_entity_relations(entity, df_train)
    results = []
    for idx in ranked:
        name = id_to_entity[idx]
        if name == entity:
            continue
        score = sims[idx]
        neighbor_rels = get_entity_relations(name, df_train)
        shared    = len(entity_rels & neighbor_rels)
        total     = len(entity_rels | neighbor_rels)
        rel_score = shared / total if total > 0 else 0.0
        results.append((name, score, shared, rel_score))
        if len(results) == k:
            break
    parts   = [f"{n}(sim={s:.2f},shared={sh},rel={rs:.2f})"
               for n, s, sh, rs in results]
    summary = f"{entity} most similar to: {', '.join(parts)}"
    return summary, results


# ── Ranking — uses winner_model.score_hrt exclusively ───────────────────────

def get_full_ranking_filtered_batched(head, relation, true_tail, batch_size=512):
    """
    Score all candidate tail entities using winner_model.score_hrt.
    Returns ranked list + rank of true_tail.
    """
    h_id    = entity_to_id[head]
    r_id    = relation_to_id[relation]
    all_ids = [i for i in range(len(entity_to_id)) if id_to_entity[i] != head]
    all_scores = []
    for i in range(0, len(all_ids), batch_size):
        batch = all_ids[i:i + batch_size]
        hrs   = torch.tensor([[h_id, r_id, t_id] for t_id in batch])
        with torch.no_grad():
            scores = winner_model.score_hrt(hrs).squeeze(-1).cpu().tolist()
        all_scores.extend(zip([id_to_entity[j] for j in batch], scores))
    all_scores.sort(key=lambda x: -x[1])
    ranked_entities = [e for e, s in all_scores]
    true_rank       = ranked_entities.index(true_tail) + 1
    return {
        "predicted":       all_scores[0][0],
        "predicted_score": all_scores[0][1],
        "true_tail":       true_tail,
        "true_score":      next(s for e, s in all_scores if e == true_tail),
        "true_rank":       true_rank,
        "full_ranking":    [(e, round(s, 3)) for e, s in all_scores],
    }


# ── Type constraints ─────────────────────────────────────────────────────────

def build_type_constraints(df):
    rel_to_tail_counts = defaultdict(lambda: defaultdict(int))
    rel_to_head_counts = defaultdict(lambda: defaultdict(int))
    rel_to_pair_counts = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        h, r, t = row["head"], row["relation"], row["tail"]
        rel_to_tail_counts[r][t] += 1
        rel_to_head_counts[r][h] += 1
        rel_to_pair_counts[r][(h, t)] += 1
    rel_head_to_ranked_tails = defaultdict(dict)
    for r, pair_counts in rel_to_pair_counts.items():
        head_to_tails = defaultdict(dict)
        for (h, t), count in pair_counts.items():
            head_to_tails[h][t] = count
        for h, tail_counts in head_to_tails.items():
            total = sum(tail_counts.values())
            rel_head_to_ranked_tails[(r, h)] = {
                t: round(c / total, 4)
                for t, c in sorted(tail_counts.items(), key=lambda x: -x[1])
            }
    rel_to_tail_dist = {}
    for r, tail_counts in rel_to_tail_counts.items():
        total = sum(tail_counts.values())
        rel_to_tail_dist[r] = {
            t: round(c / total, 4)
            for t, c in sorted(tail_counts.items(), key=lambda x: -x[1])
        }
    return {
        "rel_head_to_ranked_tails": dict(rel_head_to_ranked_tails),
        "rel_to_tail_dist":         rel_to_tail_dist,
        "rel_to_tail_counts":       dict(rel_to_tail_counts),
        "rel_to_head_counts":       dict(rel_to_head_counts),
    }


def get_type_constraint_signal(head, relation, true_tail, predicted, constraints):
    rh_tails       = constraints["rel_head_to_ranked_tails"]
    tail_dist      = constraints["rel_to_tail_dist"]
    specific       = rh_tails.get((relation, head), {})
    general        = tail_dist.get(relation, {})
    type_fit_true  = specific.get(true_tail) or general.get(true_tail, 0.0)
    type_fit_pred  = specific.get(predicted) or general.get(predicted, 0.0)
    expected_tails = list(specific.keys())[:5] or list(general.keys())[:5]
    tail_counts    = constraints["rel_to_tail_counts"]
    true_rels = {r for r, tails in tail_counts.items() if true_tail in tails}
    pred_rels = {r for r, tails in tail_counts.items() if predicted  in tails}
    return {
        "type_fit_true":  type_fit_true,
        "type_fit_pred":  type_fit_pred,
        "type_gap":       round(type_fit_true - type_fit_pred, 4),
        "expected_tails": expected_tails,
        "only_true_has":  sorted(true_rels - pred_rels),
        "only_pred_has":  sorted(pred_rels - true_rels),
        "shared_rels":    sorted(true_rels & pred_rels),
    }


def failure_summary(head, relation, true_tail, predicted_tail, constraints):
    """Uses winner_model.score_hrt to get raw scores for true vs predicted."""
    h_id = entity_to_id[head]
    r_id = relation_to_id[relation]
    t_id = entity_to_id[true_tail]
    p_id = entity_to_id[predicted_tail]
    with torch.no_grad():
        score_true = winner_model.score_hrt(torch.tensor([[h_id, r_id, t_id]])).item()
        score_pred = winner_model.score_hrt(torch.tensor([[h_id, r_id, p_id]])).item()
    sig = get_type_constraint_signal(head, relation, true_tail, predicted_tail, constraints)
    summary = (
        f"Model predicted '{predicted_tail}' (score={score_pred:.3f}) "
        f"over '{true_tail}' (score={score_true:.3f}). "
        f"Type fit: '{true_tail}'={sig['type_fit_true']:.3f} vs "
        f"'{predicted_tail}'={sig['type_fit_pred']:.3f} for '{relation}'. "
        f"Expected tails: {', '.join(sig['expected_tails'][:3])}. "
        f"Only '{true_tail}' in: {', '.join(sig['only_true_has'][:5]) or 'none'}. "
        f"Only '{predicted_tail}' in: {', '.join(sig['only_pred_has'][:5]) or 'none'}."
    )
    return summary, {
        "score_true":     score_true,
        "score_pred":     score_pred,
        "shared":         set(sig["shared_rels"]),
        "only_true":      set(sig["only_true_has"]),
        "only_pred":      set(sig["only_pred_has"]),
        "type_fit_true":  sig["type_fit_true"],
        "type_fit_pred":  sig["type_fit_pred"],
        "type_gap":       sig["type_gap"],
        "expected_tails": sig["expected_tails"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Preprocessing  →  save JSON once, reload on Kaggle
#
# On Kaggle: upload the two JSON files as a dataset input.
# Then the `if os.path.exists` block below will load them instantly
# and skip all the heavy scoring — no model calls needed for preprocessing.
# ─────────────────────────────────────────────────────────────────────────────

PREPROCESSED_TRAIN = "CODEX_S_preprocessed_train.json"
PREPROCESSED_TEST  = "CODEX_S_preprocessed_test.json"


def preprocess_all_triples(df, split_name):
    records        = []
    n_entities     = len(entity_to_id)
    hard_threshold = max(10, int(n_entities * 0.15))
    constraints    = build_type_constraints(df_train)
    total          = len(df)

    for i, row in df.iterrows():
        try:
            head     = row["head"]
            relation = row["relation"]
            tail     = row["tail"]

            if relation not in relation_to_id: continue
            if head     not in entity_to_id:   continue
            if tail     not in entity_to_id:   continue

            ranking = get_full_ranking_filtered_batched(head, relation, tail)

            sim_head, _ = similarity_summary(head, k=5)
            sim_tail, _ = similarity_summary(tail, k=5)

            subgraph = focused_subgraph(
                [head, tail, ranking["predicted"]],
                graph,
                hops=3,
                max_triples=150,
                query_relation=relation,
            )

            hop_type, hops, _ = hop_classifier(head, tail, graph, target_relation=relation)

            fail_sum, fail_raw = failure_summary(
                head, relation, tail, ranking["predicted"], constraints
            )

            records.append({
                "split":            split_name,
                "head":             head,
                "relation":         relation,
                "tail":             tail,
                "true_rank":        int(ranking["true_rank"]),
                "predicted":        ranking["predicted"],
                "score_true":       float(ranking["true_score"]),
                "score_predicted":  float(ranking["predicted_score"]),
                "top5":             ranking["full_ranking"][:5],
                "hop_type":         hop_type,
                "hop_count":        int(hops),
                "sim_head":         sim_head,
                "sim_tail":         sim_tail,
                "fail_summary":     fail_sum,
                "subgraph":         subgraph,
                "shared_relations": list(fail_raw["shared"]),
                "only_tail_has":    list(fail_raw["only_true"]),
                "only_pred_has":    list(fail_raw["only_pred"]),
                "hard_failure":     bool(ranking["true_rank"] > hard_threshold),
            })

            if (i + 1) % 50 == 0:
                print(f"  {split_name}: {i+1}/{total}")

        except Exception as e:
            print(f"[ERROR] row {i}: {e}")
            continue

    return records


# ── Run once, then reload ────────────────────────────────────────────────────

if os.path.exists(PREPROCESSED_TRAIN) and os.path.exists(PREPROCESSED_TEST):
    print("Preprocessed files found — loading from disk (no model calls needed).")
    with open(PREPROCESSED_TRAIN) as f:
        train_records = json.load(f)
    with open(PREPROCESSED_TEST) as f:
        test_records = json.load(f)
else:
    print("First run: preprocessing all triples with winner_model …")
    EMBS_CACHE = get_entity_embeddings()      # cache once — reused for all similarity calls
    print("Preprocessing train …")
    train_records = preprocess_all_triples(df_train, "train")
    print("Preprocessing test …")
    test_records  = preprocess_all_triples(df_test,  "test")
    with open(PREPROCESSED_TRAIN, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(PREPROCESSED_TEST, "w") as f:
        json.dump(test_records, f, indent=2)
    print(f"Saved → {PREPROCESSED_TRAIN}  {PREPROCESSED_TEST}")
    print("Upload these two files to Kaggle as a dataset to skip preprocessing on future runs.")

# Ensure EMBS_CACHE is always set (used if similarity called at run-time)
if EMBS_CACHE is None:
    EMBS_CACHE = get_entity_embeddings()

print(f"Train records: {len(train_records)}   Test records: {len(test_records)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Val / held-out split
# ─────────────────────────────────────────────────────────────────────────────

VAL_FILE      = "CODEX_S_val.json"
HELD_OUT_FILE = "CODEX_S_held_out.json"

if os.path.exists(VAL_FILE) and os.path.exists(HELD_OUT_FILE):
    with open(VAL_FILE)      as f: val_records      = json.load(f)
    with open(HELD_OUT_FILE) as f: held_out_records = json.load(f)
    print(f"Loaded val={len(val_records)}  held_out={len(held_out_records)}")
else:
    val_records, held_out_records = train_test_split(
        test_records, test_size=0.5, random_state=42
    )
    with open(VAL_FILE,      "w") as f: json.dump(val_records,      f, indent=2)
    with open(HELD_OUT_FILE, "w") as f: json.dump(held_out_records, f, indent=2)
    print(f"Split → val={len(val_records)}  held_out={len(held_out_records)}  (held-out SEALED)")

val_hard = [r for r in val_records if r["hard_failure"]]
print(f"Val hard failures: {len(val_hard)}")

# Type constraints used at agent run-time
constraints = build_type_constraints(df_train)
print(f"Type constraints: {len(constraints['rel_to_tail_dist'])} relations")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Memory stores
# ─────────────────────────────────────────────────────────────────────────────

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings   import HuggingFaceEmbeddings

EMBED         = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
EPISODIC_PATH = "episodic.faiss"
SEMANTIC_PATH = "semantic.faiss"
EPISODIC_TSV  = "episodic_memory.tsv"


def _load_or_create_faiss(path):
    if os.path.exists(path):
        return FAISS.load_local(path, EMBED, allow_dangerous_deserialization=True)
    store = FAISS.from_texts(["init"], EMBED)
    store.save_local(path)
    return store


episodic_vs = _load_or_create_faiss(EPISODIC_PATH)
semantic_vs = _load_or_create_faiss(SEMANTIC_PATH)


def query_episodic(head, relation):
    docs = episodic_vs.similarity_search(f"{head} {relation}", k=2)
    return "\n".join(d.page_content for d in docs if "init" not in d.page_content)


def query_semantic(failure_type):
    docs = semantic_vs.similarity_search(failure_type, k=2)
    return "\n".join(d.page_content for d in docs if "init" not in d.page_content)


def write_episodic(head, relation, tail, agent, finding):
    text = f"{head} {relation} → {tail} | agent {agent} | {finding}"
    episodic_vs.add_texts([text])
    episodic_vs.save_local(EPISODIC_PATH)


def write_semantic(agent, failure_type, key_rels):
    text = f"agent {agent} wins on {failure_type} | key_rels: {', '.join(key_rels)}"
    semantic_vs.add_texts([text])
    semantic_vs.save_local(SEMANTIC_PATH)


def load_tsv_memory(path=EPISODIC_TSV):
    memory = defaultdict(list)
    if not os.path.exists(path):
        return memory
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            memory[row["head"].strip()].append(
                (row["relation"].strip(), row["tail"].strip())
            )
    return memory


def get_memory_hint(head, memory, relation=None, k=5):
    triples = memory.get(head, [])
    if not triples:
        return ""
    if relation:
        same = [(r, t) for r, t in triples if r == relation]
        rest = [(r, t) for r, t in triples if r != relation]
        selected = same[:k] + rest[:max(0, k - len(same))]
    else:
        selected = triples[:k]
    seen, unique = set(), []
    for r, t in selected:
        if (r, t) not in seen:
            seen.add((r, t)); unique.append((r, t))
    return "\n".join(f"{head} --{r}--> {t}" for r, t in unique)


tsv_memory = load_tsv_memory()
print(f"TSV memory: {len(tsv_memory)} entities  "
      f"{sum(len(v) for v in tsv_memory.values())} triples")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — LLM caller + Agent A / B prompts  (Qwen 4-bit)
# ─────────────────────────────────────────────────────────────────────────────

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline as hf_pipeline,
)

AGENT_A_B_DELAY = 1    # seconds stagger between A and B threads
BETWEEN_RECORDS = 1    # seconds between pipeline records

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {MODEL_ID} in 4-bit …")

# On Kaggle with secrets:
# from kaggle_secrets import UserSecretsClient
# hf_token = UserSecretsClient().get_secret("HF_TOKEN")
# Add token= parameter to both from_pretrained calls below.

_tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
_mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ),
    device_map="auto",
    trust_remote_code=True,
)
_pipe = hf_pipeline(
    "text-generation",
    model=_mdl,
    tokenizer=_tok,
    do_sample=True,
    temperature=0.3,
    return_full_text=False,
)
print(f"LLM ready on {_mdl.device}")


def call_llm(system: str, user: str) -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    for attempt in range(2):
        try:
            return _pipe(messages)[0]["generated_text"].strip()
        except Exception as e:
            if attempt == 0:
                print(f"  [LLM] retry: {e}"); time.sleep(5)
            else:
                raise


def parse_json(raw: str, who: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [{who}] JSON parse failed: {e}\n  raw: {raw[:200]}")
        return {"error": str(e), "raw": raw}


# ── System prompts ────────────────────────────────────────────────────────────

_A_SYS = """You are Agent A, a knowledge graph type-constraint reasoning agent.

Your role: determine which candidate entity correctly fills the tail slot
of a CODEX-S triple (real-world entities, Wikidata-style relations).

Primary signal — TYPE FIT:
  How often does the candidate appear as tail of this relation in training?
  Higher type_fit = stronger candidate.

Secondary signal — RELATIONAL PROFILE:
  Candidates that share relations with the true tail are structurally similar.
  only_true_has = relations the true tail participates in that the predicted does NOT.

Output ONLY valid JSON:
{
  "prediction": "<entity or null>",
  "confidence": <0.0-1.0>,
  "shared_relations": ["<relations candidate shares with true tail>"],
  "failure_diagnosis": "<one sentence>",
  "evidence_type": "type_constraint | profile | mixed | none"
}"""


_B_SYS = """You are Agent B, a knowledge graph structural reasoning agent.

Your role: determine which candidate entity correctly fills the tail slot
of a CODEX-S triple by reasoning about type constraints, relational profiles,
and multi-hop subgraph paths.

Reasoning steps:
1. What does this relation expect as tail? (see expected_tails)
2. Which candidate best matches? (compare type_fit scores)
3. Use only_true_has as discriminating signal.
4. Check subgraph for multi-hop paths from head to candidate.

Hard constraints:
- Do NOT claim paths unless they appear in the subgraph.
- Output null + confidence 0.05 if no signal exists.

Output ONLY valid JSON:
{
  "prediction": "<entity or null>",
  "confidence": <0.0-1.0>,
  "key_relations": ["<discriminating relations cited>"],
  "path_found": "<path string or null>",
  "path_relation_matches_query": <true|false>,
  "reasoning": "<one sentence>",
  "failure_diagnosis": "<one sentence>",
  "evidence_type": "type_constraint | profile | mixed | none"
}"""


_USER_TMPL = """
{context}

<episodic_memory>
{episodic_hint}
</episodic_memory>

<reasoning>
STEP 1 — TYPE CONSTRAINT SIGNAL
  type_fit > 0.1 and type_rank <= 5 → strongly supported.
  type_fit = 0.0 → never appeared as tail of this relation — strong negative.

STEP 2 — only_true_has (PRIMARY DISCRIMINATING SIGNAL)
  Does any candidate participate in ALL or MOST of these relations?
  If yes → structurally closest to true tail regardless of type_fit.

STEP 3 — RELATIONAL PROFILE
  Compare candidate_appears_in to expected_tails profile.

STEP 4 — EPISODIC MEMORY
  Has this (head, relation) pair appeared before?
  If irrelevant: ignore. Do not confabulate.

STEP 5 — COMMIT
  type_fit match + only_true_has match → 0.80-0.95
  type_fit match alone                 → 0.60-0.80
  only_true_has match alone            → 0.50-0.70
  profile overlap only                 → 0.30-0.50
  no signal                            → null, 0.05
</reasoning>

Respond with valid JSON only:
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Context builder
# ─────────────────────────────────────────────────────────────────────────────

def trim_subgraph(subgraph, head, true_tail, predicted, only_tail_has, max_triples=12):
    oth_set = set(only_tail_has)
    tier1, tier2, tier3, tier4 = [], [], [], []
    for triple in subgraph:
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            h, r, t = triple
        elif isinstance(triple, dict):
            h = triple.get("head",""); r = triple.get("relation",""); t = triple.get("tail","")
        else:
            continue
        if r in oth_set:
            tier1.append(triple)
        elif t in (true_tail, predicted) or h in (true_tail, predicted):
            tier2.append(triple)
        elif h == head or t == head:
            tier3.append(triple)
        else:
            tier4.append(triple)
    return (tier1 + tier2 + tier3 + tier4)[:max_triples]


def build_subgraph_str(record, max_triples=12):
    true_tail = record.get("true_tail") or record.get("tail", "")
    trimmed   = trim_subgraph(
        record.get("subgraph", []), record["head"], true_tail,
        record.get("predicted",""), record.get("only_tail_has",[]), max_triples,
    )
    if not trimmed:
        return "  (no subgraph available)"
    only_set = set(record.get("only_tail_has", []))
    lines = []
    for triple in trimmed:
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            h, r, t = triple
        elif isinstance(triple, dict):
            h = triple.get("head","?"); r = triple.get("relation","?"); t = triple.get("tail","?")
        else:
            continue
        lines.append(f"  {h} --{r}--> {t}" + (" ◆" if r in only_set else ""))
    return "\n".join(lines)


def build_agent_context(record, tsv_memory=None):
    true_tail = record.get("true_tail") or record.get("tail", "")
    predicted = record.get("predicted", "unknown")
    only_tail = record.get("only_tail_has", [])
    only_pred = record.get("only_pred_has", [])

    memory_block = ""
    if tsv_memory:
        hint = get_memory_hint(record["head"], tsv_memory)
        if hint:
            memory_block = f"\nMemory prior for {record['head']}:\n{hint}\n"

    if only_tail:
        separator_block = (
            "⚠ SEPARATING RELATIONS — cite these or state you cannot:\n"
            + "\n".join(f"  ◆ {r}" for r in only_tail)
            + f"\nThese connect [{record['head']}] → [{true_tail}] "
              f"but NOT → [{predicted}].\n"
        )
    else:
        separator_block = (
            "⚠ NO SEPARATING RELATIONS in subgraph.\n"
            "Forbidden (connect head to WRONG entity — do not cite):\n"
            + ("\n".join(f"  ✗ {r}" for r in only_pred) or "  none")
            + "\nYou must find a multi-hop path OR output confidence 0.3.\n"
        )

    sim_head = record.get("sim_head", "")
    if sim_head:
        sim_head = ", ".join(sim_head.split(", ")[:3])

    return (
        f"Triple: ({record['head']}, {record['relation']}, ?)\n"
        f"Predicted wrong: {predicted}  |  rank of correct answer: {record['true_rank']}\n\n"
        f"{separator_block}\n"
        f"{memory_block}\n"
        f"Subgraph (◆ = gold signal triple — use these first):\n"
        f"{build_subgraph_str(record)}\n\n"
        f"Similarity (top-3 embedding neighbours of head):\n"
        f"  {sim_head}\n\n"
        f"Hop type: {record.get('hop_type','multi')}\n"
        f"Failure: {record.get('fail_summary','')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Grounded scorer + verifiers
# ─────────────────────────────────────────────────────────────────────────────

def verify_type_fit(agent_out, head, relation, constraints):
    key_rels = agent_out.get("key_relations", [])
    if not key_rels:
        return {"verified": True, "hallucinated": [], "confirmed": []}
    all_rels     = set(constraints["rel_to_tail_counts"].keys())
    confirmed    = [r for r in key_rels if r     in all_rels]
    hallucinated = [r for r in key_rels if r not in all_rels]
    if hallucinated:
        print(f"  [verify B] ✗ hallucinated: {hallucinated}")
        agent_out["key_relations"] = confirmed
    else:
        print(f"  [verify B] ✓ all relations confirmed")
    return {"verified": len(hallucinated) == 0,
            "confirmed": confirmed, "hallucinated": hallucinated}


def verify_relations(claimed_rels, head, tail, df_ref):
    if not claimed_rels:
        return {"verified": [], "hallucinated": [], "rate": 0.0}
    actual       = set(df_ref[(df_ref["head"] == head) & (df_ref["tail"] == tail)]["relation"])
    verified     = [r for r in claimed_rels if r     in actual]
    hallucinated = [r for r in claimed_rels if r not in actual]
    return {"verified": verified, "hallucinated": hallucinated,
            "rate": len(hallucinated) / max(len(claimed_rels), 1)}


def grounded_score(agent_out, record, who, df_ref=None, constraints=None):
    true_tail = record.get("true_tail") or record.get("tail", "")
    only_tail = set(record.get("only_tail_has", []))
    only_pred = set(record.get("only_pred_has", []))
    path_verified, path_verification, rel_verification = None, {}, {}

    if who == "B" and constraints is not None:
        pv = verify_type_fit(agent_out, record["head"], record["relation"], constraints)
        path_verified = pv["verified"]
        path_verification = pv
        if not pv["verified"]:
            agent_out["path_relation_matches_query"] = False
            agent_out["path_found"]    = "none"
            agent_out["key_relations"] = pv["confirmed"]

    if who == "A" and df_ref is not None:
        rv = verify_relations(
            agent_out.get("shared_relations", []), record["head"], true_tail, df_ref
        )
        rel_verification = rv
        if rv["hallucinated"]:
            print(f"  [verify A] ✗ hallucinated: {rv['hallucinated']}")
            agent_out["shared_relations"] = rv["verified"]
        if rv["verified"]:
            print(f"  [verify A] ✓ confirmed: {rv['verified']}")

    agent_rels     = set(agent_out.get("shared_relations",[]) if who=="A"
                         else agent_out.get("key_relations",[]))
    overlap_tail   = agent_rels & only_tail
    overlap_pred   = agent_rels & only_pred
    coverage_score = len(overlap_tail) / max(len(only_tail), 1)
    contamination  = len(overlap_pred) / max(len(agent_rels), 1)
    correct        = (agent_out.get("prediction","").strip().lower()
                      == true_tail.strip().lower())
    quality_score  = round(coverage_score * (1 - 0.5 * contamination), 3)

    print(f"  [score {who}] coverage={coverage_score:.2f}  "
          f"contam={contamination:.2f}  quality={quality_score}  correct={correct}")
    return {
        "agent":              who,
        "relation_score":     round(coverage_score, 3),
        "quality_score":      quality_score,
        "contamination":      round(contamination, 3),
        "overlap_tail":       sorted(overlap_tail),
        "overlap_pred":       sorted(overlap_pred),
        "agent_relations":    sorted(agent_rels),
        "only_tail_has":      sorted(only_tail),
        "prediction_correct": correct,
        "confidence":         agent_out.get("confidence", 0.0),
        "path_verified":      path_verified,
        "path_verification":  path_verification,
        "rel_verification":   rel_verification,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Aggregator + writeback
# ─────────────────────────────────────────────────────────────────────────────

def agent_a(context, record):
    print(f"  [A] ({record['head']}, {record['relation']}, {record.get('tail','')})")
    hint = query_episodic(record["head"], record["relation"])
    raw  = call_llm(_A_SYS, _USER_TMPL.format(context=context, episodic_hint=hint or "none"))
    out  = parse_json(raw, "A")
    print(f"  [A] pred={out.get('prediction')}  conf={out.get('confidence')}")
    return out


def agent_b(context, record):
    print(f"  [B] ({record['head']}, {record['relation']}, {record.get('tail','')})")
    hint = query_episodic(record["head"], record["relation"])
    raw  = call_llm(_B_SYS, _USER_TMPL.format(context=context, episodic_hint=hint or "none"))
    out  = parse_json(raw, "B")
    print(f"  [B] pred={out.get('prediction')}  conf={out.get('confidence')}")
    return out


def run_parallel_staggered(record, tsv_memory=None):
    context = build_agent_context(record, tsv_memory=tsv_memory)
    results = {}
    def _run_a(): results["A"] = agent_a(context, record)
    def _run_b(): time.sleep(AGENT_A_B_DELAY); results["B"] = agent_b(context, record)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(_run_a); fb = ex.submit(_run_b)
        fa.result(); fb.result()
    return results["A"], results["B"]


def _python_route(s_a, s_b, record):
    qa, qb = s_a["quality_score"], s_b["quality_score"]
    ca, cb = s_a["contamination"],  s_b["contamination"]
    if qa > 0.5 and qb <= 0.5: return "A"
    if qb > 0.5 and qa <= 0.5: return "B"
    if qa > 0 and qb > 0:
        if ca < cb: return "A"
        if cb < ca: return "B"
    return "A" if record["hop_type"] == "single" else "B"


def get_deciding_signal(s_a, s_b, a_out, b_out, chosen, record):
    qa, qb = s_a["quality_score"], s_b["quality_score"]
    ca, cb = s_a["contamination"],  s_b["contamination"]
    a_rels = s_a.get("agent_relations", [])
    b_rels = s_b.get("agent_relations", [])
    b_path = b_out.get("path_found", "none")
    if chosen == "A":
        if qa > 0.5 and qb <= 0.5:
            return (f"A quality ({qa:.2f}) > threshold, B quality ({qb:.2f}) did not — "
                    f"A cited {a_rels} contam={ca:.2f}")
        if ca < cb:
            return (f"A contam ({ca:.2f}) < B ({cb:.2f}) — "
                    f"A cited {s_a.get('overlap_tail',[])} vs B noise {s_b.get('overlap_pred',[])}")
        return f"Equal quality, single-hop favours A — cited {a_rels}"
    else:
        if qb > 0.5 and qa <= 0.5:
            return (f"B quality ({qb:.2f}) > threshold, A quality ({qa:.2f}) did not — "
                    f"B identified {s_b.get('overlap_tail',[])}")
        if cb < ca:
            return f"B contam ({cb:.2f}) < A ({ca:.2f}) — B path {b_path} cited {b_rels}"
        path_str = f"path {b_path} found" if b_path not in ["none","null",None,""] else "no path"
        return (f"Equal quality, multi-hop — B {path_str}, "
                f"cited {len(b_rels)} rels vs A {len(a_rels)} rels")


_AGG_SYS = """You are documenting a Knowledge Graph routing decision.
The routing decision has already been made by a deterministic scorer.

Your job:
1. Write one sentence explaining WHY the chosen agent was trusted.
   Use the deciding signal. Mention specific relations or scores.
2. Classify the failure type.

Output ONLY valid JSON:
{
  "reason": "<one specific, informative sentence>",
  "failure_type": "similarity_confusion | structural_gap | both_failed | resolved"
}"""

_AGG_USER = """Triple: ({head}, {relation}, ?)
Hop type: {hop_type}
Chosen agent: {chosen}
Deciding signal: {deciding_signal}

Agent A — prediction: {a_pred}  confidence: {a_conf}
  shared_relations: {a_relations}
  failure_diagnosis: {a_diagnosis}

Agent B — prediction: {b_pred}  confidence: {b_conf}
  key_relations: {b_relations}
  path_found: {b_path}  path_relation_matches_query: {b_rel_match}
  reasoning: {b_reasoning}
  failure_diagnosis: {b_diagnosis}

Write ONE sentence explaining why Agent {chosen} was trusted.
Do not argue for the other agent. Do not mention the correct answer."""


def aggregate(a_out, b_out, s_a, s_b, record):
    chosen       = _python_route(s_a, s_b, record)
    chosen_out   = a_out if chosen == "A" else b_out
    chosen_score = s_a   if chosen == "A" else s_b
    final_answer    = chosen_out.get("prediction", "")
    deciding_signal = get_deciding_signal(s_a, s_b, a_out, b_out, chosen, record)
    print(f"  [Route] agent {chosen}  quality={chosen_score['quality_score']}  "
          f"contam={chosen_score['contamination']}")
    print(f"  [Route] {deciding_signal}")
    print("  [Agg] LLM labelling …")
    user = _AGG_USER.format(
        head=record["head"], relation=record["relation"],
        hop_type=record["hop_type"], chosen=chosen,
        deciding_signal=deciding_signal,
        a_pred=a_out.get("prediction","unknown"), a_conf=a_out.get("confidence",0.0),
        a_relations=a_out.get("shared_relations",[]),
        a_diagnosis=a_out.get("failure_diagnosis","none"),
        b_pred=b_out.get("prediction","unknown"), b_conf=b_out.get("confidence",0.0),
        b_relations=b_out.get("key_relations",[]),
        b_path=b_out.get("path_found","none"),
        b_rel_match=b_out.get("path_relation_matches_query",False),
        b_reasoning=b_out.get("reasoning","none"),
        b_diagnosis=b_out.get("failure_diagnosis","none"),
    )
    llm_out = parse_json(call_llm(_AGG_SYS, user), "Agg")
    out = {
        "final_answer":       final_answer,
        "chosen_agent":       chosen,
        "confidence":         chosen_out.get("confidence", 0.0),
        "reason":             llm_out.get("reason", ""),
        "selected_relations": chosen_score.get("overlap_tail", []),
        "failure_type":       llm_out.get("failure_type", "resolved"),
    }
    print(f"  [Agg] final={out['final_answer']}  agent={out['chosen_agent']}  "
          f"type={out['failure_type']}")
    return out


def extract_reasoning_patterns(record, agg, constraints=None):
    patterns  = []
    head      = record["head"]
    true_tail = record.get("true_tail") or record.get("tail", "")
    only_tail = set(record.get("only_tail_has", []))
    selected  = set(agg.get("selected_relations", []))
    grounded  = selected & only_tail
    if constraints is not None:
        all_rels = set(constraints["rel_to_tail_counts"].keys())
        for rel in grounded:
            if rel in all_rels:
                patterns.append((head, rel, true_tail))
    return patterns


def writeback(agg, record, constraints=None):
    true_tail = record.get("true_tail") or record.get("tail", "")
    if agg.get("failure_type") != "resolved":
        return
    if agg.get("final_answer","").strip().lower() != true_tail.strip().lower():
        return
    patterns = extract_reasoning_patterns(record, agg, constraints=constraints)
    if not patterns:
        print("  [TSV] skipped — no grounded discriminative relations"); return
    new_file = not os.path.exists(EPISODIC_TSV)
    with open(EPISODIC_TSV, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if new_file:
            w.writerow(["head", "relation", "tail"])
        w.writerows(patterns)
    print(f"  [TSV] {len(patterns)} patterns written")
    for h, r, t in patterns:
        write_episodic(h, r, t, agg.get("chosen_agent","?"),
                       f"type_constraint: {h} --{r}--> {t}")
    write_semantic(agg.get("chosen_agent","?"), record.get("relation","?"),
                   list({r for _, r, _ in patterns}))


def run_pipeline(record, tsv_memory=None, df_ref=None, constraints=None):
    if "true_tail" not in record:
        record["true_tail"] = record.get("tail", "")
    if "hop_type" not in record:
        record["hop_type"] = "multi"
    t = f"({record['head']}, {record['relation']}, {record['true_tail']})"
    print(f"\n{'='*55}\nPIPELINE  {t}\n"
          f"rank={record['true_rank']}  hop={record['hop_type']}\n{'='*55}")
    a_out, b_out = run_parallel_staggered(record, tsv_memory=tsv_memory)
    s_a = grounded_score(a_out, record, "A", df_ref=df_ref, constraints=constraints)
    s_b = grounded_score(b_out, record, "B",                constraints=constraints)
    agg = aggregate(a_out, b_out, s_a, s_b, record)
    writeback(agg, record, constraints=constraints)
    correct = (agg.get("final_answer","").strip().lower()
               == record["true_tail"].strip().lower())
    print(f"\n{'✓' if correct else '✗'}  final={agg.get('final_answer')}  "
          f"agent={agg.get('chosen_agent')}")
    return {
        "triple":        t,
        "true_tail":     record["true_tail"],
        "hop_type":      record["hop_type"],
        "model_rank":    record["true_rank"],
        "agent_a":       a_out,
        "agent_b":       b_out,
        "score_a":       s_a,
        "score_b":       s_b,
        "aggregator":    agg,
        "final_correct": correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Run loop  →  val_hard_results.json
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_FILE = "val_hard_checkpoint.json"
RESULTS_FILE    = "val_hard_results.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {}


def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)


def triple_key(record):
    return f"{record['head']}|{record['relation']}|{record['tail']}"


checkpoint   = load_checkpoint()
already_done = set(checkpoint.keys())
remaining    = [r for r in val_hard if triple_key(r) not in already_done]

print(f"val_hard total:    {len(val_hard)}")
print(f"Already completed: {len(already_done)}")
print(f"Remaining:         {len(remaining)}")

all_results = list(checkpoint.values())

for i, record in enumerate(remaining):
    key = triple_key(record)
    print(f"\n[{i+1}/{len(remaining)}]  {key}  rank={record['true_rank']}")
    try:
        result = run_pipeline(
            record,
            tsv_memory  = tsv_memory,
            df_ref      = df_train,
            constraints = constraints,
        )
        checkpoint[key] = result
        all_results.append(result)
        save_checkpoint(checkpoint)
    except Exception as e:
        print(f"  [ERROR] {key}: {e}")
        checkpoint[key] = {"error": str(e), "triple": key}
        save_checkpoint(checkpoint)
    if i < len(remaining) - 1:
        time.sleep(BETWEEN_RECORDS)


# ── Final summary ─────────────────────────────────────────────────────────────

clean_results = [r for r in all_results if "error" not in r]
with open(RESULTS_FILE, "w") as f:
    json.dump(clean_results, f, indent=2)

errors  = [r for r in all_results if "error" in r]
correct = [r for r in clean_results if r.get("final_correct")]
agents  = [r["aggregator"].get("chosen_agent") for r in clean_results]
ftypes  = [r["aggregator"].get("failure_type")  for r in clean_results]

print(f"\n{'='*55}  FINAL SUMMARY")
print(f"Total processed: {len(all_results)}")
print(f"Errors/skipped:  {len(errors)}")
print(f"Clean results:   {len(clean_results)}")
print(f"Correct:         {len(correct)} / {len(clean_results)}"
      f"  ({100*len(correct)/max(len(clean_results),1):.1f}%)")
print(f"\nAgent chosen:")
for agent, count in Counter(agents).most_common():
    print(f"  Agent {agent}: {count}")
print(f"\nFailure types:")
for ftype, count in Counter(ftypes).most_common():
    print(f"  {ftype}: {count}")
print(f"\nFiles written:")
print(f"  {RESULTS_FILE}")
print(f"  {CHECKPOINT_FILE}")
print(f"  {EPISODIC_TSV}")
print(f"  {PREPROCESSED_TRAIN}  {PREPROCESSED_TEST}")
print(f"  {VAL_FILE}  {HELD_OUT_FILE}")
