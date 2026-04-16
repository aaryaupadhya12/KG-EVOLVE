"""
EXAMPLE USAGE - Step by Step
=============================

Complete example of running the modularized CODEX-S pipeline.
Demonstrates all major components and customization points.

For quick start, jump to FULL PIPELINE EXAMPLE at the end.
"""

# ═════════════════════════════════════════════════════════════════════════════
# 0. SETUP & IMPORTS
# ═════════════════════════════════════════════════════════════════════════════

import json
import os
from pathlib import Path

# Data loading
from data_loader import (
    load_codex_data, build_graph_from_df, build_type_constraints,
    load_or_preprocess_triples
)

# Models
from model_utils import load_pretrained_model, get_full_ranking_filtered_batched
from embedding_utils import get_entity_embeddings_cached, similarity_summary

# KG utils
from kg_utils import extract_subgraph, focused_subgraph, hop_classifier

# Agent pipeline
from llm_agents import LLMBackend, run_parallel_staggered
from context_builder import build_agent_context
from scorer import compute_quality_score, route_decision, aggregate_results
from memory_manager import MemoryManager

# Main orchestration
from pipeline import Pipeline, run_full_pipeline, analyze_results


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD CODEX-S DATA
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("STEP 1: Load CODEX-S Dataset")
print("=" * 70)

# Load dataset with English labels and ID mappings
df_train, df_test, df_valid, df_all, entity_to_id, relation_to_id, \
    id_to_entity, id_to_relation = load_codex_data(size="s", code="en")

print(f"Train: {len(df_train)}  Valid: {len(df_valid)}  Test: {len(df_test)}")
print(f"Entities: {len(entity_to_id)}  Relations: {len(relation_to_id)}")

# Build knowledge graph
graph = build_graph_from_df(df_train)
print(f"Graph nodes: {len(graph)}")

# Build type constraints (statistical priors)
constraints = build_type_constraints(df_train)
print(f"Type constraints built for {len(constraints['rel_to_tail_dist'])} relations")


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOAD PRETRAINED MODEL
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 2: Load Pretrained Model")
print("=" * 70)

model = load_pretrained_model()
# Model is cached globally, subsequent calls are instant

# Extract embeddings for similarity (done once, cached)
embeddings = get_entity_embeddings_cached(model)
print(f"Embeddings shape: {embeddings.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESS TRIPLES (Optional - skip if files exist)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 3: Preprocess Triples (First Run Only)")
print("=" * 70)

train_records = load_or_preprocess_triples(
    df_train,
    "train",
    "CODEX_S_preprocessed_train.json",
    force=False,  # Change to True to force recomputation
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    id_to_entity=id_to_entity,
    id_to_relation=id_to_relation,
    model=model,
    graph=graph,
    df_train=df_train,
    embedding_fn=lambda e, k=5: similarity_summary(e, entity_to_id, id_to_entity, embeddings, df_train, k),
    failure_summary_fn=lambda h, r, t, p, c: (None, {}),  # Simplified
    focused_subgraph_fn=focused_subgraph,
    hop_classifier_fn=hop_classifier,
    similarity_summary_fn=lambda e, k=5: similarity_summary(e, entity_to_id, id_to_entity, embeddings, df_train, k),
)

test_records = load_or_preprocess_triples(
    df_test,
    "test",
    "CODEX_S_preprocessed_test.json",
    force=False,
    # (same parameters as train_records)
)

print(f"Train records: {len(train_records)}  Test records: {len(test_records)}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. SPLIT VALIDATION & HELD-OUT SETS
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 4: Create Validation & Held-Out Splits")
print("=" * 70)

from sklearn.model_selection import train_test_split

val_records, held_out_records = train_test_split(
    test_records, test_size=0.5, random_state=42
)

# Focus on hard failures for agent evaluation
val_hard = [r for r in val_records if r.get("hard_failure", False)]

print(f"Val records: {len(val_records)}  Hard failures: {len(val_hard)}")
print(f"Held-out (sealed): {len(held_out_records)}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. INITIALIZE MEMORY (Optional)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 5: Initialize Memory Manager")
print("=" * 70)

memory_manager = MemoryManager(
    episodic_path="episodic.faiss",
    semantic_path="semantic.faiss",
    tsv_path="episodic_memory.tsv",
)


# ═════════════════════════════════════════════════════════════════════════════
# 6. INITIALIZE LLM
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 6: Initialize LLM Backend")
print("=" * 70)

llm = LLMBackend(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    quantize=True,  # 4-bit quantization to fit on GPU
    temperature=0.3  # Low temperature = more deterministic
)


# ═════════════════════════════════════════════════════════════════════════════
# 7. PROCESS SINGLE TRIPLE (Example)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 7: Process Single Triple (Manual Example)")
print("=" * 70)

record = val_hard[0]  # First hard record
print(f"Processing: {record['head']}, {record['relation']}, {record.get('tail','?')}")

# Build context
context = build_agent_context(record)
print("\n--- CONTEXT ---")
print(context[:400] + "...\n")

# Run agents (parallel)
a_out, b_out = run_parallel_staggered(record, context, llm)

print("\n--- AGENT A OUTPUT ---")
print(json.dumps(a_out, indent=2)[:300] + "...\n")

print("--- AGENT B OUTPUT ---")
print(json.dumps(b_out, indent=2)[:300] + "...\n")

# Score
score_a = compute_quality_score(a_out, record, "A", df_ref=df_train, constraints=constraints)
score_b = compute_quality_score(b_out, record, "B", df_ref=df_train, constraints=constraints)

print(f"Agent A quality: {score_a['quality_score']}")
print(f"Agent B quality: {score_b['quality_score']}\n")

# Route
chosen = route_decision(score_a, score_b, record)
print(f"Chosen agent: {chosen}")

# Aggregate
agg = aggregate_results(a_out, b_out, score_a, score_b, record)
print(f"Final answer: {agg['final_answer']}")
print(f"Failure type: {agg['failure_type']}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. CUSTOMIZE AGENT PROMPTS (Example)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 8: Customize Agent Prompts")
print("=" * 70)
print("""
To modify Agent A or B behavior:

1. Edit llm_agents.py
2. Update AGENT_A_SYSTEM or AGENT_B_SYSTEM constants
3. Change system prompt to adjust reasoning instructions

Example customization:
  - Remove type constraint focus, emphasize structural reasoning
  - Add domain-specific knowledge (medical, chemistry, etc.)
  - Change confidence calibration rules
  - Add new reasoning steps

Then re-run pipeline with modified llm_agents.py
""")


# ═════════════════════════════════════════════════════════════════════════════
# 9. CUSTOMIZE ROUTING LOGIC (Example)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 9: Customize Routing Logic")
print("=" * 70)
print("""
To change how decisions are routed (A vs B):

1. Edit scorer.py
2. Modify route_decision() function
3. Change thresholds or prioritization

Current logic:
  - If A quality > 0.5 and B <= 0.5 → use A
  - If B quality > 0.5 and A <= 0.5 → use B
  - Tie: use lower contamination
  - Last tie: single-hop→A, multi-hop→B

Custom routing examples:
  - Always prefer A for single-hop (more likely correct)
  - Use ensemble: trust consensus if A == B
  - Weight by agent expertise (track historical accuracy)
""")


# ═════════════════════════════════════════════════════════════════════════════
# 10. RUN FULL PIPELINE (Recommended)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 10: Run Full Pipeline (Recommended)")
print("=" * 70)

results, summary = run_full_pipeline(
    records=val_hard[:10],  # Process first 10 as example
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    df_train=df_train,
    constraints=constraints,
    llm_model="Qwen/Qwen2.5-7B-Instruct",
    checkpoint_dir="output/",
    use_memory=True,
    delay_between_records=1.0,
    use_llm_aggregation=False,  # Set to True to use LLM for failure classification
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Accuracy: {summary['accuracy']:.1f}%")
print(f"Agent A chosen: {summary['agent_choice'].get('A', 0)} times")
print(f"Agent B chosen: {summary['agent_choice'].get('B', 0)} times")
print(f"Failure types: {summary['failure_types']}")


# ═════════════════════════════════════════════════════════════════════════════
# 11. ANALYZE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 11: Analyze Results")
print("=" * 70)

analysis = analyze_results(results)

print(f"Overall accuracy: {analysis['overall']['accuracy']:.1f}%")
print("\nBy hop type:")
for hop, stats in analysis['by_hop'].items():
    print(f"  {hop}: {stats['accuracy']:.1f}%")

print("\nBy agent:")
for agent, stats in analysis['by_agent'].items():
    print(f"  Agent {agent}: {stats['accuracy']:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# TIPS FOR ITERATION
# ═════════════════════════════════════════════════════════════════════════════

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║ TIPS FOR ITERATING ON CODEX-S (or CODEX-M/L)                              ║
╚════════════════════════════════════════════════════════════════════════════╝

1. QUICK PROTOTYPING
   - Use val_hard[:10] to test changes quickly
   - Use_llm_aggregation=False to skip slow LLM calls
   - Check results in output/val_hard_results.json

2. AGENT TUNING
   - Modify AGENT_A_SYSTEM or AGENT_B_SYSTEM in llm_agents.py
   - A: focuses on type constraints (P(tail | relation))
   - B: focuses on structural reasoning (subgraph paths)
   - Run on subset to evaluate before full run

3. ROUTING LOGIC
   - Edit route_decision() in scorer.py
   - Log chosen agent and quality scores
   - Analyze when routing fails (wrong agent chosen)

4. CONTEXT PRUNING
   - Modify build_subgraph_str() to reduce tokens
   - Reduce max_triples to 6-8 for tighter context
   - Remove similarity block if not helpful

5. FAILURE TYPE CLASSIFICATION
   - Modify classify_failure_type() in context_builder.py
   - Or use LLM aggregation (use_llm_aggregation=True)
   - Track common patterns per dataset

6. MEMORY STRATEGY
   - Modify MemoryManager in memory_manager.py
   - Record different patterns (relation-specific, failure-specific)
   - Query with different prompts before agent call

7. FOR OTHER KG DATASETS (Nations, UMLS)
   - Replace CODEX-S loader with Nations/UMLS loader
   - Retrain ComplEx model on target dataset
   - Recompute type constraints from target training data
   - Rest of pipeline remains unchanged!

8. SCALING UP
   - Use checkpointing: Pipeline resumes from last record
   - Run on full val_hard (not subset)
   - Use held_out_records for final evaluation (sealed)
   - Monitor GPU memory with use_memory=True (saves FAISS stores)

9. EVALUATION
   - run_full_pipeline() writes val_hard_results.json
   - Use analyze_results() for breakdown by hop type / agent
   - Compare accuracy before/after modifications

10. SERIALIZATION & SHARING
    - Save results: val_hard_results.json
    - Save summary: val_hard_summary.json
    - Save analysis: run export_analysis(results, "analysis.json")
    - Results fully reproducible with same model + random_seed
""")
