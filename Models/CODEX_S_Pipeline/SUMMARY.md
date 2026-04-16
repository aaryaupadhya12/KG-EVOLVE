# CODEX-S MODULARIZED PIPELINE - COMPLETE SUMMARY

## What This Is

A **fully modularized and documented** version of the CODEX-S Knowledge Graph reasoning pipeline. Each function has detailed docstrings explaining its purpose, parameters, and return values.

**Key Benefits:**
- Easy to understand: each module has a single responsibility
- Easy to iterate: modify Agent A's prompt in isolation
- Easy to reuse: port agents to other KG datasets (Nations, UMLS, FB15k, etc.)
- Easy to scale: checkpointing allows resumable runs

---

## Architecture at a Glance

```
CODEX-S Pipeline (Modularized)
│
├── data_loader.py              Load CODEX-S dataset & build ID mappings
├── model_utils.py              Load ComplEx model & score triples
├── kg_utils.py                 Graph operations, subgraph extraction
├── embedding_utils.py          Entity similarity via embeddings
├── memory_manager.py           Episodic/semantic memory (FAISS + TSV)
├── context_builder.py          Build rich context strings for agents
├── llm_agents.py               Agent A (type-constraint), Agent B (structural)
├── scorer.py                   Verify claims, compute quality scores, route
├── pipeline.py                 Orchestration, checkpointing, results
│
└── Documentation
    ├── README.md               Start here
    ├── API_REFERENCE.py        Quick lookup of functions
    ├── EXAMPLE_USAGE.py        Step-by-step walkthrough
    └── PARAMETERS_GUIDE.md     Detailed parameter tuning
```

---

## How to Use

### Quick Start (3 minutes)

```python
from pipeline import run_full_pipeline

# Load data (elsewhere in your script)
# ... load_codex_data(), preprocess_triples(), etc.

# Run full pipeline
results, summary = run_full_pipeline(
    records=val_hard_records,
    entity_to_id=entity_to_id,
    relation_to_id=relation_to_id,
    df_train=df_train,
    constraints=constraints
)

print(f"Accuracy: {summary['accuracy']:.1f}%")
```

### Modify Agent A (5 minutes)

```python
# Edit llm_agents.py

AGENT_A_SYSTEM = """You are Agent A...
[MODIFY THIS TO CHANGE BEHAVIOR]
"""

# Re-run pipeline with modified agent
results, summary = run_full_pipeline(...)
```

### Port to Other Dataset (30 minutes)

```python
# Edit data_loader.py: replace load_codex_data() with your dataset loader
# Change entity_to_id, relation_to_id to your dataset
# Retrain ComplEx model on your data (or use pretrained if available)
# Everything else stays the same!
```

---

## Core Concepts Explained

### 1. **Agent A vs Agent B**

| Agent | Specializes In | Good At | Bad At |
|-------|----------------|---------|--------|
| **A** | Type Constraints | "This entity never appears as tail" | Multi-hop reasoning |
| **B** | Structural Reasoning | Finding paths in subgraph | Rare entity types |

**Routing**: Quality score determines which to trust
- Quality = coverage × (1 - 0.5 × contamination)
- Higher coverage + lower contamination → higher quality

### 2. **Memory System** (Optional)

**Episodic**: Store past (head, relation, tail) successes
- Query before agents: "Have we seen this before?"
- Example: "Alice hasChild → Bob | agent A | resolved"

**Semantic**: Store failure type patterns
- Example: "Agent B wins on structural_gap | key_rels: spouse, livesIn"

**TSV**: Fast structured memory (exact match)
- Quick lookup: head → [(relation, tail), ...]

### 3. **Type Constraints**

Learned from training data: P(tail | relation, head)

- **Specific**: "For Alice + hasChild, likely tails are Bob, Carol" (0.5, 0.3)
- **General**: "For any parent + hasChild, common tails are Bob, Carol, David" (0.3, 0.2, 0.2)

Agents use this to override embedding confusion.

### 4. **Subgraph Extraction**

**Tier System** (smartly prioritizes triples):
- **Tier 1**: "Separating relations" (only true_tail has, not predicted) = GOLD
- **Tier 2**: Triples connecting important entities
- **Tier 3**: Triples from head
- **Tier 4**: Background

Example: If "hasChild" only appears in true_tail's profile, it gets Tier 1 marking (◆).

### 5. **Scoring & Routing**

```
Agent Output
    ↓
Verify claims (remove hallucinations)
    ↓
Compute quality_score (coverage × purity)
    ↓
Choose A or B based on quality
    ↓
Final answer + confidence
```

---

## File-by-File Explanation

### `data_loader.py`
**What**: Load CODEX-S, build entity/relation IDs, preprocess triples
**When to modify**: 
- Different dataset (Nations, UMLS, etc.)
- Change preprocessing caching strategy
**Key functions**: `load_codex_data()`, `preprocess_all_triples()`, `build_type_constraints()`

### `model_utils.py`
**What**: Load ComplEx model, score triples, rank entities
**When to modify**:
- Use different model architecture
- Change batch sizing for memory constraints
**Key functions**: `load_pretrained_model()`, `get_full_ranking_filtered_batched()`

### `kg_utils.py`
**What**: Graph operations (subgraph extraction, hop classification)
**When to modify**:
- Change hop distance computation
- Adjust subgraph extraction strategy
**Key functions**: `extract_subgraph()`, `focused_subgraph()`, `hop_classifier()`

### `embedding_utils.py`
**What**: Entity similarity using embeddings
**When to modify**:
- Change similarity metric (cosine → other)
- Adjust relational overlap weighting
**Key functions**: `similarity_summary()`, `embedding_distance()`

### `memory_manager.py`
**What**: Episodic + semantic memory stores (FAISS + TSV)
**When to modify**:
- Change what patterns get stored
- Different memory lookup strategy
**Key functions**: `MemoryManager.record_resolution()`, `query()`

### `context_builder.py`
**What**: Build agent context strings from records
**When to modify**:
- Change context format/verbosity
- Add new signals to context
- Adjust subgraph prioritization
**Key functions**: `build_agent_context()`, `trim_subgraph()`

### `llm_agents.py`
**What**: Agent A and B implementations + LLM backend
**When to modify**:
- Change agent prompts (AGENT_A_SYSTEM, AGENT_B_SYSTEM)
- Adjust confidence calibration rules
- Add new reasoning steps
**Key functions**: `agent_a()`, `agent_b()`, `run_parallel_staggered()`

### `scorer.py`
**What**: Verification, quality scoring, routing, aggregation
**When to modify**:
- Change routing logic (which agent to trust)
- Adjust quality score formula
- Change verification rules
**Key functions**: `compute_quality_score()`, `route_decision()`, `aggregate_results()`

### `pipeline.py`
**What**: Main orchestration, checkpointing, result aggregation
**When to modify**:
- Change how checkpoints work
- Adjust result summary metrics
- Add new result analysis
**Key functions**: `Pipeline.run_triple()`, `run_full_pipeline()`

---

## Where to Start for Different Goals

### Goal: Understand the pipeline
1. Read `README.md` (overview)
2. Read `EXAMPLE_USAGE.py` (step-by-step walkthrough)
3. Read `API_REFERENCE.py` (quick lookup)

### Goal: Improve Agent A accuracy
1. Look at `llm_agents.py` → `AGENT_A_SYSTEM`
2. Understand type constraints in `kg_utils.py` → `get_type_constraint_signal()`
3. Modify prompt, re-run `run_full_pipeline()`
4. Check results in `val_hard_results.json`

### Goal: Improve Agent B accuracy
1. Look at `llm_agents.py` → `AGENT_B_SYSTEM`
2. Understand subgraph extraction in `kg_utils.py`
3. Modify prompt, re-run
4. Analyze with `analyze_results()`

### Goal: Change routing logic
1. Look at `scorer.py` → `route_decision()`
2. Modify routing rules
3. Test with `run_full_pipeline(...)`

### Goal: Port to different dataset
1. Edit `data_loader.py` → `load_codex_data()`
2. Replace with your dataset loader
3. Retrain ComplEx on your data
4. Run remainder of pipeline (should work!)

### Goal: Debug why agents fail
1. Check `val_hard_results.json` → look at agent outputs
2. Find record in `EXAMPLE_USAGE.py` step 7 (manual processing)
3. Print context, agent reasoning, quality scores
4. Identify bottleneck (A reasoning wrong? B reasoning wrong? Routing wrong?)

---

## Key Design Decisions Explained

### **Preprocessing Caching**
- **Problem**: Model scoring is bottleneck (1-2 sec per triple)
- **Solution**: Score all triples ONCE, save to JSON
- **Benefit**: Subsequent runs load JSON instantly, skip model calls
- **Trade-off**: First run takes hours, but pays off for iteration

### **Parallel Agent Execution**
- **Problem**: Both agents run sequentially = slow
- **Solution**: Run in threads with stagger delay
- **Benefit**: ~50% faster (two LLM calls happen parallelized where possible)
- **Trade-off**: Small GPU memory spike

### **Grounded Verification**
- **Problem**: Agents hallucinate relations that don't exist
- **Solution**: After agent outputs, verify claims against training data
- **Benefit**: Remove hallucinations before routing
- **Trade-off**: Extra computation, but prevents bad routing

### **Memory as Iterative Learning**
- **Problem**: Agents have no memory across triples
- **Solution**: Store successful patterns in episodic/semantic memory
- **Benefit**: Context hints improve subsequent reasoning
- **Trade-off**: Memory size grows, query overhead

### **Tier-Ranked Subgraphs**
- **Problem**: Too much subgraph noise, not enough signal
- **Solution**: Tier system prioritizes "separating relations"
- **Benefit**: Agents focus on what matters most
- **Trade-off**: Loses some background context

---

## Common Customizations

```python
# 1. Modify Agent A prompt
# File: llm_agents.py
AGENT_A_SYSTEM = """Your custom prompt..."""

# 2. Change quality_score formula
# File: scorer.py, function compute_quality_score()
quality_score = coverage_score * (1 - 0.7 * contamination)  # Changed 0.5 → 0.7

# 3. Adjust routing thresholds
# File: scorer.py, function route_decision()
if qa > 0.6 and qb <= 0.6:  # Changed 0.5 → 0.6
    return "A"

# 4. Change subgraph size
# File: context_builder.py, function build_subgraph_str()
build_subgraph_str(record, max_triples=8)  # Changed 12 → 8

# 5. Disable memory
# File: pipeline.py, function run_full_pipeline()
run_full_pipeline(..., use_memory=False)

# 6. Use LLM for failure classification
# File: pipeline.py, function run_full_pipeline()
run_full_pipeline(..., use_llm_aggregation=True)
```

---

## Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| "Agent A always picks wrong" | Type constraints too noisy | Lower confidence thresholds, check build_type_constraints() |
| "Agent B hallucinating relations" | No verification | Enable verify_type_fit() in compute_quality_score() |
| "Accuracy below 50%" | Bad routing (trusting wrong agent) | Check route_decision() thresholds |
| "Out of GPU memory" | Batch size too large | Decrease batch_size in get_full_ranking_filtered_batched() |
| "LLM timeout" | Model inference slow | Reduce context size, disable memory |
| "Results not resuming" | Checkpoint corrupted | Delete val_hard_checkpoint.json, re-run |
| "Same output every time" | Temperature too low | Increase to 0.5+ in LLMBackend() |

---

## Next Steps

1. **Read** `EXAMPLE_USAGE.py` for step-by-step walkthrough
2. **Run** on val_hard[:10] to understand flow
3. **Modify** agent prompts based on analysis
4. **Benchmark** changes with analyze_results()
5. **Scale up** to full val_hard dataset
6. **Port** to other datasets by changing data_loader

---

## Questions?

Refer to:
- **How does [function] work?** → See docstring in module file
- **What parameter should I change?** → See PARAMETERS_GUIDE.md
- **How do I modify [component]?** → See relevant module file
- **Does this work for [dataset]?** → Probably! Adjust data_loader.py

---

**Happy iterating! **
