# CODEX-S Modularized Pipeline

A fully modularized and documented version of the CODEX-S knowledge graph reasoning pipeline.

## Architecture Overview

```
CODEX-S Pipeline
├── DATA LOADING & PREPROCESSING
│   └── data_loader.py          → Load CODEX-S, build entity/relation IDs, create graph
│
├── KNOWLEDGE GRAPH UTILITIES
│   └── kg_utils.py             → Subgraph extraction, hop classification, type constraints
│
├── EMBEDDING & MODEL
│   ├── embedding_utils.py      → Entity embeddings, similarity computations
│   └── model_utils.py          → ComplEx model inference, ranking, scoring
│
├── MEMORY MANAGEMENT
│   └── memory_manager.py       → Episodic/semantic FAISS stores, TSV memory
│
├── LLM AGENTS
│   ├── llm_agents.py           → Agent A & B implementations
│   └── context_builder.py      → Build context strings for agents
│
├── VERIFICATION & SCORING
│   └── scorer.py               → Grounded scoring, hallucination detection, aggregation
│
└── ORCHESTRATION
    └── pipeline.py             → Main run loop, checkpoint handling, results aggregation
```

## Module Dependencies & Data Flow

```
1. data_loader.py
   → Loads CODEX-S, creates entity/relation IDs, builds training graph
   → Outputs: df_train, df_test, entity_to_id, relation_to_id, graph

2. model_utils.py
   → Loads pretrained ComplEx from HuggingFace
   → Used for: ranking, similarity, scoring

3. kg_utils.py
   → Takes: graph, dataframes, entity/relation IDs
   → Generates: subgraphs, type constraints, hop classifications

4. embedding_utils.py
   → Takes: model_utils entity embeddings
   → Computes: similarity summaries for entities

5. scorer.py (Verification stage)
   → Takes: agent outputs, records, type constraints
   → Verifies: relation claims, path existence
   → Routes: decision to Agent A or B

6. memory_manager.py (Async)
   → Stores: episodic (h, r patterns), semantic (failure types)
   → Read before agent call, updated after successful resolution

7. llm_agents.py (Parallel execution)
   → Takes: context from context_builder.py
   → Runs: Agent A & B in parallel with stagger
   → Outputs: predictions, confidence, reasoning

8. pipeline.py
   → Orchestrates: full workflow, checkpointing, result aggregation
   → Outputs: val_hard_results.json, episodic_memory.tsv
```

## Quick Start

### 1. First Run (Preprocessing)
```python
from data_loader import load_codex_data, preprocess_triples
from model_utils import load_pretrained_model

# Load data
df_train, df_test, entity_to_id, relation_to_id = load_codex_data()

# Load model
model = load_pretrained_model()

# Preprocess (scores all triples with model)
preprocess_triples(df_train, "train")
preprocess_triples(df_test, "test")
# Saves to JSON — never need to score again for these records
```

### 2. Subsequent Runs (No Scoring)
```python
from data_loader import load_preprocessed_records

# Load preprocessed records from JSON (instant, no model calls)
train_records = load_preprocessed_records("CODEX_S_preprocessed_train.json")
test_records = load_preprocessed_records("CODEX_S_preprocessed_test.json")

# Now use these records with agents directly
```

### 3. Run Agent Pipeline on Hard Records
```python
from pipeline import run_full_pipeline

results = run_full_pipeline(
    records=val_hard_subset,
    use_memory=True,
    parallel_agents=True
)
```

## Key Design Decisions

### 1. **Clear Separation of Concerns**
- Each module has a single responsibility
- No circular dependencies
- Easy to test and iterate individual components

### 2. **Preprocessing Cache**
- All triple scoring happens ONCE during preprocessing
- Results saved to JSON
- Subsequent runs load JSON instantly (no model inference)
- This is critical for Kaggle/GPU-limited environments

### 3. **Parallel Agent Execution**
- Agent A and Agent B run in parallel threads with stagger
- Reduces latency when using slow LLMs
- Synchronization point at aggregator

### 4. **Grounded Verification**
- After agent reasoning, verify claimed relations exist
- Remove hallucinated relations
- Compute quality_score = coverage * (1 - 0.5 * contamination)
- Route decision based on quality_score: A or B

### 5. **Memory as Iterative Learning**
- Episodic: (h, r, t) patterns from resolved cases
- Semantic: failure types agents excel at
- Updated after each successful resolution
- Provided as context hints to LLM agents

## Function Reference Map

See individual modules for complete documentation:

| Module | Key Functions |
|--------|---------------|
| `data_loader.py` | `load_codex_data()`, `preprocess_triples()`, `build_type_constraints()` |
| `kg_utils.py` | `extract_subgraph()`, `focused_subgraph()`, `hop_classifier()` |
| `embedding_utils.py` | `get_entity_embeddings()`, `similarity_summary()` |
| `model_utils.py` | `load_pretrained_model()`, `score_triple()`, `get_full_ranking_filtered_batched()` |
| `memory_manager.py` | `query_episodic()`, `write_episodic()`, `load_tsv_memory()` |
| `llm_agents.py` | `agent_a()`, `agent_b()`, `run_parallel_staggered()` |
| `context_builder.py` | `build_agent_context()`, `build_subgraph_str()` |
| `scorer.py` | `grounded_score()`, `verify_type_fit()`, `aggregate()` |
| `pipeline.py` | `run_pipeline()`, `run_full_pipeline()` |

## Expected Outputs

After running on val_hard:

```
val_hard_results.json          → Complete results with all agent reasoning
episodic_memory.tsv            → (head, relation, tail) patterns learned
val_hard_checkpoint.json       → Checkpoint for resumable runs
```

Result structure:
```json
{
  "triple": "(entity1, relation, entity2)",
  "true_tail": "entity2",
  "hop_type": "single|multi",
  "model_rank": 5,
  "agent_a": { "prediction": "...", "confidence": 0.85, ... },
  "agent_b": { "prediction": "...", "path_found": "...", ... },
  "score_a": { "quality_score": 0.75, "contamination": 0.1, ... },
  "score_b": { "quality_score": 0.65, "contamination": 0.2, ... },
  "aggregator": {
    "final_answer": "entity2",
    "chosen_agent": "A",
    "failure_type": "resolved|similarity_confusion|structural_gap|both_failed"
  },
  "final_correct": true
}
```

## Customization Guide

### To iterate Agent A prompts:
Edit `llm_agents.py` → `AGENT_A_SYSTEM_PROMPT`

### To change memory strategy:
Edit `memory_manager.py` → `write_episodic()`, `query_episodic()`

### To modify routing logic:
Edit `scorer.py` → `_python_route()`

### To skip preprocessing cache:
Edit `data_loader.py` → `preprocess_triples()` with `force=True`

### To tune quality_score weights:
Edit `scorer.py` → `quality_score = coverage_score * (1 - 0.5 * contamination)`
Change `0.5` multiplier for contamination importance

---

For detailed function documentation, see individual module files.
