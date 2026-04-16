"""
MODULE INDEX & QUICK REFERENCE
===============================

Index of all modules, files, and quick reference for what each does.
"""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ DOCUMENTATION FILES (START HERE)                                        │
# └─────────────────────────────────────────────────────────────────────────┘

# README.md
#   → Architecture overview
#   → Module dependencies & data flow
#   → Quick start examples
#   Read this FIRST

# SUMMARY.md
#   → Complete overview with design decisions explained
#   → File-by-file explanation
#   → Common customizations
#   → Troubleshooting guide
#   Read this SECOND

# EXAMPLE_USAGE.py
#   → Step-by-step walkthrough (commented Python)
#   → Shows how to use each module
#   → Tips for iteration and customization
#   → Practical examples
#   Run this to understand pipeline

# API_REFERENCE.py
#   → One-page function reference
#   → Quick lookup of function names
#   → Return types and basic usage
#   Use when you need specific function signature

# PARAMETERS_GUIDE.md
#   → Detailed parameter explanations
#   → When to adjust each parameter
#   → Tuning grid for improvements
#   → Checklist for porting to new datasets
#   Read when customizing hyperparameters


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ PIPELINE MODULES                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

# data_loader.py (300 lines)
#   ├─ load_codex_data()
#   │    → Load CODEX-S with English labels + ID mappings
#   │    ✓ Use for: Loading any CODEX variant (S/M/L)
#   │    ✗ Don't use: If using different dataset (Nations, UMLS)
#   │       → Modify: Replace with your dataset loader
#   │
#   ├─ build_graph_from_df()
#   │    → Create adjacency list {entity: [(rel, tail), ...]}
#   │    ✓ Use for: Building KG from any dataframe
#   │
#   ├─ build_type_constraints()
#   │    → Compute P(tail | relation, head) from training data
#   │    ✓ Use for: Building statistical type priors
#   │    ⚙ Tune: If type_fit too sparse/dense, adjust confidence thresholds
#   │
#   ├─ get_type_constraint_signal()
#   │    → Get type_fit scores for true vs predicted
#   │    ✓ Use for: Diagnostic signals for agents
#   │
#   └─ preprocess_all_triples()
#        → Score ALL triples with model (SLOW! Run once)
#        ✓ Use for: First-time preprocessing
#        ⚠ Warning: ~3-5 hours for 9k triples on GPU
#        ✓ Output: JSON caches for instant reload

# model_utils.py (300 lines)
#   ├─ load_pretrained_model()
#   │    → Load ComplEx from HuggingFace, cached globally
#   │    ✓ Use for: Any triple scoring
#   │    ⚠ Note: Model must match entity/relation ID order
#   │
#   ├─ score_triple(), score_batch()
#   │    → Score single or batch of triples
#   │    ✓ Use for: Debug, custom scoring workflows
#   │
#   ├─ get_full_ranking_filtered_batched()
#   │    → Rank ALL candidate entities for (head, relation)
#   │    ✓ Use for: CORE MODEL INFERENCE
#   │    ⚙ Tune: batch_size based on GPU memory
#   │
#   └─ get_entity_embeddings()
#        → Extract embeddings for similarity
#        ✓ Use for: Caching once during preprocessing

# kg_utils.py (350 lines)
#   ├─ extract_subgraph()
#   │    → BFS extraction of k-hop neighborhood around entity
#   │    ✓ Use for: Single entity neighborhood
#   │    ⚙ Tune: hops=3 for deep, hops=1 for shallow
#   │
#   ├─ focused_subgraph()
#   │    → Multi-entity subgraph with tier prioritization
#   │    ✓ Use for: Agent context building
#   │    ⚙ Tune: max_triples for context size
#   │
#   ├─ hop_classifier()
#   │    → Classify path type between entities
#   │    ✓ Use for: Understanding query difficulty
#   │
#   └─ failure_summary(), get_type_constraint_signal()
#        → Diagnostic signals for agent context

# embedding_utils.py (200 lines)
#   ├─ get_entity_embeddings_cached()
#   │    → Cache embeddings globally
#   │
#   ├─ similarity_summary()
#   │    → Find k most similar entities (embedding + relational)
#   │    ✓ Use for: Adding entity neighbors to context
#   │    ⚙ Tune: k for context size
#   │
#   ├─ embedding_distance()
#   │    → Cosine similarity between two entities
#   │
#   └─ find_similar_cluster()
#        → Find entity cluster similar to multiple entities

# memory_manager.py (400 lines)
#   ├─ FAISSMemory (vector store)
#   │    → Similarity-based memory (episodic/semantic)
#   │
#   ├─ TSVMemory (structured store)
#   │    → Fast exact-match memory (head → [(rel, tail), ...])
#   │
#   ├─ MemoryManager (high-level API)
#   │    → Unified interface for all memory types
#   │    ✓ Use for: get_context_for_query(), record_resolution()
#   │
#   └─ load_tsv_memory(), get_memory_hint() (legacy)
#        → Backward compatible functions

# context_builder.py (300 lines)
#   ├─ trim_subgraph()
#   │    → Tier-rank and truncate subgraph
#   │    ✓ Use for: Internal (called by build_agent_context)
#   │
#   ├─ build_subgraph_str()
#   │    → Convert subgraph to prose with ◆ markers
#   │    ✓ Use for: Building context
#   │    ⚙ Tune: max_triples for context size
#   │
#   ├─ build_agent_context()
#   │    → Build full agent context string (PRIMARY)
#   │    ✓ Use for: This is the context agents receive
#   │    ⚙ Tune: subgraph size, similarity k
#   │
#   ├─ build_context_minimal()
#   │    → Compact context for low-resource
#   │
#   └─ classify_failure_type()
#        → Classify failure: similarity_confusion|type_fit_gap|...

# llm_agents.py (400 lines)
#   ├─ LLMBackend (class)
#   │    → Wrapper for LLM inference
#   │    ✓ Use for: Initialize once, reuse
#   │    ⚙ Tune: model_id, quantize, temperature
#   │
#   ├─ AGENT_A_SYSTEM, AGENT_B_SYSTEM
#   │    → System prompts for agents
#   │    ✓ Modify: To change agent reasoning behavior
#   │    Key: A emphasizes type constraints, B emphasizes structure
#   │
#   ├─ agent_a(), agent_b()
#   │    → Run Agent A or B given context
#   │    ✓ Called by: run_parallel_staggered()
#   │
#   ├─ run_parallel_staggered()
#   │    → Run both agents in parallel with stagger
#   │    ✓ Use for: Primary agent execution
#   │    ⚙ Tune: stagger_delay for GPU scheduling
#   │
#   └─ compare_agent_predictions()
#        → Compare A vs B outputs for analysis

# scorer.py (450 lines)
#   ├─ verify_type_fit()
#   │    → Check Agent B claims are real (hallucination detection)
#   │    ✓ Called by: compute_quality_score()
#   │
#   ├─ verify_relations()
#   │    → Check Agent A claims are real
#   │    ✓ Called by: compute_quality_score()
#   │
#   ├─ compute_quality_score()
#   │    → Compute agent output quality (coverage × purity)
#   │    ✓ Use for: PRIMARY SCORING FUNCTION
#   │    Formula: quality = coverage * (1 - 0.5 * contamination)
#   │    ⚙ Tune: 0.5 multiplier for contamination weight
#   │
#   ├─ route_decision()
#   │    → Choose which agent to trust (A or B)
#   │    ✓ Use for: ROUTING DECISION (can customize)
#   │    ⚙ Tune: Quality thresholds for routing
#   │
#   ├─ get_routing_rationale()
#   │    → Generate explanation of routing choice
#   │
#   └─ aggregate_results()
#        → Combine A + B outputs into final answer
#        ✓ Use for: Final prediction + confidence

# pipeline.py (350 lines)
#   ├─ Pipeline (class)
#   │    → Main orchestrator with checkpointing
#   │    ✓ Use for: Resumable batch processing
#   │    Features: Checkpoint on every record, load from last
#   │
#   ├─ Pipeline.run_triple()
#   │    → Process single triple end-to-end
#   │    ✓ Use for: Understanding single example
#   │
#   ├─ Pipeline.run_batch()
#   │    → Process multiple triples
#   │    ✓ Use for: Processing val_hard or test set
#   │
#   ├─ Pipeline.write_results()
#   │    → Save results + summary JSON
#   │
#   ├─ run_full_pipeline()
#   │    → High-level API (RECOMMENDED)
#   │    ✓ Use for: PRIMARY INTERFACE
#   │    Returns: (results, summary)
#   │
#   └─ analyze_results(), export_analysis()
#        → Detailed result analysis per hop type, agent, etc.


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ QUICK LOOKUP BY PURPOSE                                                 │
# └─────────────────────────────────────────────────────────────────────────┘

# I want to...
# ─────────────────────────────────────────────────────────────────────────────

# ... load and preprocess data
#     Use: data_loader.load_codex_data()
#          data_loader.load_or_preprocess_triples()
#          data_loader.build_type_constraints()

# ... score a triple with the model
#     Use: model_utils.score_triple()
#          model_utils.get_full_ranking_filtered_batched()

# ... extract subgraph around entity
#     Use: kg_utils.extract_subgraph()
#          kg_utils.focused_subgraph()

# ... find similar entities
#     Use: embedding_utils.similarity_summary()

# ... build context for agent
#     Use: context_builder.build_agent_context()

# ... run Agent A
#     Use: llm_agents.agent_a()
#          llm_agents.run_parallel_staggered() (runs both A+B)

# ... run Agent B
#     Use: llm_agents.agent_b()
#          llm_agents.run_parallel_staggered() (runs both A+B)

# ... score agent output
#     Use: scorer.compute_quality_score()

# ... choose A or B
#     Use: scorer.route_decision()

# ... get final answer
#     Use: scorer.aggregate_results()

# ... process single triple
#     Use: pipeline.Pipeline.run_triple()

# ... process multiple triples
#     Use: pipeline.run_full_pipeline()

# ... analyze results
#     Use: pipeline.analyze_results()

# ... add/query memory
#     Use: memory_manager.MemoryManager


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ CUSTOMIZATION CHECKLIST                                                  │
# └─────────────────────────────────────────────────────────────────────────┘

# Change Agent A behavior:
#   → Edit: llm_agents.py → AGENT_A_SYSTEM
#   → Re-run: run_full_pipeline()

# Change routing logic:
#   → Edit: scorer.py → route_decision()
#   → Re-run: run_full_pipeline()

# Improve context quality:
#   → Edit: context_builder.py → build_agent_context()
#   → Adjust: max_triples, similarity k

# Adjust quality score:
#   → Edit: scorer.py → compute_quality_score()
#   → Change: multiplication factor (0.5)

# Use different LLM:
#   → Edit: llm_agents.py → LLMBackend()
#   → Change: model_id parameter

# Port to new dataset:
#   → Edit: data_loader.py → load_codex_data()
#   → Replace: with your dataset loader

# Add custom memory:
#   → Edit: memory_manager.py → MemoryManager.record_resolution()
#   → Add: custom storage logic

# Reduce context size:
#   → Edit: context_builder.py → build_subgraph_str(max_triples=6)
#   → Edit: llm_agents.py → similarity_summary(k=3)

# Increase batch size:
#   → Edit: model_utils.py → batch_size=1024
#   → Run: preprocessing again


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ FILE STATISTICS                                                          │
# └─────────────────────────────────────────────────────────────────────────┘

# data_loader.py          ~300 lines  Functions: 6
# model_utils.py          ~300 lines  Functions: 6
# kg_utils.py             ~350 lines  Functions: 5
# embedding_utils.py      ~200 lines  Functions: 5
# memory_manager.py       ~400 lines  Classes: 3, Functions: 5
# context_builder.py      ~300 lines  Functions: 6
# llm_agents.py           ~400 lines  Classes: 1, Functions: 6
# scorer.py               ~450 lines  Functions: 7
# pipeline.py             ~350 lines  Classes: 1, Functions: 4
# ─────────────────────────────────────────────────────────────────────────
# TOTAL                 ~2800 lines  ~50 functions, ~4 classes

# Documentation         ~800 lines
#   README.md           ~200 lines
#   SUMMARY.md          ~300 lines
#   API_REFERENCE.py    ~200 lines
#   EXAMPLE_USAGE.py    ~400 lines
#   PARAMETERS_GUIDE.md ~300 lines
#
# TOTAL WITH DOCS      ~3600 lines


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ HOW TO NAVIGATE THIS CODEBASE                                           │
# └─────────────────────────────────────────────────────────────────────────┘

# STEP 1: What is this pipeline?
#   → Read: SUMMARY.md (5 min)

# STEP 2: How do I use it?
#   → Read: EXAMPLE_USAGE.py (15 min)
#   → Run: python EXAMPLE_USAGE.py (30 min)

# STEP 3: How do I modify [component]?
#   → Find: module file (e.g., llm_agents.py)
#   → Read: docstrings at top of file
#   → Read: function docstrings
#   → Modify: the relevant function/constant
#   → Re-run: run_full_pipeline()

# STEP 4: How do I debug [issue]?
#   → Check: SUMMARY.md → Troubleshooting
#   → Check: PARAMETERS_GUIDE.md → relevant section
#   → Debug: manually in EXAMPLE_USAGE.py step 7

# STEP 5: How do I port to [dataset]?
#   → Read: PARAMETERS_GUIDE.md → Customization Checklist
#   → Edit: data_loader.py → load_codex_data()
#   → Run: rest of pipeline (should work!)


# ┌─────────────────────────────────────────────────────────────────────────┐
# │ KEY FILES TO MODIFY FOR COMMON TASKS                                     │
# └─────────────────────────────────────────────────────────────────────────┘

# Task                          File                          Function/Line
# ─────────────────────────────────────────────────────────────────────────
# Improve Agent A reasoning     llm_agents.py                 AGENT_A_SYSTEM
# Improve Agent B reasoning     llm_agents.py                 AGENT_B_SYSTEM
# Change routing logic          scorer.py                     route_decision()
# Adjust quality score          scorer.py                     compute_quality_score()
# Reduce context size           context_builder.py            build_agent_context()
# Increase model batch size     model_utils.py                batch_size=512
# Port to new dataset           data_loader.py                load_codex_data()
# Change temperature/model      llm_agents.py                 LLMBackend.__init__()
# Disable memory                pipeline.py                   run_full_pipeline(..., use_memory=False)
# Use LLM for aggregation       pipeline.py                   run_full_pipeline(..., use_llm_aggregation=True)
# Analyze results               pipeline.py                   analyze_results()
# ─────────────────────────────────────────────────────────────────────────


print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    CODEX-S MODULAR PIPELINE                               ║
║                                                                            ║
║  A fully documented, modularized Knowledge Graph reasoning pipeline       ║
║  easy to understand, modify, and port to other datasets.                  ║
║                                                                            ║
║  START HERE:                                                              ║
║    1. Read SUMMARY.md                                                     ║
║    2. Run EXAMPLE_USAGE.py                                                ║
║    3. Modify component of interest                                        ║
║    4. Re-run pipeline                                                     ║
║                                                                            ║
║  QUICK REFERENCE:                                                         ║
║    API_REFERENCE.py         Function signatures                          ║
║    PARAMETERS_GUIDE.md      Parameter tuning guide                       ║
║    This file               Module index                                   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
