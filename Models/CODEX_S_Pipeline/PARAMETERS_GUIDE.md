"""
DETAILED PARAMETER GUIDE
========================

Complete parameter reference for all major functions.
When upgrading to CODEX-M/L or other datasets, adjust parameters here.
"""

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
load_codex_data(size="s", code="en"):
  
  Parameters:
    size (str): Dataset size
      "s" = CODEX-S (small, ~9k entities)
      "m" = CODEX-M (medium, ~300k entities)
      "l" = CODEX-L (large, ~5M entities)
      
      For CODEX-M/L:
        - Preprocessing will take MUCH longer
        - Consider distributed scoring
        - Subgraph extraction will be slower
      
    code (str): Language for entity/relation labels
      "en" = English labels (default)
      Other codes available: "fr", "de", "es", etc.
  
  Impact on rest of pipeline:
    - size affects entity_to_id length → model scoring speed
    - Different datasets require different ComplEx models
    - Type constraints scale with entity count
"""

"""
build_type_constraints(df):
  
  Output: Contains statistical priors for entities/relations
  
  Key fields:
    - rel_head_to_ranked_tails: (relation, head) → {tail: prob}
      Specific: For this head + relation, what tails are common?
      Used for type_fit computation in Agent A reasoning
      
    - rel_to_tail_dist: relation → {tail: prob}
      General: For any head + this relation, tail distribution
      Fallback when specific (head, relation) not in training
      
  Tuning:
    - If type_fit too sparse: lower thresholds in scoring
    - If too common (all 0.5+): increase agent confidence thresholds
    - For rare relations: rely more on structural reasoning (Agent B)
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: MODEL INFERENCE PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
get_full_ranking_filtered_batched(head, relation, true_tail, 
                                   entity_to_id, relation_to_id, 
                                   id_to_entity, batch_size=512):
  
  Parameters:
    batch_size (int): Triples per GPU batch
      512  = default (good for 8GB VRAM)
      256  = if running out of memory
      1024 = if memory available (faster)
      
      Tune based on your GPU:
        RTX 3090: try batch_size=1024+
        V100: try batch_size=512-768
        T4: keep at 256-512
    
    true_rank (int): Output rank of ground truth
      Used to detect "hard" failures (rank > percentile 15)
      Higher rank = harder to predict
      Triples with rank > 1000 are very challenging
  
  Performance notes:
    - Score all ~9k entities takes ~1-2 seconds per triple
    - Preprocessing 5000 triples: ~2-3 hours on GPU
    - Consider parallel execution with careful GPU memory management
"""

"""
get_entity_embeddings(model):
  
  Output: Tensor of shape [num_entities, embedding_dim]
  
  For CODEX-S: [9000, 256] (or similar)
  For CODEX-M: [300k, 256] (larger cache, ~150MB)
  For CODEX-L: [5M, 256] (cache this to disk)
  
  Usage:
    - Cache ONCE during preprocessing
    - Reuse for all similarity_summary() calls
    - Don't extract per-entity (very slow)
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: GRAPH EXTRACTION PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
extract_subgraph(entity, graph, hops=2, max_triples=150):
  
  Parameters:
    hops (int): Maximum neighborhood depth
      1 = direct neighbors only
      2 = neighbors of neighbors (default, good balance)
      3 = third-order neighbors (sparser, more spread)
      
      For multi-hop queries: increase to 3
      For single-hop: keep at 1-2
    
    max_triples (int): Maximum triples to extract
      50  = tight budget (low-resource models)
      100 = default
      150 = generous (good for rich context)
      300 = very generous (watch LLM context overflow)
      
      If max_triples hit before max_hops:
      - Stops extraction early
      - BFS ensures closest neighbors prioritized
  
  Tuning for different queries:
    - Single-hop: extract_subgraph(..., hops=1, max_triples=50)
    - Multi-hop: extract_subgraph(..., hops=3, max_triples=150)
    - Dense relations: use lower hops (more repetition)
"""

"""
focused_subgraph(entities, graph, hops=2, max_triples=100, 
                  query_relation=None):
  
  Parameters:
    entities (list): Core entities of interest [head, tail, predicted]
    query_relation (str): If provided, rank these first
      "hasChild" → prioritize triples with hasChild relation
      None → use tier-2 (both in entities) as priority
  
  Tier System Output:
    TIER 1: Triples with query_relation (gold signal)
            ~10-20 triples if query_relation found
    TIER 2: Triples between important entities
            ~20-30 triples
    TIER 3: Triples touching entities (1-hop)
            ~40-50 triples
    TIER 4: All other triples
            fill to max_triples
  
  When to use focused vs extract:
    - Use focused: when building context (3 entities at play)
    - Use extract: when examining single entity neighborhood
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: EMBEDDING SIMILARITY PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
similarity_summary(entity, entity_to_id, id_to_entity, embeddings, 
                   df_train=None, k=5):
  
  Parameters:
    k (int): Number of neighbors to return
      3 = brief summary (for compact context)
      5 = default
      10 = verbose (watch context window)
      
    df_train (df): Optional for relational overlap scoring
      If provided: compute shared_relations + rel_score
      If None: just embedding similarity
      
      Computing relational overlap adds ~100ms per entity
      Skip if tight on latency
  
  Output format:
    "Alice most similar to: Bob(sim=0.89,shared=3,rel=0.67), ..."
    
    Each entry: (name, sim_score[0-1], shared_rel_count, rel_jaccard[0-1])
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: AGENT CONTEXT PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
build_agent_context(record, tsv_memory=None, episodic_hint=""):
  
  Building Blocks:
    1. Query statement: "(head, relation, ?)"
    2. Separating relations: "only_true_has" (primary signal)
    3. Memory block: past examples (episodic)
    4. Subgraph: local graph with ◆ markers
    5. Similarity: embedding neighbors
    6. Metadata: hop_type, failure diagnosis
  
  Size Management:
    Average context: ~500-1000 tokens
    With memory: +200-300 tokens
    
    If too large: disable memory, trim subgraph
    If too small: confidence scores might be low
    
    LLM context window:
      Qwen 7B: 128k tokens (not issue here)
      Llama2 7B: 4k tokens (watch out!)
  
  Tuning:
    build_subgraph_str(record, max_triples=6)  → compact
    build_subgraph_str(record, max_triples=15) → verbose
"""

"""
trim_subgraph(subgraph, head, true_tail, predicted, 
              only_tail_has, max_triples=12):
  
  Tier Priority:
    TIER 1: Separating relations (highest value)
            Relations only true_tail has
            Discriminating signal
            
    TIER 2: Entitynodes in question
            Triples touching true_tail or predicted
            
    TIER 3: Query head
            Triples from head
            
    TIER 4: Background
            Everything else
  
  Marker System:
    "Alice --hasChild--> Bob ◆" = tier 1 (gold signal)
    "Alice --spouse--> David"    = tier 2
    "David --age--> 45"          = tier 3
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: LLM AGENT PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
LLMBackend(model_id="Qwen/Qwen2.5-7B-Instruct", 
           quantize=True, temperature=0.3):
  
  Parameters:
    model_id (str): HuggingFace model ID
      Recommended models:
        "Qwen/Qwen2.5-7B-Instruct"    = default (balanced)
        "Llama-2-7B-chat-hf"          = open source alternative
        "mistral-7B-instruct"         = fast+capable
        "Qwen/Qwen2.5-14B-Instruct"   = stronger (more memory)
      
      For other datasets:
        - Same agents work for Nations, UMLS, FB15k
        - May need to retune prompts/confidence
        - Smaller models (3-4B): lower quality
        - Larger models (13B+): better but slower
    
    quantize (bool): 4-bit quantization
      True  = fits on 8GB VRAM
      False = needs 16GB+ VRAM
               but slightly better quality
      
    temperature (float): Sampling temperature
      0.0 = deterministic (always same output)
      0.3 = default (low randomness, good for reasoning)
      0.7 = creative (high randomness)
      
      For reasoning: use 0.2-0.4
      For diversity: use 0.6-0.9
"""

"""
run_parallel_staggered(record, context, llm_backend, 
                       episodic_hint="", stagger_delay=1.0):
  
  Parallelization:
    - Agent A and B run in threads
    - Stagger delay avoids simultaneous GPU requests
    - Default stagger_delay=1.0 sec works well
    
    Tuning:
      stagger_delay=0.0 → both agents run immediately (GPU overload risk)
      stagger_delay=2.0 → safer but slower
      stagger_delay=0.5 → aggressive (if GPU has bandwidth)
  
  Output:
    Both outputs guaranteed to arrive (synced at return)
    If one fails: exception propagates
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: SCORING PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
compute_quality_score(agent_out, record, agent_name, 
                      df_ref=None, constraints=None):
  
  Quality Score Formula:
    quality = coverage * (1 - 0.5 * contamination)
    
    coverage = (overlapping "only_true_has" relations) / total "only_true_has"
    contamination = (overlapping "only_pred_has" relations) / agent_relations_cited
    
  Tuning Quality Score:
    Increase contamination penalty: multiply 0.5 by larger value (e.g., 0.7)
      = stricter against false positives
    Decrease: multiply by 0.1-0.3
      = more forgiving, higher scores overall
    
  Threshold Interpretation:
    quality > 0.7  = agent cite strong discriminating relations
    quality 0.3-0.7 = mixed signal
    quality < 0.3  = mostly noise, low confidence
"""

"""
route_decision(score_a, score_b, record):
  
  Routing Rules (current implementation):
    1. If A quality > 0.5 && B <= 0.5 → choose A
    2. If B quality > 0.5 && A <= 0.5 → choose B
    3. If both/neither above 0.5:
       → choose agent with lower contamination
    4. Tie: single-hop → A,  multi-hop → B
  
  Customization Examples:
    
    # Always prefer A for type constraints
    if score_a["quality_score"] > 0.1:
        return "A"
    
    # Ensemble: only if consensus
    if a_out["prediction"] == b_out["prediction"]:
        return chosen
    else:
        return None  # no consensus
    
    # Weighted: custom formula
    total = score_a["quality_score"] + score_b["quality_score"]
    prob_a = score_a["quality_score"] / total
    return "A" if random.random() < prob_a else "B"
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: MEMORY PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
MemoryManager(episodic_path="episodic.faiss",
              semantic_path="semantic.faiss", 
              tsv_path="episodic_memory.tsv"):
  
  Memory Types:
    
    episodic: Specific past examples
      "(head, relation) → tail | agent A | resolved"
      Queried with fuzzy similarity search
      Good for: finding analogous past cases
      
    semantic: Failure type patterns
      "Agent B wins on structural_gap | key_rels: spouse, livesIn"
      Higher level abstraction
      Good for: understanding systematic strengths
      
    tsv: Structured quick lookup
      head → [(relation, tail), ...]
      Exact matching, no similarity
      Good for: fast hints before LLM call
  
  When to Use:
    Use episodic: when you want LLM to see past reasoning
    Use semantic: when you want abstract patterns
    Use tsv: always (fast, lightweight)
  
  Recording Patterns:
    Modify record_resolution() to store custom data
    Example: store confidence delta (predicted - true)
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: PIPELINE ORCHESTRATION PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════

"""
run_full_pipeline(records, entity_to_id, relation_to_id,
                  df_train=None, constraints=None,
                  llm_model="Qwen/Qwen2.5-7B-Instruct",
                  checkpoint_dir=".",
                  use_memory=True,
                  delay_between_records=1.0,
                  use_llm_aggregation=True):
  
  Parameters:
    records (list): Preprocessed records to process
      Example: val_hard (hard failure subset)
      Size: 10-10000 records (test on small first)
    
    checkpoint_dir (str): Where to save results + resumable checkpoint
      Creates: val_hard_results.json, val_hard_summary.json, etc.
      Resumable: if interrupted, run again same arguments
    
    use_memory (bool): Enable episodic/semantic memory
      True  = slower but learns from past
      False = stateless (useful for baseline)
    
    delay_between_records (float): Seconds between triples
      1.0   = default (safe for rate limits)
      0.5   = faster (for local testing)
      2.0   = conservative (avoid overloading LLM service)
    
    use_llm_aggregation (bool): Use LLM to classify failure type
      True  = expensive (extra LLM calls)
             gives detailed failure classification
      False = heuristic (fast)
             simpler failure types
  
  Recommended Configs:
    
    # Baseline (quick, no memory)
    run_full_pipeline(records, ..., use_memory=False, use_llm_aggregation=False)
    
    # Full (slow, everything enabled)
    run_full_pipeline(records, ..., use_memory=True, use_llm_aggregation=True)
    
    # Balanced (good default)
    run_full_pipeline(records, ..., use_memory=True, use_llm_aggregation=False)
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: CUSTOMIZATION CHECKLIST FOR NEW DATASETS
# ═════════════════════════════════════════════════════════════════════════════

"""
When porting to CODEX-M, CODEX-L, Nations, UMLS, etc:

☐ 1. DATA LOADING
    Modify: load_codex_data()
    Change: CODEX-S → CODEX-M/L loader (or Nations/UMLS)
    Keep: entity_to_id, relation_to_id, entity_embeddings patterns
    
☐ 2. MODEL
    Retrain: ComplEx model on target dataset
    Or: Use existing pretrained model (if available)
    Check: entity count matches (affects batch sizing)
    
☐ 3. PREPROCESSING
    Adjust: batch_size in get_full_ranking_filtered_batched()
    If dataset larger: batch_size=256 instead of 512
    Consider: Time (preprocessing very expensive for large datasets)
    
☐ 4. AGENT PROMPTS
    Optionally: Retune AGENT_A_SYSTEM, AGENT_B_SYSTEM for domain
    Example: Add medical knowledge for UMLS
    Example: Domain-specific relation types for specialized KGs
    
☐ 5. TYPE CONSTRAINTS
    Automatically: Recomputed from new df_train
    Monitor: How sparse/dense type distributions are
    Adjust: Confidence thresholds if needed
    
☐ 6. ROUTING LOGIC
    Test: Does route_decision() work on new data?
    Analyze: Agent choice distribution (should be varied)
    Adjust: If one agent always wins (may be overfitting)
    
☐ 7. EVALUATION
    Split: Val/held-out on new dataset
    Baseline: Evaluate agents on val_hard
    Analyze: accuracy breakdown by hop type, relation type
    Report: Final held-out accuracy (different from val if hyperparameter tuned)
    
☐ 8. MEMORY (Optional)
    Retrain: Or start fresh (memory initialized empty)
    Reuse: Previous memory from different dataset (not recommended)
    Monitor: Memory size (TSV grows with each success)
    
☐ 9. PERFORMANCE
    Profile: Which component is slowest?
    Bottleneck likely: model scoring (batch_size tuning helps)
    LLM inference: usually 30-50% of time (fixed cost)
"""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11: HYPERPARAMETER TUNING GRID
# ═════════════════════════════════════════════════════════════════════════════

"""
To improve Agent A accuracy:

A1. PROMPT TUNING
    - Increase emphasis on type_fit in AGENT_A_SYSTEM
    - Add examples of good reasoning
    - Clarify what "shared_relations" means
    
A2. FEATURE TUNING  
    - Increase max_triples in subgraph (more context)
    - Increase k in similarity_summary (more neighbors)
    - Add relation frequency in type constraints

A3. CONFIDENCE CALIBRATION
    - Current rule: type_fit match → 0.60-0.80
    - Adjust bounds if A systematically under/over-confident


To improve Agent B accuracy:

B1. PROMPT TUNING
    - Add instructions: "Prioritize paths in the subgraph"
    - Clarify path matching: "path_found must come from provided subgraph"
    - Example multi-hop reasoning

B2. FEATURE TUNING
    - Increase hops in extract_subgraph (deeper exploration)
    - Increase max_triples (more paths visible)
    - Add relation frequency weights

B3. CONFIDENCE CALIBRATION
    - Current rule: path evidence → 0.60-0.80
    - Adjust if B systematically miscalibrated


To improve routing:

R1. ADJUST THRESHOLDS
    Current: quality > 0.5 → choose agent
    Try:     quality > 0.4 (more lenient)
    Try:     quality > 0.6 (stricter)

R2. CHANGE CONTAMINATION WEIGHT
    Current: 1 - 0.5 * contamination
    Try:     1 - 0.3 * contamination (forgive noise)
    Try:     1 - 0.7 * contamination (penalize noise)

R3. ENSEMBLE VOTING
    Instead: just pick higher quality
    Try: require consensus (both must agree)
"""
