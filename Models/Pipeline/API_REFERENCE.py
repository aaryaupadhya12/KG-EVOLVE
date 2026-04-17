"""
QUICK API REFERENCE
===================

One-page reference for all public functions in the modularized pipeline.
Use Ctrl+F to find what you need quickly.
"""

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING (data_loader.py)
# ═════════════════════════════════════════════════════════════════════════════

# CODEX-S Data Loading
def load_codex_data(size="s", code="en"):
    """Load CODEX-S dataset with English labels and entity/relation ID mappings."""
    # Returns: (df_train, df_test, df_valid, df_all, entity_to_id, relation_to_id, 
    #           id_to_entity, id_to_relation)

def build_graph_from_df(df):
    """Convert dataframe to adjacency list graph {entity: [(relation, tail), ...]}"""

def build_type_constraints(df):
    """Compute statistical type constraints for entities/relations from training data."""
    # Returns: {
    #   "rel_head_to_ranked_tails": {(r, h): {t: prob}},
    #   "rel_to_tail_dist": {r: {t: prob}},
    #   "rel_to_tail_counts": {r: {t: count}},
    #   "rel_to_head_counts": {r: {h: count}}
    # }

def get_type_constraint_signal(head, relation, true_tail, predicted, constraints):
    """Get type fit scores for true vs predicted entities (for agent context)."""
    # Returns: {"type_fit_true": 0.3, "type_fit_pred": 0.1, "type_gap": 0.2, ...}

def preprocess_all_triples(df, split_name, entity_to_id, relation_to_id, ...):
    """Score ALL triples with model — SLOW! Run once, cache to JSON."""
    # Returns: List of record dicts with scor rankings, subgraphs, etc.

def load_or_preprocess_triples(df_split, split_name, output_file, force=False, **kwargs):
    """Load cached JSON or compute if not exists."""

def load_preprocessed_records(filepath):
    """Quick load of cached preprocessed records."""


# ═════════════════════════════════════════════════════════════════════════════
# MODEL INFERENCE (model_utils.py)
# ═════════════════════════════════════════════════════════════════════════════

def load_pretrained_model(repo_id=None, filename="trained_pipeline.pkl"):
    """Load ComplEx model from HuggingFace. Cached globally."""

def get_model():
    """Get already-loaded model."""

def score_triple(head_id, relation_id, tail_id, model=None):
    """Score single triple."""
    # Returns: float (higher = more likely to be true)

def score_batch(triples, model=None):
    """Score multiple triples in batch."""
    # Returns: list of scores

def get_full_ranking_filtered_batched(head, relation, true_tail, entity_to_id, 
                                       relation_to_id, id_to_entity, batch_size=512):
    """Rank ALL candidate entities for a (head, relation) query."""
    # Returns: {
    #   "predicted": "top_entity",
    #   "predicted_score": 0.92,
    #   "true_tail": "ground_truth",
    #   "true_score": 0.85,
    #   "true_rank": 42,
    #   "full_ranking": [(entity, score), ...]
    # }

def get_entity_embeddings(model=None):
    """Extract entity embeddings tensor for similarity."""
    # Returns: tensor of shape [num_entities, embedding_dim]

def get_relation_embeddings(model=None):
    """Extract relation embeddings (rarely used)."""


# ═════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH UTILITIES (kg_utils.py)
# ═════════════════════════════════════════════════════════════════════════════

def extract_subgraph(entity, graph, hops=2, max_triples=150):
    """BFS extraction of k-hop neighborhood around entity."""
    # Returns: [(head, relation, tail), ...]

def focused_subgraph(entities, graph, hops=2, max_triples=100, query_relation=None):
    """Extract and prioritize subgraph for multiple entities (tier-ranked)."""
    # Returns: [(head, relation, tail), ...] (tier-ranked)

def hop_classifier(head, tail, graph, target_relation=None):
    """Classify path type and distance between entities."""
    # Returns: (hop_type="single|multi|none", distance=1.0|1.5|2.0|99, evidence)

def failure_summary(head, relation, true_tail, predicted_tail, constraints, score_fn=None):
    """Diagnose why model predicted wrong."""
    # Returns: (summary_text, analysis_dict)

def get_type_constraint_signal(head, relation, true_tail, predicted, constraints):
    """Compute type fit signals (moved from data_loader for kg utilities)."""


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS & SIMILARITY (embedding_utils.py)
# ═════════════════════════════════════════════════════════════════════════════

def get_entity_embeddings_cached(model):
    """Extract and globally cache entity embeddings."""

def similarity_summary(entity, entity_to_id, id_to_entity, embeddings, df_train=None, k=5):
    """Find k most similar entities by embedding + relation overlap."""
    # Returns: (summary_string, [(entity, sim_score, shared_rels, rel_score), ...])

def get_entity_relations(entity, df):
    """Find all relations an entity participates in."""
    # Returns: set of relations

def embedding_distance(entity1, entity2, entity_to_id, embeddings):
    """Cosine similarity between two entities."""
    # Returns: float in [0, 1]

def find_similar_cluster(entities, entity_to_id, id_to_entity, embeddings, k=10):
    """Find entity cluster similar to a set of entities."""
    # Returns: [(entity, avg_similarity), ...]


# ═════════════════════════════════════════════════════════════════════════════
# MEMORY MANAGEMENT (memory_manager.py)
# ═════════════════════════════════════════════════════════════════════════════

class FAISSMemory:
    """Vector store for similarity memory (episodic/semantic)."""
    def query(text, k=2):
        """Find k most similar stored patterns."""
    def add(text):
        """Add new pattern to memory."""
    def add_batch(texts):
        """Add multiple patterns."""

class TSVMemory:
    """Lightweight structured memory (head -> [(relation, tail), ...])."""
    def get_hint(head, relation=None, k=5):
        """Get past examples for (head, relation) pair."""
    def add(head, relation, tail):
        """Record successful triple."""
    def save():
        """Persist to disk."""

class MemoryManager:
    """High-level memory interface combining episodic, semantic, TSV."""
    def get_context_for_query(head, relation, failure_type=None):
        """Get all memory hints before agent reasoning."""
    def record_resolution(head, relation, tail, agent, failure_type, key_relations):
        """Record successful resolution to memory."""
    def save_all():
        """Persist all memory stores."""

# Legacy compatibility
def load_tsv_memory(path="episodic_memory.tsv"):
    """Load TSV memory into dict."""

def get_memory_hint(head, memory, relation=None, k=5):
    """Get hint from TSV memory dict (legacy)."""


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDING (context_builder.py)
# ═════════════════════════════════════════════════════════════════════════════

def trim_subgraph(subgraph, head, true_tail, predicted, only_tail_has, max_triples=12):
    """Tier and truncate subgraph for agent context."""
    # Returns: trimmed subgraph (tier-ranked)

def build_subgraph_str(record, max_triples=12):
    """Convert subgraph to readable prose with ◆ markers."""
    # Returns: multi-line string

def build_agent_context(record, tsv_memory=None, episodic_hint=""):
    """Build comprehensive context string for agent reasoning."""
    # Returns: formatted prose context

def build_context_minimal(record):
    """Build compact context (for low-resource environments)."""

def build_agent_context_with_memory(record, memory_manager):
    """Build context using unified memory manager."""

def classify_failure_type(record):
    """Classify failure type: similarity_confusion | structural_gap | type_fit_gap etc."""
    # Returns: failure_type string


# ═════════════════════════════════════════════════════════════════════════════
# LLM AGENTS (llm_agents.py)
# ═════════════════════════════════════════════════════════════════════════════

class LLMBackend:
    """Wrapper for LLM inference with quantization support."""
    def __init__(model_id, quantize=True, temperature=0.3):
        """Initialize LLM backend."""
    def call(system, user, max_retries=2):
        """Call LLM and return response."""

def parse_json_output(raw, agent_name):
    """Parse JSON from LLM output, handle markdown blocks."""

def agent_a(context, llm_backend, episodic_hint=""):
    """Run Agent A (type-constraint reasoning)."""
    # Returns: {"prediction": "entity", "confidence": 0.75, "shared_relations": [...], ...}

def agent_b(context, llm_backend, episodic_hint=""):
    """Run Agent B (structural reasoning)."""
    # Returns: {"prediction": "entity", "confidence": 0.85, "key_relations": [...], "path_found": "...", ...}

def run_parallel_staggered(record, context, llm_backend, episodic_hint="", stagger_delay=1.0):
    """Run Agent A and B in parallel with stagger."""
    # Returns: (agent_a_output, agent_b_output)

def compare_agent_predictions(a_out, b_out):
    """Compare Agent A and B predictions."""
    # Returns: {"same_prediction": bool, "confidence_gap": float, "agreement_strength": str, ...}


# ═════════════════════════════════════════════════════════════════════════════
# SCORING & AGGREGATION (scorer.py)
# ═════════════════════════════════════════════════════════════════════════════

def verify_type_fit(agent_out, constraints):
    """Verify Agent B's cited relations are real (hallucination detection)."""
    # Returns: {"verified": bool, "hallucinated": [...], "confirmed": [...]}

def verify_relations(claimed_rels, head, tail, df_ref):
    """Verify Agent A's shared relations claim."""
    # Returns: {"verified": [...], "hallucinated": [...], "rate": float}

def compute_quality_score(agent_out, record, agent_name, df_ref=None, constraints=None):
    """Compute quality_score for agent output."""
    # Returns: {
    #   "quality_score": 0.65,  # coverage * (1 - 0.5 * contamination)
    #   "contamination": 0.2,   # false positives
    #   "overlap_tail": [...],   # cited good relations
    #   "overlap_pred": [...],   # cited bad relations
    #   "prediction_correct": bool,
    #   ...
    # }

def route_decision(score_a, score_b, record):
    """Route to Agent A or B based on quality scores."""
    # Returns: "A" or "B"

def get_routing_rationale(score_a, score_b, a_out, b_out, chosen, record):
    """Generate explanation of routing decision."""
    # Returns: one-sentence string

def aggregate_results(a_out, b_out, score_a, score_b, record, llm_backend=None, 
                      use_llm_aggregation=False):
    """Aggregate A + B outputs into final answer."""
    # Returns: {
    #   "final_answer": "entity",
    #   "chosen_agent": "A",
    #   "confidence": 0.75,
    #   "reason": "...",
    #   "failure_type": "resolved|similarity_confusion|..."
    # }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE (pipeline.py)
# ═════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """Main orchestrator with checkpointing and memory management."""
    def __init__(checkpoint_dir, use_memory=True, use_llm_aggregation=False):
        """Initialize pipeline."""
    def is_processed(record):
        """Check if triple already processed (resumable)."""
    def run_triple(record, llm_backend, entity_to_id, relation_to_id, ...):
        """Process single triple through full pipeline."""
    def run_batch(records, llm_backend, ...):
        """Process multiple triples."""
    def write_results():
        """Persist results and generate summary."""

def run_full_pipeline(records, entity_to_id, relation_to_id, df_train=None, 
                      constraints=None, llm_model="Qwen/Qwen2.5-7B-Instruct", 
                      checkpoint_dir=".", use_memory=True, ...):
    """High-level API: run full pipeline end-to-end."""
    # Returns: (results, summary)

def analyze_results(results):
    """Detailed analysis of pipeline results."""
    # Returns: {
    #   "overall": {"accuracy": 75.5},
    #   "by_hop": {"single": {"accuracy": 80}, "multi": {"accuracy": 70}},
    #   "by_failure_type": {...},
    #   "by_agent": {"A": {"accuracy": 78}, "B": {"accuracy": 72}}
    # }

def export_analysis(results, output_path):
    """Export analysis to JSON."""


# ═════════════════════════════════════════════════════════════════════════════
# QUICK LOOKUP TABLE
# ═════════════════════════════════════════════════════════════════════════════

# Want to...                                  | Use this function
# ────────────────────────────────────────────────────────────────────────────
# Load CODEX-S data                           | load_codex_data()
# Score a triple with the model               | score_triple()
# Rank all tail entities                      | get_full_ranking_filtered_batched()
# Get subgraph around entity                  | extract_subgraph()
# Find similar entities                       | similarity_summary()
# Get type constraints for a query            | get_type_constraint_signal()
# Build context for agents                    | build_agent_context()
# Run Agent A                                 | agent_a()
# Run Agent B                                 | agent_b()
# Verify agent claims                         | verify_type_fit(), verify_relations()
# Compute quality score                       | compute_quality_score()
# Route to A or B                             | route_decision()
# Process one triple end-to-end               | Pipeline.run_triple()
# Process multiple triples                    | run_full_pipeline()
# Analyze results                             | analyze_results()
# Load/save episodic memory                   | MemoryManager
