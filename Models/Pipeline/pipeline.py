"""
MAIN PIPELINE MODULE
====================

Orchestrates the full CODEX-S reasoning pipeline end-to-end.

Workflow:
  1. Load/preprocess triples (Section 4 in original)
  2. Split into validation and held-out sets
  3. For each record:
     - Build context from record
     - Run agents A and B in parallel
     - Score outputs
     - Route decision
     - Aggregate
     - Save results
  4. Write summary statistics
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

from context_builder import build_agent_context
from llm_agents import LLMBackend, run_parallel_staggered
from scorer import compute_quality_score, route_decision, aggregate_results
from memory_manager import MemoryManager, get_memory_hint


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Main pipeline orchestrator.
    Manages checkpointing, memory, and result aggregation.
    """

    def __init__(
        self,
        checkpoint_dir: str = ".",
        use_memory: bool = True,
        use_llm_aggregation: bool = False,
    ):
        """
        Initialize pipeline.
        
        Args:
            checkpoint_dir (str): Directory for checkpoints and results
            use_memory (bool): Use episodic/semantic memory?
            use_llm_aggregation (bool): Use LLM for failure type classification?
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.checkpoint_file = self.checkpoint_dir / "val_hard_checkpoint.json"
        self.results_file = self.checkpoint_dir / "val_hard_results.json"
        self.summary_file = self.checkpoint_dir / "val_hard_summary.json"

        self.checkpoint = self._load_checkpoint()
        self.results = []
        self.use_memory = use_memory
        self.use_llm_aggregation = use_llm_aggregation

        if use_memory:
            self.memory_manager = MemoryManager(
                checkpoint_dir / "episodic.faiss",
                checkpoint_dir / "semantic.faiss",
                checkpoint_dir / "episodic_memory.tsv",
            )
        else:
            self.memory_manager = None

        print(f"[Pipeline] Checkpoint dir: {checkpoint_dir}")
        print(f"[Pipeline] Already processed: {len(self.checkpoint)}")

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint file or create new."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                cp = json.load(f)
                print(f"[Pipeline] Loaded checkpoint with {len(cp)} entries")
                return cp
        return {}

    def _save_checkpoint(self):
        """Persist checkpoint to disk."""
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.checkpoint, f, indent=2)

    def _triple_key(self, record: Dict) -> str:
        """Create unique key for triple."""
        return f"{record['head']}|{record['relation']}|{record.get('tail','?')}"

    def is_processed(self, record: Dict) -> bool:
        """Check if triple already processed."""
        return self._triple_key(record) in self.checkpoint

    def mark_processed(self, record: Dict, result: Dict):
        """Mark triple as processed and save checkpoint."""
        key = self._triple_key(record)
        self.checkpoint[key] = result
        self.results.append(result)
        self._save_checkpoint()

    def run_triple(
        self,
        record: Dict,
        llm_backend: LLMBackend,
        entity_to_id: Dict,
        relation_to_id: Dict,
        df_train=None,
        constraints=None,
    ) -> Optional[Dict]:
        """
        Process single triple through full pipeline.
        
        Args:
            record: Preprocessed record
            llm_backend: LLM instance
            entity_to_id: Entity ID mapping
            relation_to_id: Relation ID mapping
            df_train: Training data (for verification)
            constraints: Type constraints (for verification)
        
        Returns:
            dict with full reasoning trace, or None on error
        """
        key = self._triple_key(record)
        
        if self.is_processed(record):
            print(f"  Skipped (already processed)")
            return None

        true_tail = record.get("true_tail") or record.get("tail", "")

        print(f"\n{'='*55}\nPIPELINE  {key}\n"
              f"rank={record['true_rank']}  hop={record.get('hop_type','multi')}\n{'='*55}")

        try:
            # 1. BUILD CONTEXT
            episodic_hint = ""
            if self.memory_manager:
                mem_ctx = self.memory_manager.get_context_for_query(
                    record["head"], record["relation"]
                )
                episodic_hint = mem_ctx.get("episodic", "")

            context = build_agent_context(record, episodic_hint=episodic_hint)

            # 2. RUN AGENTS (parallel)
            a_out, b_out = run_parallel_staggered(
                record, context, llm_backend, episodic_hint=episodic_hint
            )

            # 3. SCORE OUTPUTS
            score_a = compute_quality_score(
                a_out, record, "A", df_ref=df_train, constraints=constraints
            )
            score_b = compute_quality_score(
                b_out, record, "B", df_ref=df_train, constraints=constraints
            )

            # 4. AGGREGATE
            agg = aggregate_results(
                a_out, b_out, score_a, score_b, record,
                llm_backend=llm_backend if self.use_llm_aggregation else None,
                use_llm_aggregation=self.use_llm_aggregation,
            )

            # 5. UPDATE MEMORY (if resolved correctly)
            if self.memory_manager and agg.get("final_answer", "").strip().lower() == true_tail.strip().lower():
                if agg.get("failure_type") == "resolved":
                    self.memory_manager.record_resolution(
                        record["head"],
                        record["relation"],
                        true_tail,
                        agg.get("chosen_agent", "?"),
                        agg.get("failure_type", "resolved"),
                        agg.get("selected_relations", []),
                    )
                    print("  [Memory] Recorded resolution")

            # 6. CHECK CORRECTNESS
            correct = (
                agg.get("final_answer", "").strip().lower()
                == true_tail.strip().lower()
            )

            print(f"\n{'✓' if correct else '✗'}  final={agg.get('final_answer')}  "
                  f"agent={agg.get('chosen_agent')}")

            # 7. BUILD RESULT
            result = {
                "triple": key,
                "true_tail": true_tail,
                "hop_type": record.get("hop_type", "multi"),
                "model_rank": record.get("true_rank", 99),
                "agent_a": a_out,
                "agent_b": b_out,
                "score_a": score_a,
                "score_b": score_b,
                "aggregator": agg,
                "final_correct": correct,
            }

            self.mark_processed(record, result)
            return result

        except Exception as e:
            print(f"  [ERROR] {e}")
            error_result = {"error": str(e), "triple": key}
            self.checkpoint[key] = error_result
            self._save_checkpoint()
            return None

    def run_batch(
        self,
        records: List[Dict],
        llm_backend: LLMBackend,
        entity_to_id: Dict,
        relation_to_id: Dict,
        df_train=None,
        constraints=None,
        delay_between_records: float = 1.0,
    ) -> List[Dict]:
        """
        Process multiple triples.
        
        Args:
            records: List of records
            llm_backend: LLM instance
            *: Other parameters from run_triple()
            delay_between_records: Delay between triples (to manage rate limits)
        
        Returns:
            List of results
        """
        remaining = [r for r in records if not self.is_processed(r)]
        print(f"[Pipeline] Processing {len(remaining)} / {len(records)} records")

        for i, record in enumerate(remaining):
            print(f"\n[{i+1}/{len(remaining)}]  {self._triple_key(record)}")
            
            self.run_triple(
                record, llm_backend,
                entity_to_id, relation_to_id,
                df_train=df_train,
                constraints=constraints,
            )

            if i < len(remaining) - 1:
                time.sleep(delay_between_records)

        return self.results

    def write_results(self):
        """Persist results and generate summary."""
        # Write results
        clean_results = [r for r in self.results if "error" not in r]
        with open(self.results_file, "w") as f:
            json.dump(clean_results, f, indent=2)

        # Generate summary
        errors = [r for r in self.results if "error" in r]
        correct = [r for r in clean_results if r.get("final_correct")]
        agents = [r["aggregator"].get("chosen_agent") for r in clean_results]
        ftypes = [r["aggregator"].get("failure_type") for r in clean_results]

        summary = {
            "total_processed": len(self.results),
            "errors": len(errors),
            "clean_results": len(clean_results),
            "correct": len(correct),
            "accuracy": 100 * len(correct) / max(len(clean_results), 1),
            "agent_choice": dict(Counter(agents).most_common()),
            "failure_types": dict(Counter(ftypes).most_common()),
        }

        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n{'='*55}  FINAL SUMMARY")
        print(f"Total processed: {len(self.results)}")
        print(f"Errors/skipped:  {len(errors)}")
        print(f"Clean results:   {len(clean_results)}")
        print(f"Correct:         {len(correct)} / {len(clean_results)}"
              f"  ({summary['accuracy']:.1f}%)")
        print(f"\nAgent chosen:")
        for agent, count in Counter(agents).most_common():
            print(f"  Agent {agent}: {count}")
        print(f"\nFailure types:")
        for ftype, count in Counter(ftypes).most_common():
            print(f"  {ftype}: {count}")
        print(f"\nFiles written:")
        print(f"  {self.results_file}")
        print(f"  {self.summary_file}")
        print(f"  {self.checkpoint_file}")

        return summary


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: HIGH-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    records: List[Dict],
    entity_to_id: Dict,
    relation_to_id: Dict,
    df_train=None,
    constraints=None,
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    checkpoint_dir: str = ".",
    use_memory: bool = True,
    delay_between_records: float = 1.0,
    use_llm_aggregation: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Run full CODEX-S pipeline end-to-end.
    
    Args:
        records: Preprocessed records to process
        entity_to_id: Entity ID mapping
        relation_to_id: Relation ID mapping
        df_train: Training dataframe (for verification)
        constraints: Type constraints dictionary
        llm_model: HuggingFace model ID
        checkpoint_dir: Where to save results
        use_memory: Use episodic/semantic memory?
        delay_between_records: Delay between records (seconds)
        use_llm_aggregation: Use LLM for failure type classification?
    
    Returns:
        Tuple of (results_list, summary_dict)
    
    Example:
        results, summary = run_full_pipeline(
            records=val_hard_records,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            df_train=df_train,
            constraints=constraints,
            checkpoint_dir="output/",
            use_memory=True,
        )
        
        print(f"Accuracy: {summary['accuracy']:.1f}%")
    """
    # Initialize LLM
    print("[Pipeline] Initializing LLM...")
    llm = LLMBackend(model_id=llm_model, quantize=True)

    # Initialize pipeline
    pipeline = Pipeline(
        checkpoint_dir=checkpoint_dir,
        use_memory=use_memory,
        use_llm_aggregation=use_llm_aggregation,
    )

    # Process records
    print("[Pipeline] Starting processing loop...")
    pipeline.run_batch(
        records,
        llm,
        entity_to_id,
        relation_to_id,
        df_train=df_train,
        constraints=constraints,
        delay_between_records=delay_between_records,
    )

    # Write outputs
    summary = pipeline.write_results()

    return pipeline.results, summary


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RESULT ANALYSIS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_results(results: List[Dict]) -> Dict:
    """
    Detailed analysis of pipeline results.
    
    Args:
        results: List of result dicts
    
    Returns:
        dict with detailed breakdowns
    
    Analysis:
        - Accuracy overall
        - Accuracy by hop type
        - Accuracy by failure type
        - Agent performance
        - Common error patterns
    """
    clean = [r for r in results if "error" not in r]

    # Overall
    correct = sum(1 for r in clean if r.get("final_correct"))

    # By hop type
    hop_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in clean:
        hop = r.get("hop_type", "unknown")
        hop_stats[hop]["total"] += 1
        if r.get("final_correct"):
            hop_stats[hop]["correct"] += 1

    # By failure type
    ftype_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in clean:
        ftype = r["aggregator"].get("failure_type", "unknown")
        ftype_stats[ftype]["total"] += 1
        if r.get("final_correct"):
            ftype_stats[ftype]["correct"] += 1

    # Agent performance
    agent_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in clean:
        agent = r["aggregator"].get("chosen_agent", "unknown")
        agent_stats[agent]["total"] += 1
        if r.get("final_correct"):
            agent_stats[agent]["correct"] += 1

    return {
        "overall": {
            "total": len(clean),
            "correct": correct,
            "accuracy": 100 * correct / max(len(clean), 1),
        },
        "by_hop": {
            k: {
                "accuracy": 100 * v["correct"] / max(v["total"], 1),
                **v
            }
            for k, v in hop_stats.items()
        },
        "by_failure_type": {
            k: {
                "accuracy": 100 * v["correct"] / max(v["total"], 1),
                **v
            }
            for k, v in ftype_stats.items()
        },
        "by_agent": {
            k: {
                "accuracy": 100 * v["correct"] / max(v["total"], 1),
                **v
            }
            for k, v in agent_stats.items()
        },
    }


def export_analysis(results: List[Dict], output_path: str):
    """Export detailed analysis to JSON."""
    analysis = analyze_results(results)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[Pipeline] Analysis saved to {output_path}")
