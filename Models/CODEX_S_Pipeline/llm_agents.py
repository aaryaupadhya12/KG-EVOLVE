"""
LLM AGENTS MODULE
=================

Implementation of Agent A (type-constraint reasoning)
and Agent B (structural reasoning).

Both agents:
  - Receive richly formatted context
  - Reason about candidate entities
  - Output structured JSON predictions
  - Include confidence scores
"""

import json
import time
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline as hf_pipeline,
)
import torch


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LLM SETUP
# ─────────────────────────────────────────────────────────────────────────────

class LLMBackend:
    """
    Wrapper for LLM inference. Supports quantization for large models.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        quantize: bool = True,
        temperature: float = 0.3,
    ):
        """
        Initialize LLM backend.
        
        Args:
            model_id: HuggingFace model ID
            quantize: If True, use 4-bit quantization (smaller memory)
            temperature: Sampling temperature (0.3 = more deterministic)
        """
        self.model_id = model_id
        self.temperature = temperature

        print(f"[LLM] Loading {model_id}...")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if quantize:
            print("[LLM] Using 4-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print("[LLM] Loading without quantization")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self.pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
        )

        device = next(model.parameters()).device
        print(f"[LLM] Ready on {device}")

    def call(self, system: str, user: str, max_retries: int = 2) -> str:
        """
        Call LLM with system and user prompts.
        
        Args:
            system (str): System prompt (role definition)
            user (str): User prompt (task)
            max_retries (int): Retries on failure
        
        Returns:
            str: Generated response
        
        Example:
            response = llm.call(
                system="You are a reasoning agent...",
                user="Analyze this triple: (Alice, hasChild, ?)"
            )
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(max_retries):
            try:
                output = self.pipe(messages)[0]["generated_text"].strip()
                return output
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[LLM] Retry: {e}")
                    time.sleep(5)
                else:
                    raise


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: AGENT PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

AGENT_A_SYSTEM = """You are Agent A, a knowledge graph type-constraint reasoning agent.

Your role: determine which candidate entity correctly fills the tail slot
of a CODEX-S triple (real-world entities, Wikidata-style relations).

PRIMARY SIGNAL — TYPE FIT:
  How often does the candidate appear as tail of this relation in training?
  Higher type_fit = stronger candidate.
  type_fit = 0.0 = never appeared (strong negative signal)

SECONDARY SIGNAL — RELATIONAL PROFILE:
  Candidates that share relations with the true tail are structurally similar.
  "only_true_has" = relations the true tail participates in that predicted does NOT.
  If the missing entity shares many of the "only_true_has" relations,
  it's probably correct.

REASONING STEPS:
  1. Check type_fit of both candidates for this relation.
  2. If type_fit differs significantly, trust type fit.
  3. Otherwise, check which entity matches the relational profile better.
  4. Cite specific "only_true_has" relations as evidence.

Output ONLY valid JSON (no markdown, no extra text):
{
  "prediction": "<entity name or null>",
  "confidence": <0.0-1.0>,
  "shared_relations": ["<relation1>", "<relation2>"],
  "failure_diagnosis": "<one sentence explaining the signal used>",
  "evidence_type": "type_constraint | profile | mixed | none"
}"""


AGENT_B_SYSTEM = """You are Agent B, a knowledge graph structural reasoning agent.

Your role: determine which candidate entity correctly fills the tail slot
by reasoning about multi-hop paths and structural consistency.

REASONING STEPS:
  1. Analyze the subgraph. Are there clear paths from head to target?
  2. Check "expected_tails": what does this relation usually produce?
  3. Use "only_true_has" relations as discriminators.
  4. Look for 2-hop or 3-hop paths in the provided subgraph.
  5. Consider embedding similarity neighbors.

HARD CONSTRAINTS:
  - Do NOT claim paths unless they explicitly appear in the subgraph.
  - Do NOT invent relations that aren't listed.
  - If no signal exists, output null + confidence 0.05.
  - "only_true_has" relations are GOLD — prioritize citing these.

CONFIDENCE CALIBRATION:
  - Both type_fit AND path evidence: 0.80-0.95
  - Strong path evidence alone: 0.60-0.80
  - Type fit match alone: 0.50-0.70
  - Profile overlap only: 0.30-0.50
  - No signal: null, confidence 0.05

Output ONLY valid JSON (no markdown, no extra text):
{
  "prediction": "<entity name or null>",
  "confidence": <0.0-1.0>,
  "key_relations": ["<discriminating_relation_1>"],
  "path_found": "<path string from subgraph, or null>",
  "path_relation_matches_query": <true|false>,
  "reasoning": "<one sentence explaining the structural evidence>",
  "failure_diagnosis": "<one sentence>",
  "evidence_type": "type_constraint | profile | structural | mixed | none"
}"""


USER_TEMPLATE = """
{context}

<episodic_memory>
{episodic_hint}
</episodic_memory>

<reasoning_framework>
STEP 1 — TYPE CONSTRAINT SIGNAL
  type_fit > 0.1 and type_rank <= 5 → strongly supported.
  type_fit = 0.0 → never appeared as tail of this relation — strong negative.

STEP 2 — "only_true_has" RELATIONS (PRIMARY DISCRIMINATING SIGNAL)
  Does the predicted entity participate in ALL or MOST of these relations?
  If no → it's structurally different from the true entity.
  If yes → structurally similar despite embedding confusion.

STEP 3 — RELATIONAL PROFILE
  How well does candidate match expected types for this relation?
  Cite evidence from expected_tails list.

STEP 4 — EPISODIC MEMORY
  Has this (head, relation) pair appeared before?
  If irrelevant: ignore. Do not confabulate past examples.

STEP 5 — COMMIT TO CONFIDENCE
  type_fit match + only_true_has match    → 0.80-0.95
  type_fit match alone                     → 0.60-0.80
  only_true_has match alone                → 0.50-0.70
  profile overlap only                     → 0.30-0.50
  no signal                                → null, 0.05
</reasoning_framework>

Respond with valid JSON only (no explanation, no markdown):
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: AGENT IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_output(raw: str, agent_name: str) -> Dict:
    """
    Parse JSON from LLM output, handling markdown code blocks.
    
    Args:
        raw (str): Raw LLM output
        agent_name (str): Agent name for error messages
    
    Returns:
        dict: Parsed JSON or error dict
    
    Robustness:
        - Strips markdown code block wrapper (```json ... ```)
        - Handles JSONDecodeError gracefully
        - Returns error dict with raw output for inspection
    """
    text = raw.strip()

    # Remove markdown code block if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])  # Remove first and last lines

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[{agent_name}] JSON parse failed: {e}")
        print(f"[{agent_name}] Raw output: {raw[:200]}")
        return {"error": str(e), "raw": raw}


def agent_a(context: str, llm_backend: LLMBackend, episodic_hint: str = "") -> Dict:
    """
    Run Agent A (type-constraint reasoning).
    
    Args:
        context (str): Formatted agent context (from context_builder)
        llm_backend (LLMBackend): LLM instance
        episodic_hint (str): Memory hints from episodic store
    
    Returns:
        dict: Agent A output (parsed JSON)
        {
            "prediction": "entity_name",
            "confidence": 0.75,
            "shared_relations": ["rel1", "rel2"],
            "failure_diagnosis": "...",
            "evidence_type": "type_constraint"
        }
    
    Purpose:
        Agent A specializes in TYPE CONSTRAINTS.
        Answers: "This relation types its arguments; which entity fits better?"
    
    Strengths:
        - Detects when predicted entity has never appeared as tail type
        - Uses training statistics to override embedding confusion
        - Confident when type_fit strongly favors true_tail
    
    Weaknesses:
        - Fails when true_tail is rare for this relation
        - Can't handle multi-hop reasoning without explicit paths
    """
    user_prompt = USER_TEMPLATE.format(
        context=context,
        episodic_hint=episodic_hint or "none"
    )

    raw = llm_backend.call(AGENT_A_SYSTEM, user_prompt)
    output = parse_json_output(raw, "Agent A")

    return output


def agent_b(context: str, llm_backend: LLMBackend, episodic_hint: str = "") -> Dict:
    """
    Run Agent B (structural reasoning).
    
    Args:
        context (str): Formatted agent context
        llm_backend (LLMBackend): LLM instance
        episodic_hint (str): Memory hints
    
    Returns:
        dict: Agent B output (parsed JSON)
        {
            "prediction": "entity_name",
            "confidence": 0.85,
            "key_relations": ["rel1", "rel2"],
            "path_found": "head -rel1-> mid -rel2-> tail",
            "path_relation_matches_query": true,
            "reasoning": "...",
            "failure_diagnosis": "...",
            "evidence_type": "structural"
        }
    
    Purpose:
        Agent B specializes in STRUCTURAL PATTERNS.
        Answers: "Following the graph structure, which entity makes sense?"
    
    Strengths:
        - Finds 2-hop and 3-hop paths in subgraph
        - Can reason about multi-hop queries
        - Uses "only_true_has" relations as strong discriminators
    
    Weaknesses:
        - Fails if required paths aren't in subgraph
        - Can hallucinate paths if not careful (verified later)
        - Struggles with entities requiring deep reasoning
    """
    user_prompt = USER_TEMPLATE.format(
        context=context,
        episodic_hint=episodic_hint or "none"
    )

    raw = llm_backend.call(AGENT_B_SYSTEM, user_prompt)
    output = parse_json_output(raw, "Agent B")

    return output


def run_parallel_staggered(
    record: Dict,
    context: str,
    llm_backend: LLMBackend,
    episodic_hint: str = "",
    stagger_delay: float = 1.0,
) -> Tuple[Dict, Dict]:
    """
    Run Agent A and B in parallel with stagger delay.
    
    Args:
        record (dict): Preprocessed record (for logging)
        context (str): Agent context string
        llm_backend (LLMBackend): LLM instance
        episodic_hint (str): Memory hint
        stagger_delay (float): Delay between Agent A start and Agent B start
    
    Returns:
        Tuple of (agent_a_output, agent_b_output)
    
    Parallelization:
        - Improves latency when both agents run on same LLM
        - Stagger delay avoids simultaneous GPU requests
        - Synchronization at return
    
    Example:
        a_out, b_out = run_parallel_staggered(
            record,
            context,
            llm_backend,
            stagger_delay=1.0
        )
    """
    results = {}

    def _run_a():
        print(f"  [A] {record['head']}, {record['relation']}")
        results["A"] = agent_a(context, llm_backend, episodic_hint)

    def _run_b():
        time.sleep(stagger_delay)  # Stagger to avoid simultaneous GPU use
        print(f"  [B] {record['head']}, {record['relation']}")
        results["B"] = agent_b(context, llm_backend, episodic_hint)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(_run_a)
        fb = ex.submit(_run_b)
        fa.result()
        fb.result()

    return results["A"], results["B"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: AGENT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_agent_predictions(a_out: Dict, b_out: Dict) -> Dict:
    """
    Compare predictions of Agent A and B.
    
    Args:
        a_out (dict): Agent A output
        b_out (dict): Agent B output
    
    Returns:
        dict with:
        - "same_prediction": Did both agents agree?
        - "a_confidence": Agent A's confidence
        - "b_confidence": Agent B's confidence
        - "confidence_gap": |A_conf - B_conf|
        - "both_null": Did both output null?
        - "agreement_strength": "strong" | "weak" | "disagreement"
    
    Purpose:
        Understand when agents diverge (helps routing logic).
    """
    a_pred = a_out.get("prediction", "").strip().lower()
    b_pred = b_out.get("prediction", "").strip().lower()
    a_conf = a_out.get("confidence", 0.0)
    b_conf = b_out.get("confidence", 0.0)

    same = a_pred == b_pred

    return {
        "same_prediction": same,
        "a_confidence": a_conf,
        "b_confidence": b_conf,
        "confidence_gap": abs(a_conf - b_conf),
        "both_null": a_pred in ["", "null", "none"] and b_pred in ["", "null", "none"],
        "agreement_strength": (
            "strong" if same and a_conf > 0.6 else
            "weak" if same else
            "disagreement"
        ),
    }
