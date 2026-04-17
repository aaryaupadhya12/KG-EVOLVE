"""
MEMORY MANAGER MODULE
=====================

Manages episodic and semantic memory stores for iterative learning.

Two types of memory:
  1. EPISODIC: (head, relation, tail) patterns from confirmed correct answers
     → Consulted before agent reasoning as "prior" examples
     
  2. SEMANTIC: Failure type patterns (which agent wins on which types)
     → Consulted to understand systematic agent strengths
     
  3. TSV MEMORY: Structured triple patterns for quick lookup
     → Fast retrieval without vector DB for simple cases

Key Responsibilities:
  1. Load FAISS vector stores for episodic/semantic memory
  2. Query similar past examples before reasoning
  3. Write new patterns after successful resolution
"""

import os
import csv
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: FAISS VECTOR STORES
# ─────────────────────────────────────────────────────────────────────────────

class FAISSMemory:
    """
    Wrapper around FAISS vector store for similarity memory.
    Supports both episodic (past examples) and semantic (patterns) memory.
    """

    def __init__(self, store_path: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize or load FAISS vector store.
        
        Args:
            store_path (str): Path to save/load FAISS index
            embedding_model (str): Embedding model from HuggingFace Hub
        """
        if not FAISS_AVAILABLE:
            print("[Memory] FAISS not installed — memory disabled")
            self.store = None
            return

        self.store_path = store_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        if os.path.exists(store_path):
            self.store = FAISS.load_local(
                store_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"[Memory] Loaded {store_path}")
        else:
            # Create new store with dummy init text
            self.store = FAISS.from_texts(["init"], self.embeddings)
            self.store.save_local(store_path)
            print(f"[Memory] Created {store_path}")

    def query(self, text: str, k: int = 2) -> str:
        """
        Find k most similar stored texts.
        
        Args:
            text (str): Query text
            k (int): Number of neighbors
        
        Returns:
            str: Concatenated documents, filtered to exclude "init" placeholder
        """
        if self.store is None:
            return ""

        docs = self.store.similarity_search(text, k=k)
        # Filter out initialization placeholder
        results = [d.page_content for d in docs if "init" not in d.page_content]
        return "\n".join(results)

    def add(self, text: str):
        """Add new text to memory and persist."""
        if self.store is None:
            return

        self.store.add_texts([text])
        self.store.save_local(self.store_path)

    def add_batch(self, texts: List[str]):
        """Add multiple texts at once."""
        if self.store is None:
            return

        if texts:
            self.store.add_texts(texts)
            self.store.save_local(self.store_path)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TSV MEMORY (Fast Lookup)
# ─────────────────────────────────────────────────────────────────────────────

class TSVMemory:
    """
    Lightweight structured memory stored in TSV format.
    Fast lookup: head -> [(relation, tail), ...]
    
    Used to provide hints before agent reasoning:
    "Hey, we've seen this (head, relation) pair before — here's what worked"
    """

    def __init__(self, path: str = "episodic_memory.tsv"):
        """
        Initialize or load TSV memory.
        
        Args:
            path (str): Path to TSV file
        """
        self.path = path
        self.memory = self._load()

    def _load(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load TSV file into memory dict."""
        memory = defaultdict(list)
        
        if not os.path.exists(self.path):
            return memory

        with open(self.path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                return memory
            
            for row in reader:
                head = row.get("head", "").strip()
                relation = row.get("relation", "").strip()
                tail = row.get("tail", "").strip()
                
                if head and relation and tail:
                    memory[head].append((relation, tail))

        return memory

    def get_hint(
        self, 
        head: str, 
        relation: Optional[str] = None, 
        k: int = 5
    ) -> str:
        """
        Get memory hint for a (head, relation) pair.
        
        Args:
            head (str): Entity head
            relation (str): Relation to prioritize (if any)
            k (int): Max triples to return
        
        Returns:
            str: Formatted triple list for agent context
        
        Example:
            hint = memory.get_hint("Alice", "hasChild")
            # "Alice --hasChild--> Bob\nAlice --hasChild--> Carol"
        """
        triples = self.memory.get(head, [])
        
        if not triples:
            return ""

        # Prioritize matching relation
        if relation:
            same = [(r, t) for r, t in triples if r == relation]
            rest = [(r, t) for r, t in triples if r != relation]
            selected = same[:k] + rest[:max(0, k - len(same))]
        else:
            selected = triples[:k]

        # Remove duplicates
        seen, unique = set(), []
        for r, t in selected:
            if (r, t) not in seen:
                seen.add((r, t))
                unique.append((r, t))

        return "\n".join(f"{head} --{r}--> {t}" for r, t in unique)

    def add(self, head: str, relation: str, tail: str):
        """Record successful resolution."""
        if (relation, tail) not in self.memory[head]:
            self.memory[head].append((relation, tail))

    def save(self):
        """Persist memory to TSV file."""
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["head", "relation", "tail"])
            
            for head, triples in self.memory.items():
                for relation, tail in triples:
                    writer.writerow([head, relation, tail])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: UNIFIED MEMORY MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    High-level interface for all memory operations.
    Combines episodic (FAISS), semantic (FAISS), and TSV stores.
    """

    def __init__(
        self,
        episodic_path: str = "episodic.faiss",
        semantic_path: str = "semantic.faiss",
        tsv_path: str = "episodic_memory.tsv",
    ):
        """
        Initialize memory manager.
        
        Args:
            episodic_path: FAISS store for (h, r, t) patterns
            semantic_path: FAISS store for failure type patterns
            tsv_path: TSV file for quick lookup
        """
        self.episodic = FAISSMemory(episodic_path)
        self.semantic = FAISSMemory(semantic_path)
        self.tsv = TSVMemory(tsv_path)

    def get_context_for_query(
        self, 
        head: str, 
        relation: str, 
        failure_type: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Get all relevant memory context before agent reasoning.
        
        Args:
            head (str): Query head
            relation (str): Query relation
            failure_type (str): Classified failure type (optional)
        
        Returns:
            dict with:
            - "episodic": Similar (h, r, t) examples
            - "semantic": Patterns for this failure type
            - "tsv": Known triples with this head
        
        Example:
            context = memory.get_context_for_query(
                "Alice", 
                "hasChild", 
                "type_fit_gap"
            )
            
            print(context["episodic"])
            # "Alice hasChild → Bob | agent A | resolved
            #  Alice hasChild → Carol | agent B | resolved"
        """
        return {
            "episodic": self.episodic.query(f"{head} {relation}", k=2),
            "semantic": self.semantic.query(failure_type or relation, k=2) if failure_type else "",
            "tsv": self.tsv.get_hint(head, relation, k=5),
        }

    def record_resolution(
        self,
        head: str,
        relation: str,
        tail: str,
        agent: str,
        failure_type: str,
        key_relations: List[str],
    ):
        """
        Record successful resolution to memory.
        Called after agent correctly predicts tail.
        
        Args:
            head (str): Head entity
            relation (str): Query relation
            tail (str): Correct tail
            agent (str): Which agent (A or B) resolved it
            failure_type (str): Classified failure type
            key_relations (list): Relations cited for resolution
        """
        # Episodic: specific example
        episodic_text = f"{head} {relation} → {tail} | agent {agent} | {failure_type}"
        self.episodic.add(episodic_text)

        # Semantic: pattern
        semantic_text = f"agent {agent} wins on {failure_type} | key_rels: {', '.join(key_relations)}"
        self.semantic.add(semantic_text)

        # TSV: for quick lookup
        self.tsv.add(head, relation, tail)
        self.tsv.save()

    def save_all(self):
        """Persist all memory stores."""
        self.tsv.save()
        # FAISS stores auto-save on add, but can force here


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LEGACY INTERFACE (Backward Compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv_memory(path: str = "episodic_memory.tsv") -> Dict[str, List[Tuple[str, str]]]:
    """
    Quick load TSV memory into dict.
    Legacy function for backward compatibility.
    
    Returns:
        dict: {head: [(relation, tail), ...]}
    """
    tsv = TSVMemory(path)
    return dict(tsv.memory)


def get_memory_hint(
    head: str, 
    memory: Dict[str, List[Tuple[str, str]]], 
    relation: Optional[str] = None, 
    k: int = 5
) -> str:
    """
    Get hint from TSV memory dict.
    Legacy function for backward compatibility.
    """
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
            seen.add((r, t))
            unique.append((r, t))

    return "\n".join(f"{head} --{r}--> {t}" for r, t in unique)
