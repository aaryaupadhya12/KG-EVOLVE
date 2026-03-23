We propose Knowledge Graph Episodic Memory (KGEM), a continual learning framework for LLM-based routing agents where failure diagnoses are encoded as typed knowledge graph triples rather than unstructured text.

To enable scalable retrieval over growing memory, we augment the graph with a vector index over compressed graph summaries, allowing efficient access without altering the underlying symbolic representation.

Unlike weight-space continual learning methods that suffer from catastrophic forgetting, our graph substrate accumulates structured failure memory persistently across deployment cycles. LLM agents reason over compact (~150-token) graph summaries and output typed triples, converting symbolic reasoning into structured supervision for RotatE embeddings.

A lightweight meta-learner distills agent experience into sub-millisecond inference without graph access at runtime.

Validated on the Nations dataset, we show that MRR improves monotonically across outer-loop cycles and that failure vector stability across random seeds is a more reliable model selection criterion than MRR alone — establishing typed graph writeback as a scalable and effective memory substrate for continual agentic learning.