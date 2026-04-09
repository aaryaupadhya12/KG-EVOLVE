
# Training Phase 

Input Dataset
     ↓
3× Seeded RotatE Models
(select best / ensemble)
     ↓
Failure Detection (rank > threshold)
     ↓
Agent A (similarity) + Agent B (path reasoning)
     ↓
Agent AGG (teacher / oracle using true answer)
     ↓
Episodic Memory (JSON)
(rich structured supervision)
     ↓
Distillation Layer
(extract patterns, signals, features)
     ↓
Gated Memory Model (Encoder-only, e.g., BERT)
(train as reasoning scorer)



# Inference Phase 

Input (h, r, ?)
     ↓
RotatE → top-K candidates
     ↓
Retrieval Agent
(fetch relevant patterns / signals)
     ↓
Agent C (BERT scorer)
(score candidates using learned patterns)
     ↓
Agent A + Agent B (optional, lightweight)
     ↓
Final Aggregator
(combine A, B, C)
     ↓
Final Prediction