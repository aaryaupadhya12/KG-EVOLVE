We propose Knowledge Graph Episodic Memory (KGEM), a continual learning framework for LLM-based routing agents where failure diagnoses are encoded as typed knowledge graph triples rather than unstructured text.

To enable scalable retrieval over growing memory, we augment the graph with a vector index over compressed graph summaries, allowing efficient access without altering the underlying symbolic representation.

Unlike weight-space continual learning methods that suffer from catastrophic forgetting, our graph substrate accumulates structured failure memory persistently across deployment cycles. LLM agents reason over compact (~150-token) graph summaries and output typed triples, converting symbolic reasoning into structured supervision for RotatE embeddings.

A lightweight meta-learner distills agent experience into sub-millisecond inference without graph access at runtime.

Validated on the Nations dataset, we show that MRR improves monotonically across outer-loop cycles and that failure vector stability across random seeds is a more reliable model selection criterion than MRR alone — establishing typed graph writeback as a scalable and effective memory substrate for continual agentic learning.


A structured reasoning evanluation dataset comparing 2 agents on KG completion tasks, wiht fioine grained signals about reasoning quality , noise and missing information 


Updated wrt to Agentic Mmemory 

The KG usually is a static store. Our dynamic episodic memory that groows from Agent experience. We in this minimal dataset see KG link predictions , Their system has 0 memory - every query starts cold , Our accumaltion of relation triples across runs and injects that prior into future reasoning. Their agents do not disagree - there is only one reasoner , we have 2 structurally different agents who disagreemnt is the signal 

The deepest differentce is teh use of PRM-800K as a dataset to build the static KG , we use it as a methadology - we use teacher student distillaton paraddigm where quality labels aer produced knowing the correct answers and enabling a learned scored that operates wihtout it at inference.

We are here to demostrate that a quality scorrer aka a aggregator to be able to reason over thje mathematical prodedures over the qulity scores from the static KG itself , enabling a student - teacher distaillation training loop that produces quality labels without any answer leakage at inference , directly instatiating the PRM-8OOk methadology to a new domain that is the a reasoning domain we stand at the intesreaction of Agentic memory and LLm reeasning 

We dont only baseline RotatE acoss multiple seeds , we had a system that runs without Mmeory KG-RAR where a LLM anotates it ones and we see if the new KG become sbetter  which is called as Cold start and then The delta between that and our system that is the Meemory augmented KG (MAKG) is our contribiution as the paper explcitly says that active resonin or dynamic KG is what is required 

Through abilating this with the Nations dataset and then PRM-800K we have a s olida foundation , we will onyle vlaaute them based on a subset of maximum 1k examples as its a tedious process to run them forever and we would loose compute , if the hypothesis stands for PRM-800K susbset a full study will be created :) 


Upadate changes to the POST-HOC evaluation metrics are as follows:
Correctness in knowledge graph link prediction is insufficient to evaluate reasoning quality. We hypothesize that a post-hoc reasoning metric (PACS) can distinguish between spurious and grounded predictions, and that this signal can be learned and used to improve inference-time decision making
Mainly an easy way for us to qunitfy where agent are able to guess the correct answer but not able to show the correct path that they 
traced.

As KG size increases , path availability increaes but path relaiability decreses , necassitating path quality reasoning rather than path existance based reasoning. 
Symmetry - ambiguit score , symmetric relation measured and present in the CODEX 
CoDEx-S → 17.46% symmetric
CoDEx-M → ~4%
CoDEx-L → ~3%
The more symmetruc the dataset its more ambigoius , large dataset = less symmetric and more directional structure 

Composiotinal Paths 
Your findings:
Dataset	Path coverage	Confidence
CoDEx-S	10%	0.63
CoDEx-M	16%	0.56
CoDEx-L	31%	0.46

New Findings that RotatE is not the best method for CODEX-S its ComplEX which is a Convolutional based KGE extractor adn from the finidings is that we need get rank > 10 and not like 4 or 3 like in nations as this dataset basic benchmarks excepts it to be near 10 not above that :)

