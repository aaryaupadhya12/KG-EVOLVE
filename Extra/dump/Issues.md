15:46 - 23-03-2026
The Agent B is skewed towards The higher degreee nodes hence causing Higher amount of bias to such nodes 
The Agent prompts must be changed so that the lower degrees nodes must be trusted with higher amount of prompting to help with lower hallucinations     
-- Fixed 

22:26 - 24-03-2026
Boosted routing vs learned reward model over reasnoning traces -- Take a look at this we reasoned why it wont work 


23:09 - 24-03-2026
Policy-level differences in agentic reasoning can be learned and optimized via a learned reasoning critic. - this is the real thing and look into how to solve this with PRM800K 

27-03-2026
There are errors in the Aggregator agent due to to the monlithic code structure finding it hard to correct , mainly not resoaning it out rpoperly hence needs some error correction in the inference time as welll and working on that fix right now has very low chances of succeeding , needs a tuned prompt 
Problem only existing during inference and seems fine during training need to enrichen the aggregator shcema or allow it have more space 

28-03-2026
The data set delta is 1.9% hence the dataset is proven that its the best that it can improve, SO we move forawrd to the FG15K link precition pred where the dataset is higher and then we go into deeper with the PRM based aggregator


1=04-2026
We need to look into posthoc methods on how to qunotfy the Lucky rate and the pass@k metric and how that works into the workflow and change inot the Posthoc HYpothesis - check the issues -md for the future logs - Aarya Upadhya

1-04-2026
We hypothesize that different LLM families specialize in distinct reasoning regimes, where instruction-tuned models excel at similarity-based inference but hallucinate under structural constraints, while reasoning-optimized models produce more faithful multi-hop reasoning with reduced hallucination
-- The abilation should include hte working of MOE , Qwen instruct alone and then Deepseek R1 code isntruct alone and then PRM instroduction into the model so that tabilations are complete 


2-04-2026
The major probelsm in memorization in lower datasets look into slower warmups to help us move forward and understnd how to lower the data without the increase in the memorization of the dataset

3-04-2026
Another major bug that was fixed on the LLm editor was the agent B hallucination they where makingup paths not even related to head to solve problem which causes naunce in the files and hence a path verifer a deterministc agent that is just a Pythons script was entered so that no hallucinated paths are taken and LLm drift is quantified 

04-04-2026
There is might be problem in how the episodic memory is being used we need to fix this and then the other main problem is that we might need to use co-attentaion as theyre might be masking problem and we need to fix that , make changes to the user template to force the usuage of the Epsidoic hints sent 


05-04-2026
We need to tracj the weighted lucky score as the problesm dont get segregated equally and Agent A need not be the the actual culprit and as Agent B gets more routed towards it based on the Grounded score that we created 

2) We need to recreated the KGC values for the COEDEX-S meet the SOTA that they claim so taht we have the right priors right now diw to low KGE MRR the mode is collapsing to the LLm weights to solve the answer and would still get ut correct as Qwen was trained on the wikipedia dataset and hence knows the answer and hence we will reduce the weights on the LLm as well. 

06-04-26
The main problem is that the agents are not enfrocing the episodic memory during inferece stage and hence completly collapsing to the paramteric weights of the LLm and that has to be changed else its going to be a Hige problem this was seen by the pass@k values and hence that should be the main goal to check during the next steps that we take 

2) The inference stage just has to be Static that dynamic and has to just abide the rules that is the KGE prior union the tsv memory and this should be added with the path verifier and then be grounded for the next runs this is how it must be moved forward. 


12-04-2026
3) Increased the K to 50 in UMLS as the entites are higher as Top-10 makes it very easier for the model as we want to make the model reason out more -- 