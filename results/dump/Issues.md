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
We need to look into posthoc methods on how to qunotfy the Lucky rate and the pass@k metric and how that works into the workflow and change inot the Posthoc HYpothesis - check the issues -md for the future logs - Aarya Upadhy