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