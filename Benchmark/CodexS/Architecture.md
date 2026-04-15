This is to mainly talk about the LibKGE configuration and mainly Understand how to reproduce the results as well 
The main components that are explained here are from the YAML file to Run CODEX-S 
The other 2 datasets that are used Use Pykeen but the experimentations are mainly close world datasets and hence we pivoted towards CODEX

1. Model Architecture : ComplEX, wrapped inside a reciprocal model
Reciprocal model learns a inverse relation not only (France,capital,Paris) but alse (Paris,capital,France), this doubles the training triples and then boosts the MRR 

2. Embeddings: entity and relation gets 512 dimenional complex valued vector , intialization using Xavier normal 

3. Complex Specific settings : entity -> Dropout and regurlization -> made to almost 0 same with relational embeddings settings 
These exact values came from their automated hyperparameter search (Ax)

4. 1vsAll: For each training triple (s, p, o), the model scores the correct object o against all 2034 entities simultaneously. Much more efficient than random negative samplin

5. Computes Hits@1, @3, @10, @50... — what % of the time is the correct entity in the top K predictions
rounded_mean_rank: when multiple entities have the same score (tie), uses the average of their ranks
relation_type breakdown: reports metrics separately for 1-to-1, 1-to-many, many-to-1, many-to-many relations

6.  Seeds are not fixed, so each training run will give slightly different results. To get fully reproducible results, change all -1 to the same number e.g. 42.