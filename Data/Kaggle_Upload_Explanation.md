winner_model.pt
The actual trained ComplEx model. Contains the learned embedding vectors for all 2034 entities and 42 relations. This is the brain — everything else just helps you use it correctly.

The ID Mapping Files
These are the most critical files after the model itself. LibKGE assigned every entity and relation an integer index during preprocessing. These files tell you which integer maps to which Wikidata ID.

entity_ids.delindex → Wikidata QID for all 2034 entities0	Q100
relation_ids.delindex → Wikidata PID for all 42 relations0	P17

Without these the 0 in the embedding matrix mean nothing with these we understand the index 0 means something 

The Triple Files
![alt text](image.png)

The Picke Files
![alt text](image-1.png)


Precomputed Pickle Lookups for faster infernece lookups 
![alt text](image-2.png)