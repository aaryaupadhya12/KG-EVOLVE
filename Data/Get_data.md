In root 
 Get-ChildItem "C:\Users\aaryaupadhya\Documents\Aarya\NLP\codex\kge\data\codex-s"


# Write it to a new Folder to add the dataset to Kaggle 
New-Item -ItemType Directory -Force -Path "C:\Users\aaryaupadhya\Documents\Aarya\NLP\codex\kaggle_upload"
>>
>> # The 3 essential files
>> Copy-Item "kge\data\codex-s\entity_ids.del"   "kaggle_upload\entity_ids.del"
>> Copy-Item "kge\data\codex-s\relation_ids.del" "kaggle_upload\relation_ids.del"
>> Copy-Item "kge\local\experiments\20260414-032055-config\checkpoint_best.pt" "kaggle_upload\winner_model.pt"
>>
>> # All the pickle index files (speeds up evaluation massively)
>> Copy-Item "kge\data\codex-s\index-train_sp_to_o.pckl"  "kaggle_upload\"
>> Copy-Item "kge\data\codex-s\index-train_po_to_s.pckl"  "kaggle_upload\"
>> Copy-Item "kge\data\codex-s\index-valid_sp_to_o.pckl"  "kaggle_upload\"
>> Copy-Item "kge\data\codex-s\index-valid_po_to_s.pckl"  "kaggle_upload\"
>> Copy-Item "kge\data\codex-s\index-test_sp_to_o.pckl"   "kaggle_upload\"
>> Copy-Item "kge\data\codex-s\index-test_po_to_s.pckl"   "kaggle_upload\"
>> Copy-Item "kge\data\codex-s\index-relation_types.pckl"  "kaggle_upload\"