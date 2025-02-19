# Code for ARR paper: 'ALSC: A Localized Solution for Secure Cloud Large Language Model Interactions in Open-Ended Question Answering'


## Requirements

torch >= 2.0.0
transformers
ms-swift
deepspeed

## Description
code/
---finetune_hide.sh: bash script to fully finetune the hide model from open-source Qwen2_0.5B with ms-swift
---finetune_recover.sh: bash script to fully finetune the recover model from open-source Qwen2_1.5B with ms-swift
---eval/: directory for inference and evaluation script.
-----eval_sensitive.py: use sensitive model to detect sensitive words within a user query and evaluate the Precision and Recall according to the ground truth.
-----eval_hide.py: use detected sensitive words and local hide model to generate substitution pairs.
-----sub_text_withpair.py: use pairs to replace sensitive information in user query.
-----func.py: text obfuscation functions.
-----eval_recover.py: recover cloud LLM response with local recover model.
-----eval_generation.py: evaluate recovered response quality according to the ground-truth response.
-----util.py: utility functions.

