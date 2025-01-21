# Code
- synthetic dataset generation
    - Synthetic_dataset_generation.ipynb

- inference code : code for inference your model (greedy generation)
    - inference_code.py

- LLM_as_Judge
    - LLM_as_Judge.ipynb

- data
    - msmarco_samples.jsonl (before synthetic dataset generation, it just has 20 retrieved passages for each)
    - dpo_train.jsonl (Synthetic dataset, it has chosen, rejected, and input (contains question, context))

- evaluation_data
    - nq_open_40_single.jsonl : the dataset which has only one question related knowledge segment in 40 retrieved passages
    - nq_open_40_multiple.jsonl : the dataset which has multiple question related knowledge segment in 40 retrieved passages
    
- dpo code : to train model with our generated synthetic dataset
    - OpenRLHF/train_dpo.sh : DPO training code