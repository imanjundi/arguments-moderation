# Arguments Moderation

## Getting Started
install requirements  
`pip install -r requirements.txt`  
create `data` folder and download data  
`(mkdir data && cd data && wget --content-disposition https://scholarship.law.cornell.edu/context/ceri/article/1019/type/native/viewcontent)`

## Training
You can call the training bash scripts and pass arguments to overwrite the default parameters e.g.  
`moderator_intervention_full/train_any_moderation.sh --learning_rate 7e-06 --comment_parents_num 2`

## Hyperparameter Tuning using Weights & Biases
First, init a sweep  
`wandb sweep moderation_intervention_full/any_moderation_sweep_config.yaml`  
or  
`wandb sweep moderation_intervention_full/quality_moderation_sweep_config.yaml`  
Then launch agents (copy id from initialization output)  
`export SWEEP_PROJECT=name-of-project && wandb agent sweep-id-from-init`