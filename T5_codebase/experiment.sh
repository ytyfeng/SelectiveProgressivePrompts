#!/bin/bash

# To run this experiment, you need to unzip the amazon zip file in datasets directory first

# similarity_threshold = 0.7 -> selective progressive prompts
python train_t5_cl.py --task_list imdb amazon \
--select_k_per_class 1000 \
--lr 0.3 --num_epochs 50 \
--freeze_weights 1 --prefix_len 10 \
--model_name t5-large --early_stopping 1 \
--save_name T5_experiment_sim_07 --save_dir save_dir \
--batch_size 4 --similarity_threshold 0.7 > sim07.txt

# similarity_threshold = 0 -> progressive prompts
python train_t5_cl.py --task_list imdb amazon \
--select_k_per_class 1000 \
--lr 0.3 --num_epochs 50 \
--freeze_weights 1 --prefix_len 10 \
--model_name t5-large --early_stopping 1 \
--save_name T5_experiment_sim_0 --save_dir save_dir \
--batch_size 4 --similarity_threshold 0 > sim0.txt

# nonprogressive per-task prompt tuning
python train_t5_cl.py --task_list imdb amazon \
--select_k_per_class 1000 \
--lr 0.3 --num_epochs 100 \
--freeze_weights 1 --prefix_len 10 \
--model_name t5-large --early_stopping 1 \
--save_name T5_experiment_nonprogressive --save_dir save_dir \
--progressive 0
--batch_size 4 --similarity_threshold 0 > nonprogressive.txt