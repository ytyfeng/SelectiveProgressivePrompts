#!/bin/bash

# similarity_threshold = 0.7
python train_t5_cl.py --task_list imdb cb sst2 \
--select_k_per_class 1000 \
--lr 0.3 --num_epochs 3 \
--freeze_weights 1 --prefix_len 10 \
--model_name t5-large --early_stopping 1 \
--save_name T5_experiment_sim_07 --save_dir save_dir \
--batch_size 4 --similarity_threshold 0.7 > sim07_3epochs.txt

# similarity_threshold = 0
python train_t5_cl.py --task_list imdb cb sst2 \
--select_k_per_class 1000 \
--lr 0.3 --num_epochs 3 \
--freeze_weights 1 --prefix_len 10 \
--model_name t5-large --early_stopping 1 \
--save_name T5_experiment_sim_0 --save_dir save_dir \
--batch_size 4 --similarity_threshold 0 > sim0_3epochs.txt
