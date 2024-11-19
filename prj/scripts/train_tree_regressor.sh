#!/bin/sh
for partition_id in $(seq 4 5)
do
    python prj/scripts/train_tree_regressor.py \
        --model "catboost" \
        --start_train_partition_id $partition_id \
        --end_train_partition_id $partition_id \
        --n_training_seeds 1
done
