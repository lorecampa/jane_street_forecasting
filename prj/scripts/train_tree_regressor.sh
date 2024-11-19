#!/bin/sh
for partition_id in $(seq 4 4)
do
    python prj/scripts/train_tree_regressor.py \
        --start_train_partition_id $partition_id \
        --end_train_partition_id $partition_id \
        --n_training_seeds 3
done
