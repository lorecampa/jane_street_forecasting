#!/bin/sh
for partition_id in $(seq 1 9)
do
    python /Users/lorecampa/Desktop/Projects/jane_street_forecasting/prj/scripts/train_tree_regressor.py \
        --start_train_partition_id $partition_id \
        --end_train_partition_id $partition_id \
        --n_training_seeds 1
done
