# Neural

python prj/scripts/tune_neural.py \
    --model mlp \
    --n_seeds 1 \
    --out_dir experiments/tuning/neural/mlp


# Tree
START_PARTITION=7
END_PARTITION=7
python prj/scripts/tune_tree.py \
    --model lgbm \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION \
    --start_val_partition $(END_PARTITION+1) \
    --end_val_partition $(END_PARTITION+1) \
    --n_seeds 1 \
    --out_dir experiments/tuning/tree/lgbm \
    --verbose -1