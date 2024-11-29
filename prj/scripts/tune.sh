# mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet

# Neural
START_PARTITION=7
END_PARTITION=7
START_VAL_PARTITION=8
END_VAL_PARTITION=8
MODEL=mlp

python prj/scripts/tune_neural.py \
    --model $MODEL \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION \
    --start_val_partition $START_VAL_PARTITION \
    --end_val_partition $END_VAL_PARTITION \
    --n_seeds 1 \
    --out_dir experiments/tuning/neural/$MODEL \
    --verbose -1


# Tree
START_PARTITION=7
END_PARTITION=7
START_VAL_PARTITION=8
END_VAL_PARTITION=8
MODEL=lgbm

python prj/scripts/tune.py \
    --model $MODEL \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION \
    --start_val_partition $START_VAL_PARTITION \
    --end_val_partition $END_VAL_PARTITION \
    --n_seeds 1 \
    --verbose -1 \
    --storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet
