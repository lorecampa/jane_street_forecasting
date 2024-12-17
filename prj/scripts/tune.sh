# mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet

# Neural
MODEL=mlp

python prj/scripts/tune_neural.py \
    --model mlp \
    --n_trials 100 \
    --n_seeds 1 \
    --verbose -1 \
    --gpu



# Tree
START_PARTITION=4
END_PARTITION=8
START_VAL_PARTITION=9
END_VAL_PARTITION=9

python prj/scripts/tune.py \
    --model xgb \
    --n_trials 100 \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION \
    --start_val_partition $START_VAL_PARTITION \
    --end_val_partition $END_VAL_PARTITION \
    --n_seeds 1 \
    --storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    --gpu

# Tree LGBM Binary
python prj/scripts/tune_lgbm_binary.py \
    --model lgbm \
    --n_trials 300 \
    --n_seeds 1 \
    --verbose -1 \
    --study_name lgbm_1seeds_max_bin_128_0_8-9_9_20241211_003129 \
    --storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    --train


# OAMP
START_PARTITION=9
END_PARTITION=9

python prj/scripts/tune_oamp.py \
    --n_trials 500 \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION