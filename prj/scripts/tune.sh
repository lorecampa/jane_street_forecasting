# mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet

# Neural
START_PARTITION=8
END_PARTITION=8
START_VAL_PARTITION=9
END_VAL_PARTITION=9
MODEL=mlp

python prj/scripts/tune.py \
    --model mlp \
    --n_trials 1 \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION \
    --start_val_partition $START_VAL_PARTITION \
    --end_val_partition $END_VAL_PARTITION \
    --n_seeds 1 \
    --verbose -1 \
    --gpu \
    --train

# Tree
START_PARTITION=6
END_PARTITION=7
START_VAL_PARTITION=8
END_VAL_PARTITION=8

python prj/scripts/tune.py \
    --model lgbm \
    --n_trials 100 \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION \
    --start_val_partition $START_VAL_PARTITION \
    --end_val_partition $END_VAL_PARTITION \
    --n_seeds 1 \
    --storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet

# Tree LGBM Binary
python prj/scripts/tune_lgbm_binary.py \
    --model lgbm \
    --n_trials 200 \
    --n_seeds 1 \
    --verbose -1 \
    --storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet


# OAMP
START_PARTITION=9
END_PARTITION=9

python prj/scripts/tune_oamp.py \
    --n_trials 500 \
    --start_partition $START_PARTITION \
    --end_partition $END_PARTITION