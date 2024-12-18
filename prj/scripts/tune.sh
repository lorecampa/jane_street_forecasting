# mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet

# Neural
MODEL=mlp
START_DT=1020
END_DT=1529
VAL_RATIO=0.2

python prj/scripts/tune_neural.py \
    --model mlp \
    --start_dt $START_DT \
    --end_dt $END_DT \
    --val_ratio $VAL_RATIO \
    --n_trials 100 \
    --n_seeds 1 \
    --verbose -1 \
    --storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    --gpu



# Tree
START_DT=1020
END_DT=1529

python prj/scripts/tune_tree.py \
    --model lgbm \
    --start_dt $START_DT \
    --end_dt $END_DT \
    --val_ratio 0.2 \
    --n_trials 10 \
    --n_seeds 1 \
    --study_name lgbm_1seeds_1020_1529-0.2_20241218_104118 \
    --storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \



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