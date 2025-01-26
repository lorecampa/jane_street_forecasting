
python prj/scripts/lgbm_tuning_online.py \
    -storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    -n_trials 300 \
    -study_name "lgbm_online_tuning_2025-01-07_18-13-58"


python prj/scripts/catboost_tuning_online.py \
    -storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    -study_name catboost_online_tuning_2025-01-13_07-47-00 \
    -n_trials 300

python prj/scripts/catboost_tuning_online_sum_models.py \
    -storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    -n_trials 300

python prj/scripts/tune_oamp.py \
    -storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    -n_trials 300