


python prj/scripts/catboost_tuning.py \
    -storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    -study_name catboost_offline_2025-01-03_20-42-18 \
    -n_trials 65


python prj/scripts/lgbm_tuning.py \
    -storage mysql+pymysql://admin:F1g5w#6zP4TN@janestreet.c3uaekuseqse.us-east-1.rds.amazonaws.com/janestreet \
    -n_trials 200 \
    -study_name lgbm_offline_2025-01-04_19-42-20