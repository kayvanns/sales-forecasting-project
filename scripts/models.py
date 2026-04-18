X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

lgb_train_full = lgb.Dataset(X_trainval, label=np.log1p(y_trainval))

lgb_tuned = lgb.train(
    {**best_params,         "objective":        "regression",
        "metric":           "rmse",
        "boosting_type":    "gbdt",     "seed":             42,
        "verbose":          -1,
        "nthread":          -1,},
    lgb_train_full,
    num_boost_round=best_lgb_num_boost_round
)

cat_tuned = CatBoostRegressor(
    **best_params_cat,iterations=best_iteration_cat, 
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
    allow_writing_files=False,
)
cat_tuned.fit(X_trainval, np.log1p(y_trainval))

xgb_tuned = XGBRegressor(
    **best_params_xgb,
    n_estimators=best_iteration_xgb,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
xgb_tuned.fit(X_trainval, np.log1p(y_trainval))
