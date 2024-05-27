def wrap_param(trial=None, refit=False, config=None):
    """
    Generate LightGBM parameters based on given configuration, trial, and refit flags.

    Parameters:
    trial (optuna.trial.Trial, optional): Optuna trial object for hyperparameter tuning. Default is None.
    refit (bool, optional): Flag indicating if the model is being refit. Default is False.
    config (ConfigurationSetting): Configuration settings

    Returns:
    dict: Dictionary containing the LightGBM parameters.

    Notes:
    - The `params` dictionary is initialized with default parameters suitable for a multiclass classification task.
    - If `trial` is provided, additional hyperparameters are suggested using the Optuna trial object.
    - If `refit` is True, the task is set to "refit".
    - If neither `trial` nor `refit` is provided, the task is set to "predict".
    """
    params = {
        "task": "train",
        "objective": "multiclass",
        "num_class": config.num_labels,
        "is_unbalanced": True,
        "boosting": "gbdt",
        "device_type": config.device,
        "num_thread": config.n_worker,
        "tree_learner": "feature_parallel",  # (https://lightgbm.readthedocs.io/en/stable/Parallel-Learning-Guide.html)
        "seed": config.seed,  # Overall seed.
        "verbosity": -1,
        "learning_rate": 1e-1,
        "metric": ["multi_logloss", "multi_error", "auc_mu", ],
        "first_metric_only": True,
        "data_sample_strategy": "goss",  # Gradient-Based One-Side Sampling --> Focusing on higher errored records
        "boost_from_average": True,
        "extra_trees": True,  # EXTremely RAndomize trees: speed up training, decrease training overfitting.
        "is_provide_training_metric": True,
        # "force_col_wise": True, # Force because large number of columns. Forcing reduce overhead due to testing the best data representation.
    }
    if trial:
        params |= {
            "feature_fraction": trial.suggest_float("feature_fraction", .4, 1.0),
            "sigmoid": trial.suggest_float("sigmoid", 10e-4, 5.0),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 10.0),
            "min_sum_hessian": trial.suggest_int("min_sum_hessian", 1, 50),
            "bagging_fraction": trial.suggest_float("bagging_fraction", .4, 1.0),
            "sigmoid": trial.suggest_float("sigmoid", 10e-4, 8),
        }
    elif refit:
        params |= {"task": "refit"}
    else:
        params |= {"task": "predict"}
    return params
