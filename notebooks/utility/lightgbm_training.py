from dataclasses import dataclass, field
from lightgbm import Dataset as LGBMDataSet
from lightgbm import early_stopping, cv, LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import numpy as np
from .modeling import TargetTransformer
from typing import List, Any, Callable


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

@dataclass
class LGBMCVArgs:

    # TODO: Interface class to fix the mandatory columns.

    params_fct: Callable[..., Any]
    callbacks: List[Callable[..., Any]]
    config:Any
    training_set:Any = field(init=False)
    nfold: int = field(default=10)
    stratified:bool = field(default=True)
    num_boost_round: int = field(default=1000)

    def set_training_data(self, data):
        self.training_set=data

    def run(self, trial):
        reg_cv = cv(
            params=self.params_fct(trial, config=self.config),
            train_set=self.training_set,
            num_boost_round=self.num_boost_round,  # == num_iterations
            nfold=self.nfold,
            stratified=self.stratified,
            callbacks=self.callbacks,
            return_cvbooster=True,
        )
        trial.set_user_attr("num_iterations", reg_cv.get("cvbooster").best_iteration)
        return np.min(
            np.add(
                reg_cv.get("valid multi_logloss-mean"),
                reg_cv.get("valid multi_logloss-stdv")
            )
        )


class LGBMTraining:
    study:optuna.study.Study = None
    def __init__(self, X_training, y_training, training_strategy, sampler, transformer=None):
        # TODO: Scaler
        if transformer:
            y_training = transformer.fit_transform(y_training)

        self.training_set = LGBMDataSet(
            data=X_training,
            label=y_training,
            free_raw_data=False
        )
        self.training_strategy = training_strategy
        self.training_strategy.set_training_data(self.training_set)
        self.sampler = sampler


    def objective(self):
        return self.training_strategy.run

    def train(self, n_trials=10, study_name="default"):
        self.study = optuna.create_study( study_name=study_name, sampler=self.sampler, )
        self.study.optimize(func=self.objective(), n_trials=n_trials, show_progress_bar=True, )

    def __getattr__(self, item):
        if self.study:
            getattr(self.study, item)
        #