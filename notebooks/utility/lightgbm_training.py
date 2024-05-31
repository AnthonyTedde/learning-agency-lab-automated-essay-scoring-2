from dataclasses import dataclass, field, fields
from lightgbm import Dataset as LGBMDataSet
from lightgbm import early_stopping, cv, LGBMClassifier
import optuna
from optuna.samplers import TPESampler
import numpy as np
from .modeling import TargetTransformer
from typing import List, Any, Callable


@dataclass
class LGBMModelParameters:
    is_unbalanced: bool = field(init=False)
    num_thread: int = field(init=False)
    tree_learner: str = field(init=False)
    learning_rate: int = field(default=1e-1)
    task: str = field(default="train")
    objective: str = field(default="multiclass")
    num_class: int = field(default=2)
    boosting: str = field(default="gbdt")
    device_type: str = field(default="cpu")
    seed: int = field(default=1010)
    verbosity: int = field(default=0)
    metric: List[str] = field(default_factory=lambda: ["multi_logloss", "multi_error", "auc_mu", ])
    first_metric_only: bool = field(default=True)
    data_sample_strategy: str = field(default="goss")  # Gradient-Based One-Side Sampling --> Focusing on higher errored records
    boost_from_average: bool = field(default=True)
    extra_trees: bool = field(default=True)  # EXTremely RAndomize trees: speed up training, decrease training overfitting.
    is_provide_training_metric: bool = field(default=True)

    def __post_init__(self):
        self.params = {f.name: getattr(self, f.name) for f in fields(self) if f.name in self.__dict__.keys()}
        self._full_params = [f.name for f in fields(self)]

    def update_params(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in self._full_params}
        # TODO: Validate the type of the kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_params(self, trial=None, refit=False, config=None):
        # TODO:
        #   * Pass argument to filter parametrized hyperparameters
        #   * Pass Kwargs to increate parametrized hyperparameters
        if trial:
            self.params |= {
                "feature_fraction": trial.suggest_float("feature_fraction", .4, 1.0),
                "sigmoid": trial.suggest_float("sigmoid", 10e-4, 5.0),
                "num_leaves": trial.suggest_int("num_leaves", 8, 256),
                "lambda_l2": trial.suggest_float("lambda_l2", 0, 10.0),
                "min_sum_hessian": trial.suggest_int("min_sum_hessian", 1, 50),
                "bagging_fraction": trial.suggest_float("bagging_fraction", .4, 1.0),
                "sigmoid": trial.suggest_float("sigmoid", 10e-4, 8),
            }
        elif refit:
            self.params |= {"task": "refit"}
        else:
            self.params |= {"task": "predict"}
        return self.params



@dataclass
class LGBMCVArgs:
    # TODO: Interface class to fix the mandatory columns.

    params_fct: Callable[..., Any]
    callbacks: List[Callable[..., Any]]
    config: Any
    training_set: Any = field(init=False)
    nfold: int = field(default=10)
    stratified: bool = field(default=True)
    num_boost_round: int = field(default=1000)

    def set_training_data(self, data):
        self.training_set = data

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
    study: optuna.study.Study = None

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
        self.study = optuna.create_study(study_name=study_name, sampler=self.sampler, )
        self.study.optimize(func=self.objective(), n_trials=n_trials, show_progress_bar=True, )

    def _get_params(self):
        # TODO: get training param.
        # TODO: Which best strategy to include wrap_param ?
        fixed_params = wrap_param(refit=True)
        hyperparametrized_params = self.study.best_trial.params
        n_iteration = self.study.best_trial.user_attrs
        params = fixed_params | hyperparametrized_params | n_iteration

    def __getattr__(self, item):
        if self.study:
            getattr(self.study, item)
        #
