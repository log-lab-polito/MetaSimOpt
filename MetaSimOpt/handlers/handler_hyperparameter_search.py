import random
import json
import optuna
import os
import copy
import multiprocessing
from optuna.pruners import MedianPruner
import numpy as np
from joblib import Parallel, delayed
import math
from MetaSimOpt.handlers import HandlerTraining
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING: # only type hints
    from optuna import Trial

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to serialize NumPy data types.

    This encoder converts NumPy integers, floats, and arrays into their
    corresponding native Python types, making them serializable using the
    standard `json` module.

    Methods
    default(obj)
        Overrides the default method to handle NumPy data types.
    """

    def default(self, obj):
        if isinstance(obj, (np.integer, )):
            return int(obj)
        elif isinstance(obj, (np.floating, )):
            return float(obj)
        elif isinstance(obj, (np.ndarray, )):
            return obj.tolist()
        return super().default(obj)
    
def run_random_trial(args) -> tuple[str, dict]:
    """
    Run a single random hyperparameter trial.

    This function copies the handler, samples random hyperparameters from the search
    space, trains the model using those hyperparameters, and returns the results.

    Args:
        args (tuple): A tuple containing:
            - handler (HandlerTraining): Training handler object.
            - search_space (dict): Dictionary defining the hyperparameter search space.
            - cv_mode (str): Cross-validation mode ('split' or 'kfold').
            - n_folds (int): Number of folds for k-fold cross-validation.
            - test_size (float): Test set proportion for validation split.
            - path_dir (str): Directory path to save logs and results.

    Returns:
        tuple[str, dict]: A tuple where the first element is a JSON string representing
        the sampled hyperparameters, and the second is a dictionary with trial results.
    """

    handler, search_space, cv_mode, n_folds, test_size, path_dir = args
    handler_copy = copy.deepcopy(handler)
    searcher_copy = HandlerHyperSearch(
        handler=handler_copy,
        search_space=search_space,
        method="random",
        cv_mode=cv_mode,
        n_folds=n_folds,
        test_size=test_size,
        path_dir=path_dir
    )
    hyper_model, hyper_train = searcher_copy._sample_hyperparams()
    _, result = searcher_copy._objective(hyper_model = hyper_model, hyper_train = hyper_train)
    return result

class HandlerHyperSearch:

    def __init__(self, handler : "HandlerTraining", search_space : dict, method : str = "optuna", score_fn : callable = None, direction : list = None, cv_mode : str = "split", n_folds : int = 4, test_size : float = 0.2, path_dir : str = None):
        """
        Class to perform hyperparameter search using either Optuna or random search.

        Attributes:
            handler (HandlerTraining): The training handler instance used to build, train, and evaluate the model.
            search_space (dict): Dictionary defining the search space for hyperparameters.
            method (str): Search method to use ("optuna" or "random").
            score_fn (callable): Function to calculate the composite score from validation results.
            direction (list) : List of optimisation direction for objectives.
            cv_mode (str): Cross-validation mode ("split" or "kfold").
            n_folds (int): Number of folds used for k-fold cross-validation.
            test_size (float): Test set proportion used in train/validation split.
            results (list): List of results from each trial.
            tested_configs (set): Set of already tested configurations.
            path_dir (str): Directory where results and logs are stored.
        """        

        self.handler = handler
        self.search_space = search_space
        self.method = method
        self.pruner = None
        self.score_fn = score_fn or self.default_score_fn
        self.direction = direction or ["minimize", "minimize", "minimize"]
        self.cv_mode = cv_mode
        self.n_folds = n_folds
        self.test_size = test_size
        self.results = []

        self.tested_configs = set()
        self.path_dir = path_dir
        self.config_log_path = os.path.join(self.path_dir, "tested_configs.json")
        self._load_tested_configs()


    def enable_pruner(self, method : str = "median", **kwargs):
        """
        Enable pruning strategy for early stopping during Optuna search.

        Args:
            method (str): Pruner method to use. Currently, only "median" is supported.
            **kwargs: Additional keyword arguments passed to the pruner, such as:
                - n_startup_trials (int): Number of trials to run before pruning starts.
                - n_warmup_steps (int): Number of steps before pruning is considered.
        """

        if method == "median":
            n_startup_trials = kwargs.get("n_startup_trials", 10)
            n_warmup_steps = kwargs.get("n_warmup_steps", 150)
            self.pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)
        else:
            raise ValueError(f"Unsupported pruner method: {method}")
        

    def default_score_fn(self, loss : float, val_loss : float, residuals_stats : dict) -> float:
        """
        Default scoring function combining loss, score, and residuals statistics.

        Args:
            loss (float): Best training loss.
            val_loss (float): Validation loss of the best training epoch.
            residuals_stats (dict): Dictionary with residual statistics (keys: mean, stdev, skew, kurtosis).

        Returns:
            float: Composite score calculated from the provided metrics.
        """

        res_skew = abs(residuals_stats["skew"])
        res_bias = abs(residuals_stats["mean"])

        # composite_score = val_loss + 10 * res_skew + delta_losses
        # (validation loss ^2 ) / 3
        # (res_bias ^2 ) / 3
        
        return val_loss, res_skew, res_bias
    

    def _load_tested_configs(self):
        """
        Load previously tested configurations from disk to avoid duplication.
        """

        try:
            with open(self.config_log_path, "r") as f:
                configs = json.load(f)
                self.tested_configs = set(configs)
        except FileNotFoundError:
            self.tested_configs = set()


    def _save_tested_configs(self):
        """
        Save the set of tested configurations to disk as JSON.
        """

        with open(self.config_log_path, "w") as f:
            json.dump(list(self.tested_configs), f, indent=2, cls=NumpyEncoder)


    def _sample_hyperparams(self, trial: "Trial" = None) -> tuple[dict, dict]:
        """
        Sample model and training hyperparameters from the search space with conditional dependencies.

        Supports both Optuna-based sampling and random sampling.

        Returns:
            tuple:
                - hyper_model (dict): Sampled model hyperparameters.
                - hyper_train (dict): Sampled training hyperparameters.
        """

        sp = self.search_space
        if isinstance(self.handler, HandlerTraining):
            default_hp = self.handler.factory.model_class.get_model_hyperparameters()
        else:
            raise ValueError("Type of handler not supported. Support only HandlerTraining")

        sampled = {}

        def hp(key, values):
            if not values:
                raise ValueError(f"Search space for '{key}' is empty")
            if self.method == "optuna" and trial is not None:
                if isinstance(values, list):
                    return trial.suggest_categorical(key, values) if len(values) > 1 else values[0]
                elif isinstance(values, tuple) and len(values) == 2:
                    return trial.suggest_float(key, values[0], values[1])
                else:
                    raise ValueError(f"Unsupported format for key '{key}': {values}")
            elif self.method == "random":
                if isinstance(values, list):
                    return random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    return random.uniform(values[0], values[1])
                else:
                    raise ValueError(f"Unsupported format for key '{key}': {values}")
            else:
                raise ValueError("Unsupported method or trial is None for optuna")

        for k, v in sp.items():
            if isinstance(v, dict):
                cond = v.get("condition", lambda hp: True)
                values = v.get("values", [])
            else:
                cond = lambda hp: True
                values = v

            if cond(sampled):
                sampled[k] = hp(k, values)
            # if cond is False, continue

        model_keys = set(default_hp.keys())
        hyper_model = {k: v for k, v in sampled.items() if k in model_keys}
        hyper_train = {k: v for k, v in sampled.items() if k not in model_keys}

        return hyper_model, hyper_train
    
    
    def _set_handler(self, hyper_model : dict, hyper_train : dict):
        """
        Set up the training handler with the provided hyperparameters.

        Args:
            hyper_model (dict): Dictionary of model-specific hyperparameters.
            hyper_train (dict): Dictionary of training-specific hyperparameters.
        """
        if isinstance(self.handler, HandlerTraining):
            self.handler.reset_training()
            self.handler.set_model_hyperparameters(hyper_model)
            self.handler.set_training_hyperparameters(hyper_train)
            self.handler.set_optimiser()
        else:
            raise ValueError("Type of handler not supported. Support only HandlerTraining")


    def _objective(self, trial : "Trial" = None, hyper_model : dict = None, hyper_train : dict = None) -> tuple[float, dict]:
        """
        Runs a single trial for hyperparameter optimization.

        Depending on the search method ("optuna" or "random"), this method either samples new 
        hyperparameters or uses the ones provided. Then it configures the model and evaluates its performance.

        Args:
            trial (Trial, optional): Optuna trial object, required if using Optuna.
            hyper_model (dict, optional): Model hyperparameters to use (e.g., layers, units, dropout).
            hyper_train (dict, optional): Training hyperparameters to use (e.g., epochs, learning rate).

        Returns:
            tuple:
                - composite_score (float): Final score based on validation performance and residual stats.
                - result (dict): Dictionary containing:
                    - 'hyper_model': Used model hyperparameters
                    - 'hyper_train': Used training hyperparameters
                    - 'val_loss': Final validation loss
                    - 'val_score': Final validation score
                    - 'residuals_stats': Residual distribution statistics
                    - 'composite_score': Composite evaluation score
        """
        
        if self.method == "optuna":
            hyper_model, hyper_train = self._sample_hyperparams(trial)

        self._set_handler(hyper_model = hyper_model, hyper_train = hyper_train)

        if self.cv_mode == "split":
            if isinstance(self.handler, HandlerTraining):
                losses, val_losses, scores, val_scores, residuals_stats = self.handler.train_and_validate(test_size = self.test_size)
            else:
                raise ValueError("Type of handler not supported. Support only HandlerTraining")
            
        elif self.cv_mode == "kfold":
            if isinstance(self.handler, HandlerTraining):
                losses, val_losses, scores, val_scores, residuals_stats = self.handler.k_fold_cross_val(n_folds = self.n_folds, parallel = True)
            else:
                raise ValueError("Type of handler not supported. Support only HandlerTraining")
            
        else:
            raise ValueError("Unsupported cv_mode. Choose 'split' or 'kfold'")
        
        idx_min = np.argmin(losses)
        loss = losses[idx_min]
        val_loss = val_losses[idx_min]

        val_loss, res_skew, res_bias = self.score_fn(loss, val_loss, residuals_stats)

        score = 0.0
        val_score = 0.0

        if scores:
            score = scores[-1]
            val_score = val_scores[-1]

        result = {
            "hyper_model": hyper_model,
            "hyper_train": hyper_train,
            "loss" : loss,
            "val_loss": val_loss,
            "score" : score,
            "val_score" : val_score,
            "residuals_stats": residuals_stats,
            #"composite_score": composite_score
        }

        return (val_loss, res_skew, res_bias), result
    

    def run(self, n_trials : int = 100, clear_results : bool = False, reset_study : bool = False, save_to_excel : bool = True, parallelism : bool = False) -> dict:
        """
        Run the hyperparameter search process.

        Works with either Optuna optimization or random search. Can optionally reset or reuse previous results.

        Args:
            n_trials (int) : Number of trials to run.
            clear_results (bool) : If True, previous results (JSON files) will be deleted.
            reset_study (bool) : If True, delete existing Optuna study database and start fresh.
            save_to_excel (bool) : If True, results will be saved to an Excel file (only if using Optuna).
            parallelism (bool) : If True, enable parallelism of hyperparameter search.

        Returns:
            dict: Dictionary of the best trial result containing hyperparameters and evaluation metrics.
        """

        # remove previous results
        if clear_results:
            path_results = os.path.join(self.path_dir, f'results_{self.method}.json')
            if os.path.exists(path_results):
                os.remove(path_results)
        
        # parallelism
        if parallelism:
            if hasattr(self.handler, "device") and str(self.handler.device).lower() == "cpu":
                max_cores = math.floor(multiprocessing.cpu_count() / 3)
                if self.cv_mode == "kfold":
                    n_jobs = max(1, math.floor(max_cores / self.n_folds))
                else:
                    n_jobs = max_cores
                print(f"[Info] Parallelism enabled with {n_jobs} jobs (CPU mode)")
            else:
                n_jobs = 1
                print(f"[Info] Parallelism disabled (Device: {getattr(self.handler, 'device', 'unknown')})")
        else:
            n_jobs = 1

        if self.method == "optuna":
            study_name = "optuna_hyperparameter_search"
            storage_path = os.path.join(self.path_dir, "optuna_study.db")
            storage_url = f"sqlite:///{storage_path}"

            if reset_study and os.path.exists(storage_path):
                os.remove(storage_path)

            def safe_objective(trial):
                # handler copy
                handler_copy = copy.deepcopy(self.handler)

                searcher_copy = HandlerHyperSearch(
                    handler=handler_copy,
                    search_space=self.search_space,
                    method=self.method,
                    cv_mode=self.cv_mode,
                    n_folds=self.n_folds,
                    test_size=self.test_size,
                    path_dir=self.path_dir,
                )
                
                (val_loss, res_skew, res_bias), result = searcher_copy._objective(trial=trial)
                self.results.append(result)

                # save results
                result_path = os.path.join(self.path_dir, "results_optuna.json")
                try:
                    with open(result_path, "r") as f:
                        current_results = json.load(f)
                except FileNotFoundError:
                    current_results = []

                current_results.append(result)
                with open(result_path, "w") as f:
                    json.dump(current_results, f, indent=2, cls=NumpyEncoder)

                return val_loss, res_skew, res_bias

            def progress_callback(study, trial):
                values = trial.values  # Always safe to use
                if len(values) == 1:
                    print(f"[Optuna] Trial {trial.number} done → Score: {values[0]:.4f}")
                else:
                    score_str = ", ".join(f"{v:.4f}" for v in values)
                    print(f"[Optuna] Trial {trial.number} done → Scores: ({score_str})")

            if len(self.direction) == 1:
                study = optuna.create_study(
                    direction=self.direction,
                    study_name=study_name,
                    storage=storage_url,
                    pruner=self.pruner,
                    load_if_exists=True
                )
            else:
                study = optuna.create_study(
                    directions=self.direction,
                    study_name=study_name,
                    storage=storage_url,
                    pruner=self.pruner,
                    load_if_exists=True
                )

            study.optimize(
                safe_objective,
                n_trials=n_trials,
                callbacks=[progress_callback],
                n_jobs=n_jobs,
                catch=(Exception,)
            )

        else:
            
            # load if exists
            result_path = os.path.join(self.path_dir, "results_random.json")
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    self.results = json.load(f)

            args_list = [
                (self.handler, self.search_space, self.cv_mode, self.n_folds, self.test_size, self.path_dir)
                for _ in range(n_trials)
            ]

            if n_jobs > 1:
                all_results = Parallel(n_jobs=n_jobs)(delayed(run_random_trial)(args) for args in args_list)
            else:
                all_results = [run_random_trial(args) for args in args_list]

            self.results.extend(all_results[:n_trials])
            for i, r in enumerate(all_results[:n_trials]):
                print(f"[Random] Trial {i+1}/{n_trials} → Score: {r['val_loss']:.2f}")

            # save results
            with open(result_path, "w") as f:
                json.dump(self.results, f, indent=2, cls=NumpyEncoder)

        best_result = min(self.results, key=lambda r: r["val_loss"])

        if save_to_excel:
            if self.method == "optuna":
                df = study.trials_dataframe()

                if len(study.directions) == 1:
                    # single objective
                    df = df.sort_values("value")
                else:
                    # multi objective
                    df = df.sort_values("values_0")

            else:
                flat_data = []
                for entry in self.results:
                    flat_entry = {}
                    
                    # hyperparameters
                    for k, v in entry['hyper_model'].items():
                        flat_entry[f"{k}"] = v
                    for k, v in entry['hyper_train'].items():
                        flat_entry[f"{k}"] = v

                    # metrics
                    flat_entry['loss'] = entry['loss']
                    flat_entry['val_loss'] = entry['val_loss']
                    flat_entry['score'] = entry['score']
                    flat_entry['val_score'] = entry['val_score']

                    for k, v in entry['residuals_stats'].items():
                        flat_entry[f"{k}"] = v
                    
                    flat_data.append(flat_entry)

                # create df
                df = pd.DataFrame(flat_data)
            
            df.to_excel(os.path.join(self.path_dir, f"results_{self.method}.xlsx"))

        return best_result
