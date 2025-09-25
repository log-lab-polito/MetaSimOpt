import os
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, suppress=True)
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from functools import partial
from joblib import Parallel, delayed
import multiprocessing
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

from MetaSimOpt.metamodels import RNN_Metamodel, LSTM_Metamodel, GRU_Metamodel
from MetaSimOpt.utils import _normalise_dataset, _generate_tensors
from MetaSimOpt.utils import print_training, plot_training, compute_residual_stats, print_residuals


class HandlerTraining:

    @staticmethod
    def get_training_hypermeters(print_hyper = False):
        
        hyper = {
            'epochs' : 100, 
            'batch_size' : 32, 
            'learning_rate' : 1e-3, 
            'w_decay' : 1e-4,
            'l1_lambda' : 1e-4
        }

        if print_hyper:
            print("TRAINING HYPERPARAMETERS:")
            for i in hyper.keys():
                print(i)

        return hyper
    
    
    @staticmethod
    def get_search_space(model_class):
        
        if model_class in [RNN_Metamodel, LSTM_Metamodel, GRU_Metamodel]: 
            ss = {
                "epochs": [250],
                "batch_size": [16, 32, 64],
                "learning_rate": [5e-4, 1e-3, 5e-3],
                "w_decay": [1e-4, 1e-3, 1e-2],
                "l1_lambda" : [1e-4, 1e-3, 1e-2]
            }
        else:
            raise ValueError("Specify a valid model class. Supported RNN_Metamodel, LSTM_Metamodel, GRU_Metamodel")

        return ss


    def __init__(self, model_class, factory, device = 'cpu'):
        """
        Class to handle the training of the metamodels

        Args:
            data (dict) : dictionary in which the key represents the type of data (features_rec, features_lin, labels) and the values are arrays with the data
            factory (MetamodelFactory) : instance of class used to create the models
            device (str) : can be "cpu" or "cuda"

        """

        self.model_class = model_class
        self.factory = factory
        self.x = None
        self.y = None
        self.scaler_type = StandardScaler
        self.model_instance = None
        self.loss_function = None
        self.optimiser = None
        self.score_function = mean_absolute_percentage_error
        self.model_hyperparameters = None
        self.training_hyperparameters = None
        self.device = device
        
        self.losses = None
        self.val_losses = None
        self.scores = None
        self.val_scores = None
        self.best_model_state = None
        self.best_train_loss = float('inf')
        self.scalers = None


    def reset_training(self):

        self.best_model_state = None
        self.losses = None
        self.val_losses = None
        self.scores = None
        self.val_scores = None
        

    def load_dataset(self, x, y, normalisation = "min-max"):
        self.x = x
        self.y = y
        
        if normalisation not in ["min-max", "standard"]:
            raise ValueError(f"Normalisation {normalisation} not supported. Only supported min-max, standard")
        if normalisation == "min-max":
            self.scaler_type = MinMaxScaler
        elif normalisation == "standard":
            self.scaler_type = StandardScaler

    
    def set_model_hyperparameters(self, hyper):

        self.model_hyperparameters = hyper


    def set_training_hyperparameters(self, hyper):

        # check hyperameters
        default_hp = HandlerTraining.get_training_hypermeters()
        for param in default_hp.keys():
            if param not in hyper:
                hyper[param] = default_hp[param]
                print(f"Missing required hyperparameter: {param}. Set to default value {default_hp[param]}")

        self.training_hyperparameters = hyper


    def set_loss_function(self, loss_function = nn.MSELoss):

        self.loss_function = loss_function()

    
    def _instanciate_model(self):

        if self.model_hyperparameters is None:
            raise ValueError(f"Set model hyperparameters before istanciate model. Call set_model_hyperparameters()")
        else:
            self.factory.default_args['hyperparameters'] = self.model_hyperparameters
            self.model_instance = self.factory.create()

    
    def set_optimiser(self, optimiser_class = optim.AdamW, force_instance = False):

        if self.model_instance is None or force_instance:
            self._instanciate_model()

        if optimiser_class in [optim.Adam, optim.AdamW]:
            self.optimiser = optimiser_class(
                self.model_instance.parameters(),
                lr=self.training_hyperparameters.get('learning_rate'),
                weight_decay=self.training_hyperparameters.get('w_decay')
            )
        else:
            raise ValueError("Unsupported optimiser class. Supported optim.Adam, optim.AdamW")
    

    def training_loop(self, train_loader, val_loader, validation, compute_score = True, print_progress=False):

        if self.training_hyperparameters is None:
            raise ValueError("Set training hyperparameters before instantiating the model. Call set_training_hyperparameters().")
        if self.loss_function is None:
            raise ValueError("Set loss function before proceeding. Call set_loss_function().")
        if self.optimiser is None:
            raise ValueError("Set optimiser before proceeding. Call set_optimiser().")

        losses, val_losses = [], []
        scores, val_scores = [], []

        weight_decay = self.optimiser.param_groups[0].get('weight_decay', 0.0)
        n_epochs = self.training_hyperparameters.get('epochs', 1)

        # move model to device
        self.model_instance.to(self.device)

        best_train_loss = np.inf
        best_val_loss = np.inf
        patience = 10
        patience_stable = 5
        epochs_no_improve = 0
        tolerance = 2.5e-2
        window_size = 5
        loss_window = []
        epochs_stable_loss = 0

        if print_progress:
            print("\nTRAINING...")

        for epoch in range(n_epochs):
            self.model_instance.train()
            total_loss = 0.0
            if compute_score:
                total_score = 0.0

            for batch in train_loader:
                *inputs, targets = batch
                # move data to device
                inputs = [x.to(self.device) for x in inputs]
                targets = targets.to(self.device)

                # calculate lengths if supported
                if hasattr(self.model_instance, "compute_lengths"):
                    lengths = self.model_instance.compute_lengths(*inputs)
                    output = self.model_instance(*inputs, lengths)
                    
                else:
                    output = self.model_instance(*inputs)

                # output: (B, T_batch, D)
                # targets: (B, T_dataset, D)
                # mask: (B, T_batch) perché T_batch = max(lengths)

                mask = torch.arange(output.shape[1], device=lengths.device)[None, :] < lengths[:, None]  # (B, T_batch)

                if targets.dim() == 3:
                    mask = mask.unsqueeze(-1)  # (B, T_batch, 1)
                    mask = mask.expand(-1, -1, targets.shape[-1])  # (B, T_batch, D)

                # Trim targets to output shape:
                trimmed_targets = targets[:, :output.shape[1], :]  # (B, T_batch, D)

                # Apply mask
                masked_output = output[mask]
                masked_targets = trimmed_targets[mask]

                loss = self.loss_function(masked_output, masked_targets)

                # ---- L1 Regularization ----
                l1_lambda = self.training_hyperparameters.get('l1_lambda', 0.0)
                if l1_lambda > 0:
                    l1_reg = sum(param.abs().sum() for param in self.model_instance.parameters() if param.requires_grad)
                    loss += l1_lambda * l1_reg

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                total_loss += loss.item()

                # Score
                if compute_score:
                    targets_np = masked_targets.detach().cpu().numpy()
                    output_np = masked_output.detach().cpu().numpy()
                    if self.score_function in [mean_absolute_percentage_error] and output_np.ndim == 3:
                        # Handle many-to-many outputs for sklearn metrics
                        b,t,d = output_np.shape
                        output_np = output_np.reshape(b*t,d)
                        targets_np = targets_np.reshape(b*t,d)
                    score = self.score_function(targets_np, output_np)                   
                    total_score += score

            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)

            if compute_score:
                avg_score = total_score / len(train_loader)
                scores.append(avg_score)

            if not validation:
                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    self.best_model_state = self.model_instance.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            else:
                if avg_loss < best_train_loss:
                    best_train_loss = avg_loss
                    self.best_model_state = self.model_instance.state_dict()


            if not validation:

                loss_window.append(avg_loss)
                if len(loss_window) > window_size:
                    loss_window.pop(0)

                if len(loss_window) == window_size:
                    epsilon = 1e-8

                    deltas = [
                        abs(loss_window[i] - loss_window[i-1]) / (abs(loss_window[i-1]) + epsilon)
                        for i in range(1, window_size)
                    ]
                    delta_perc_mean = sum(deltas) / len(deltas)

                    if delta_perc_mean < tolerance:
                        epochs_stable_loss += 1
                    else:
                        epochs_stable_loss = 0

            # Validation
            if validation:
                self.model_instance.eval()
                val_total_loss = 0.0
                if compute_score:
                    val_total_score = 0.0

                with torch.no_grad():
                    for batch in val_loader:
                        *val_inputs, val_targets = batch
                        # move data to device
                        val_inputs = [x.to(self.device) for x in val_inputs]
                        val_targets = val_targets.to(self.device)

                        if hasattr(self.model_instance, "compute_lengths"):
                            val_lengths = self.model_instance.compute_lengths(*val_inputs)
                            val_output = self.model_instance(*val_inputs, val_lengths)
                        else:
                            val_output = self.model_instance(*val_inputs)

                        # output: (B, T_batch, D)
                        # targets: (B, T_dataset, D)
                        # mask: (B, T_batch) perché T_batch = max(lengths)

                        val_mask = torch.arange(val_output.shape[1], device=val_lengths.device)[None, :] < val_lengths[:, None]  # (B, T_batch)

                        if val_targets.dim() == 3:
                            val_mask = val_mask.unsqueeze(-1)  # (B, T_batch, 1)
                            val_mask = val_mask.expand(-1, -1, val_targets.shape[-1])  # (B, T_batch, D)

                        # Trim targets to output shape:
                        trimmed_val_targets = val_targets[:, :val_output.shape[1], :]  # (B, T_batch, D)

                        # Apply mask
                        masked_val_output = val_output[val_mask]
                        masked_val_targets = trimmed_val_targets[val_mask]

                        val_loss = self.loss_function(masked_val_output, masked_val_targets)

                        #val_loss = self.loss_function(val_output, val_targets)
                        val_total_loss += val_loss.item()

                        if compute_score: 
                            val_targets_np = masked_val_targets.detach().cpu().numpy()
                            val_output_np = masked_val_output.detach().cpu().numpy()
                            
                            if self.score_function in [mean_absolute_percentage_error] and val_output_np.ndim == 3:
                                # Handle many-to-many outputs for sklearn metrics
                                b,t,d = val_output_np.shape
                                val_output_np = val_output_np.reshape(b*t,d)
                                val_targets_np = val_targets_np.reshape(b*t,d)
                            
                            val_score = self.score_function(val_targets_np, val_output_np)
                            val_total_score += val_score

                avg_val_loss = val_total_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                loss_window.append(avg_val_loss)
                if len(loss_window) > window_size:
                    loss_window.pop(0)

                if len(loss_window) == window_size:
                    epsilon = 1e-8

                    deltas = [
                        abs(loss_window[i] - loss_window[i-1]) / (abs(loss_window[i-1]) + epsilon)
                        for i in range(1, window_size)
                    ]
                    delta_perc_mean = sum(deltas) / len(deltas)

                    if delta_perc_mean < tolerance:
                        epochs_stable_loss += 1
                    else:
                        epochs_stable_loss = 0

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.best_model_state = self.model_instance.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if compute_score:
                    avg_val_score = val_total_score / len(val_loader)
                    val_scores.append(avg_val_score)

            if epochs_no_improve >= patience or epochs_stable_loss >= patience_stable:
                print(f"Early stopping triggered. Epochs no improve {epochs_no_improve}, epochs stable loss {epochs_stable_loss}")
                break

            if print_progress:
                base_msg = f"Epoch {epoch + 1}/{n_epochs} ({round(((epoch+1)/n_epochs)*100,2)} %) --> loss: {avg_loss:.4f}"
                
                if validation:
                    base_msg += f" | val loss: {val_losses[-1] if val_losses else 0:.4f}"
                
                if compute_score:
                    base_msg += f" | score: {avg_score*100:.2f}%"
                    if validation: # Only add val score if validation is also enabled
                        base_msg += f" | val score: {val_scores[-1]*100 if val_scores else 0:.2f}%"
                        
                print(base_msg)

        self.model_instance.load_state_dict(self.best_model_state)

        return losses, val_losses, scores, val_scores


    def train_no_validate(self, print_progress, compute_score = True):

        x_train_scaled, _, self.scalers = _normalise_dataset(self.x, scaler_type=self.scaler_type)
        tensors = _generate_tensors(data=[x_train_scaled, self.y], device = self.device)  # generate (x1, x2, ..., y)
        train_dataset = TensorDataset(*tensors)

        train_loader = DataLoader(train_dataset, batch_size=self.training_hyperparameters.get('batch_size'), shuffle=True)
        losses, val_losses, scores, val_scores = self.training_loop(train_loader = train_loader, val_loader = None, validation = False, print_progress = print_progress, compute_score = compute_score)
        
        return losses, val_losses, scores, val_scores
    

    def train_and_validate(self, test_size, print_progress = False, compute_score = True):

        validation = True if test_size > 0 else False

        # Split
        indices = list(range(self.y.shape[0]))
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, shuffle=True
        )

        x_train = [t[train_idx] for t in self.x]
        x_val = [t[val_idx] for t in self.x]

        x_train_scaled, x_val_scaled, _ = _normalise_dataset(x_train, x_val, self.scaler_type)

        y_train = self.y[train_idx]
        y_val = self.y[val_idx]

        tensors_train = _generate_tensors(data=[x_train_scaled, y_train], device = self.device)
        inputs_train, targets_train = tensors_train[:-1], tensors_train[-1]
        tensors_val = _generate_tensors(data=[x_val_scaled, y_val], device = self.device)
        inputs_val, targets_val = tensors_val[:-1], tensors_val[-1]

        train_dataset = TensorDataset(*inputs_train, targets_train)
        val_dataset = TensorDataset(*inputs_val, targets_val)

        batch_size = self.training_hyperparameters.get('batch_size')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        losses, val_losses, scores, val_scores = self.training_loop(train_loader=train_loader, val_loader=val_loader, validation=validation, print_progress=print_progress, compute_score = compute_score)
        
        # Residual stats
        predictions = self.model_instance.predict(x = inputs_val)
        _, residuals_stats = compute_residual_stats(predictions, y_val)

        return losses, val_losses, scores, val_scores, residuals_stats


    @staticmethod
    def process_fold(args):
        (
            train_idx, val_idx,
            inputs, target,
            scaler_type,
            device,
            batch_size,
            training_loop_fn,
            prediction_fn,
            print_progress,
            compute_score,
            reset_fn,
            reset_opt
        ) = args

        # Split and generate tensors

        x_train = [t[train_idx] for t in inputs]
        x_val = [t[val_idx] for t in inputs]

        x_train_scaled, x_val_scaled, _ = _normalise_dataset(x_train, x_val, scaler_type)

        y_train = target[train_idx]
        y_val = target[val_idx]

        tensors_train = _generate_tensors(data=[x_train_scaled, y_train], device = device)
        inputs_train, targets_train = tensors_train[:-1], tensors_train[-1]
        tensors_val = _generate_tensors(data=[x_val_scaled, y_val], device = device)
        inputs_val, targets_val = tensors_val[:-1], tensors_val[-1]

        # Dataset e loader
        train_dataset = TensorDataset(*inputs_train, targets_train)
        val_dataset = TensorDataset(*inputs_val, targets_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Execute training
        losses, val_losses, scores, val_scores = training_loop_fn(
            train_loader=train_loader,
            val_loader=val_loader,
            validation=True,
            print_progress=print_progress,
            compute_score = compute_score
        )

        # residuals
        predictions = prediction_fn(x = inputs_val)
        _, residuals_stats = compute_residual_stats(predictions, y_val)

        reset_fn()
        reset_opt()

        return losses, val_losses, scores, val_scores, residuals_stats
    

    def k_fold_cross_val(self, n_folds=5, print_progress=False, parallel=False, compute_score = True):

        kf = KFold(n_splits=n_folds, shuffle=True)
        folds = list(kf.split(self.x[0]))  # assume all inputs have same first dimension

        batch_size = self.training_hyperparameters.get('batch_size')

        residuals_fn = partial(self.model_instance.predict)
        reset_opt = partial(self.set_optimiser, optimiser_class = type(self.optimiser), force_instance = True)

        args_list = [
            (
                train_idx, val_idx,
                self.x, self.y,
                self.scaler_type,
                self.device,
                batch_size,
                self.training_loop,
                residuals_fn,
                print_progress,
                compute_score,
                self.reset_training,
                reset_opt
            )
            for train_idx, val_idx in folds
        ]

        if parallel and str(self.device).lower() == "cpu":
            n_processes = min(n_folds, max(1, math.floor(multiprocessing.cpu_count() / 3)))
            results = Parallel(n_jobs=n_processes)(
                delayed(HandlerTraining.process_fold)(args) for args in args_list
            )
        else:
            results = [HandlerTraining.process_fold(args) for args in args_list]

        # aggregate results
        losses_folds, val_losses_folds, scores_folds, val_scores_folds, residual_stats_folds = zip(*results)

        n_el = np.inf
        for fold in losses_folds:
            if len(fold) < n_el:
                n_el = len(fold)

        losses = [sum(row[i] for row in losses_folds) / n_folds for i in range(n_el)]
        val_losses = [sum(row[i] for row in val_losses_folds) / n_folds for i in range(n_el)]
        scores = [sum(row[i] for row in scores_folds) / n_folds for i in range(n_el)]
        val_scores = [sum(row[i] for row in val_scores_folds) / n_folds for i in range(n_el)]
        residuals_stats = {
            k: np.mean([fold[k] for fold in residual_stats_folds]) for k in residual_stats_folds[0]
        }

        return losses, val_losses, scores, val_scores, residuals_stats
    
    
    def train(self, validation = False, test_size=0.2, k_fold = False, n_folds = 5, print_progress = True, print_results = True, plot_results = True, print_res_stats = True, parallel = True, compute_score = True):

        if k_fold == True and n_folds == 1:
            raise ValueError("Impossible to perform k-fold with 1 fold. Specify number of folds with parameter n_fold > 1")
        
        if not validation and k_fold:
            raise ValueError("Impossible to perform k-fold. If you want to perform k-fold validation set the parameter validation = True, else set the parameter k-fold = False")
        
        if (not validation or (validation and not k_fold)) and parallel:
            print("Impossible to enable parallelism. Only available with k-fold")
        
        if not validation:
            losses, val_losses, scores, val_scores = self.train_no_validate(print_progress = print_progress, compute_score=compute_score)
        
        if validation and not k_fold:
            losses, val_losses, scores, val_scores, stats_res = self.train_and_validate(test_size=test_size, print_progress = print_progress, compute_score=compute_score)
        
        if k_fold and n_folds > 1:
            losses, val_losses, scores, val_scores, stats_res = self.k_fold_cross_val(n_folds = n_folds, print_progress = print_progress, parallel = parallel, compute_score=compute_score)

        self.losses = losses
        self.val_losses = val_losses
        self.scores = scores
        self.val_scores = val_scores

        if print_results:
            print_training(losses = self.losses, val_losses = self.val_losses, scores = self.scores, val_scores = self.val_scores)

        if plot_results:
            plot_training(losses = self.losses, val_losses = self.val_losses, scores = self.scores, val_scores = self.val_scores)
        
        if print_res_stats and validation:
            print_residuals(data = stats_res)

        return self.losses, self.val_losses, self.scores, self.val_scores


    def save_to_excel(self, path_dir, file_name):
        
        if self.losses is None:
            raise ValueError("Cannot save prior training")
        if path_dir is None:
            raise ValueError("Please provide path directory")
        if file_name is None:
            raise ValueError("Please provide file name")

        if not file_name.endswith('.xlsx'):
            raise TypeError("Type file not corret. Supported only .xlsx")
        
        data = {}
        filepath = os.path.join(path_dir,file_name)

        # save file of final training
        data['epoch'] = [x+1 for x in range(len(self.losses))]
        data['loss'] = self.losses
        data['val_loss'] = self.val_losses
        data['score'] = self.scores
        data['val_score'] = self.val_scores
        df = pd.DataFrame(data)

        df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"\nData saved to {filepath}")


    def save_model(self, path_dir: str, file_name: str = "metamodel.pth", save_data_scaler: bool = True, scaler_file_name: str = 'data_scaler.pth'):
        if self.best_model_state is None:
            raise ValueError("No best model to save found. Train the model before saving.")

        if not path_dir or not os.path.exists(path_dir):
            raise ValueError(f"Invalid or non-existent path: {path_dir}")

        if not file_name or not file_name.endswith('.pth'):
            raise ValueError("Please provide a valid .pth filename.")

        metadata = self.model_instance.get_metadata()

        # Save model
        self.model_instance.load_state_dict(self.best_model_state)
        model_path = os.path.join(path_dir, file_name)
        torch.save({'model_state': self.model_instance.state_dict(), 'metadata': metadata}, model_path)
        print(f"\nModel saved to {model_path}")

        # Save scalers if requested
        if save_data_scaler:
            path_scaler = os.path.join(path_dir, scaler_file_name)
            joblib.dump(self.scalers, path_scaler)
            print(f"\nScaler data saved to {path_scaler}")