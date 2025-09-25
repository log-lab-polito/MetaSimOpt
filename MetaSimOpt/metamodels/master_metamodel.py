import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import importlib
import numpy as np
from typing import List, Dict, Any, Union, Callable, Type


class Metamodel(nn.Module):
    """
    A flexible neural network model combining recurrent and feedforward blocks, 
    suitable for variable-length sequence data and structured inputs.

    Supports dynamic architecture definition through hyperparameters, and includes:
    - Recurrent block (GRU/LSTM/RNN)
    - Feedforward linear blocks (2 stages)
    - Monte Carlo dropout for uncertainty estimation
    - Utilities for hyperparameter search and metadata serialization
    """

    @staticmethod
    def get_model_hyperparameters() -> Dict[str, Any]:
        """
        Returns the default set of hyperparameters used to build the model.

        Returns:
            Dict[str, Any]: Dictionary containing default values for model hyperparameters.
        """

        max_rec_layers = 3
        max_lin_layers_1 = 3
        max_lin_layers_2 = 3

        hyperparameters = {
            "hid_rec_layers": max_rec_layers,
            "rec_dropout": 0.1,
            "hid_rec_size": 32,
            "hid_lin_layers_1": max_lin_layers_1,
            "hid_lin_layers_2": max_lin_layers_2,
            "bidirectional": True
        }

        for i in range(max_lin_layers_1):
            hyperparameters[f"hid_lin_size_1_{i}"] = 64
            hyperparameters[f"linear_dropout_1_{i}"] = 0.1

        for i in range(max_lin_layers_2):
            hyperparameters[f"hid_lin_size_2_{i}"] = 64
            hyperparameters[f"linear_dropout_2_{i}"] = 0.2

        hyperparameters['many_to_many'] = False

        return hyperparameters
    

    @staticmethod
    def get_search_space() -> Dict[str, Union[List[Any], Dict[str, Any]]]:
        """
        Returns the search space for hyperparameter optimization. The values must be compatible
        with `get_model_hyperparameters()`.

        Returns:
            Dict[str, Union[List[Any], Dict[str, Any]]]: A dictionary specifying candidate values
            and conditional activation for each hyperparameter.
        """
        # WARNING!!! The limit of the search space need to correspond to maximum values set in get_hyperparameters()

        ss = {
            "hid_rec_layers": [1, 2, 3],
            "hid_rec_size": [16, 32, 64],
            "rec_dropout": [0.0, 0.1, 0.2, 0.3],
            "hid_lin_layers_1": [1, 2, 3],
            "hid_lin_layers_2": [1, 2, 3],
            "bidirectional": [True, False],
        }

        model_hp = Metamodel.get_model_hyperparameters()
        keys = ['hid_rec_layers', 'hid_lin_layers_1', 'hid_lin_layers_2']
        for key in keys:
            if max(ss[key]) > model_hp[key]:
                raise ValueError(f"Incoherence between search space and allowed hyperparameters. Check {key}.")

        for i in range(max(ss["hid_lin_layers_1"])):
            ss[f"hid_lin_size_1_{i}"] = {
                "values": [32, 64, 128],
                "condition": lambda hp, i=i: hp.get("hid_lin_layers_1", 0) > i
            }
            ss[f"linear_dropout_1_{i}"] = {
                "values": [0.0, 0.1, 0.2, 0.3],
                "condition": lambda hp, i=i: hp.get("hid_lin_layers_1", 0) > i
            }

        for i in range(max(ss["hid_lin_layers_2"])):
            ss[f"hid_lin_size_2_{i}"] = {
                "values": [32, 64, 128],
                "condition": lambda hp, i=i: hp.get("hid_lin_layers_2", 0) > i
            }
            ss[f"linear_dropout_2_{i}"] = {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5],
                "condition": lambda hp, i=i: hp.get("hid_lin_layers_2", 0) > i
            }

        return ss
    
    
    @staticmethod
    def compute_lengths(*inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes the effective sequence lengths for each sample in a batch, based on non-zero entries.

        Args:
            *inputs (List[torch.Tensor]): The first tensor is expected to be of shape (B, T, D).

        Returns:
            torch.Tensor: A 1D tensor (B,) with lengths of each sequence.
        """

        inputs_1 = inputs[0]

        return (inputs_1.abs().sum(dim=2) != 0).sum(dim=1)
    

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        """
        Initializes the weights of recurrent and linear layers using best practices.

        Args:
            m (nn.Module): A PyTorch module to initialize.
        """

        if isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    
    @classmethod
    def from_metadata(cls: Type[nn.Module], metadata: Dict[str, Any]) -> nn.Module:
        """
        Constructs a model instance from a saved metadata dictionary.

        Args:
            cls (Type[nn.Module]): The class itself (used for dynamic loading).
            metadata (Dict[str, Any]): Dictionary containing all model configuration parameters.

        Returns:
            nn.Module: An instance of the model.
        """

        module = importlib.import_module(metadata["model_module"])
        model_class = getattr(module, metadata["model_class"])
        return model_class(
            input_size_rec = metadata["input_size_rec"],
            input_size_lin = metadata["input_size_lin"],
            output_size = metadata["output_size"],
            hyperparameters = metadata["hyperparameters"],
            max_seq_length = metadata["max_seq_length"]
        )


    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns a dictionary with metadata needed to reconstruct the model.

        Returns:
            Dict[str, Any]: A dictionary including class name, module path, dimensions,
            and hyperparameters.
        """

        return {
            "model_class": self.__class__.__name__,
            "model_module": self.__class__.__module__,
            'input_size_rec' : self.input_size_rec,
            'input_size_lin' : self.input_size_lin,
            'output_size' : self.output_size,
            'max_seq_length' : self.max_seq_length,
            'hyperparameters' : self.hp
        }

    
    def __init__(
        self,
        input_size_rec: int,
        input_size_lin: int,
        output_size: int,
        rec_cell: Type[nn.Module],
        hyperparameters: Dict[str, Any],
        max_seq_length: int
    ) -> None:
        """
        Initializes the Metamodel architecture with recurrent and linear layers.

        Args:
            input_size_rec (int) : Input feature size for the recurrent block.
            input_size_lin (int) : Input feature size for the first linear block.
            output_size (int) : Number of output features.
            rec_cell (Type[nn.Module]) : Recurrent cell class (e.g., nn.LSTM or nn.GRU).
            hyperparameters (Dict[str, Any]) : Dictionary of architecture hyperparameters.
            max_seq_length (int) : Maximum input sequence length (for padding).
        """
              
        super().__init__()

        self.rec_cell = rec_cell
        self.input_size_rec = input_size_rec
        self.input_size_lin = input_size_lin
        self.output_size = output_size
        self.max_seq_length = max_seq_length
        self.hp = hyperparameters

        # check hyperameters
        default_hp = Metamodel.get_model_hyperparameters()
        for param in default_hp.keys():
            if param not in hyperparameters:
                hyperparameters[param] = default_hp[param]
                print(f"Missing required hyperparameter: {param}. Set to default value {default_hp[param]}")

        # recurrent block
        hid_rec_layers = self.hp["hid_rec_layers"]
        hid_rec_size = self.hp["hid_rec_size"]
        rec_dropout = self.hp["rec_dropout"] if hid_rec_layers > 1 else 0.0
        bidirectional = self.hp["bidirectional"]

        self.rec = rec_cell(
            input_size = input_size_rec,
            hidden_size = hid_rec_size,
            num_layers = hid_rec_layers,
            batch_first = True,
            bidirectional = bidirectional,
            dropout = rec_dropout
        )
        Metamodel.init_weights(self.rec)

        total_hidden_neurons_rec = hid_rec_size * (2 if bidirectional else 1)

        # linear block 1
        self.linears_1 = nn.ModuleList()
        self.dropouts_1 = nn.ModuleList()
        prev = input_size_lin

        for i in range(self.hp['hid_lin_layers_1']):
            size = self.hp[f'hid_lin_size_1_{i}']
            dropout = self.hp[f'linear_dropout_1_{i}']
            linear = nn.Linear(prev, size)
            Metamodel.init_weights(linear)
            self.linears_1.append(linear)
            self.dropouts_1.append(nn.Dropout(p=dropout))
            prev = size  

        total_hidden_neurons_linear = prev

        # linear block 2
        self.linears_2 = nn.ModuleList()
        self.dropouts_2 = nn.ModuleList()
        prev = total_hidden_neurons_rec + total_hidden_neurons_linear

        for i in range(self.hp['hid_lin_layers_2']):
            size = self.hp[f'hid_lin_size_2_{i}']
            dropout = self.hp[f'linear_dropout_2_{i}']
            linear = nn.Linear(prev, size)
            Metamodel.init_weights(linear)
            self.linears_2.append(linear)
            self.dropouts_2.append(nn.Dropout(p=dropout))
            prev = size  

        self.output_layer = nn.Linear(prev, output_size)
        Metamodel.init_weights(self.output_layer)
        

    def forward(self, x_rec: torch.Tensor, x_lin: torch.Tensor, lengths: Union[torch.Tensor, List[int]]) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_rec (torch.Tensor): Input tensor for the recurrent block, shape (B, T, D1).
            x_lin (torch.Tensor): Input tensor for the first linear block, shape (B, D2).
            lengths (Union[torch.Tensor, list[int]]): Sequence lengths for each sample in the batch, used to pack sequences.

        Returns:
            torch.Tensor: Model output. If output dimension is 1, returns raw output (B, 1). 
                        Otherwise returns cumulative sum over positive outputs (B, output_dim).
        """
        
        # recurrent block
        packed = pack_padded_sequence(x_rec, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rec(packed)
        x_rec, _ = pad_packed_sequence(packed_output, batch_first=True)

        many_to_many = self.hp['many_to_many']
        if not many_to_many:
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, x_rec.size(2))
            x_rec = x_rec.gather(1, idx).squeeze(1)
        else:
            pass

        # linear block 1
        num_lin1 = self.hp["hid_lin_layers_1"]
        for i in range(num_lin1):
            x_lin = F.relu(self.linears_1[i](x_lin))
            x_lin = self.dropouts_1[i](x_lin)

        if many_to_many:
            # expand x_lin to (B, T, D) for concat
            x_lin_exp = x_lin.unsqueeze(1).expand(-1, x_rec.size(1), -1)  # (B, T, D_lin)
            x = torch.cat((x_rec, x_lin_exp), dim=2)  # (B, T, hidden_dim + D_lin)
        else:
            x = torch.cat((x_rec, x_lin), dim=1)  # (B, hidden_dim + D_lin)

        # linear block 2
        num_lin2 = self.hp["hid_lin_layers_2"]
        for i in range(num_lin2):
            x = F.relu(self.linears_2[i](x))
            x = self.dropouts_2[i](x)

        # output
        if many_to_many:
            raw = self.output_layer(x)  # (B, T, out_dim)
            if self.output_layer.out_features == 1:
                return raw  # (B, T, 1)
            else:
                return torch.cumsum(F.relu(raw), dim=1)  # cumulated over T
        else:
            if self.output_layer.out_features == 1:
                return self.output_layer(x)  # (B, 1)
            else:
                raw = self.output_layer(x)  # (B, out_dim)
                return torch.cumsum(F.relu(raw), dim=1)
        

    def predict(self, x : list, device : str = "cpu") -> np.ndarray:
        """
        Perform a prediction using the model in evaluation mode.

        Args:
            x (list): List of torch.Tensors with model inputs. Expected format: [x_rec, x_lin].
            device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').

        Returns:
            np.ndarray: Model output as a NumPy array, shape depends on model configuration.
        """

        if not isinstance(x[0], torch.Tensor):
            raise ValueError("Generate torch tensor. Call generate_tensors()")
        self.eval()
        self.to(device)
        x = [i.to(device) for i in x]
        lengths = self.compute_lengths(*x)

        with torch.no_grad():
            output = self.forward(*x, lengths)

        return output.detach().cpu().numpy()
    

    def predict_mc_dropout(self, x : list, device : str ="cpu", n_samples : int =100) -> np.ndarray:
        """
        Perform multiple forward passes using Monte Carlo dropout to estimate uncertainty.

        Args:
            x (list): List of torch.Tensors with model inputs. Expected format: [x_rec, x_lin].
            device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
            n_samples (int): Number of stochastic forward passes to perform.

        Returns:
            np.ndarray: Array of predictions with shape (n_samples, B, output_dim), 
                        where output_dim depends on the model's final layer.
        """
        
        self.train()  # Attiva i dropout
        self.to(device)

        x = [i.to(device) for i in x]
        lengths = self.compute_lengths(*x)

        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Esegui forward pass con dropout attivo
                output = self.forward(*x, lengths)
                preds.append(output.detach().cpu().numpy())  # [1, B, output_size]

        preds = np.stack(preds, axis = -1)
        return np.squeeze(preds)