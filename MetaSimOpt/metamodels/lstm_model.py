from .master_metamodel import Metamodel
import torch.nn as nn

class LSTM_Metamodel(Metamodel):
    def __init__(self, input_size_rec, input_size_lin, output_size, hyperparameters, max_seq_length):
        super().__init__(input_size_rec, input_size_lin, output_size, rec_cell = nn.LSTM, hyperparameters = hyperparameters, max_seq_length = max_seq_length)