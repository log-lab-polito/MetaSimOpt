import torch
from MetaSimOpt.utils import _generate_tensors, _normalise_dataset, _load_model_from_file, _load_model_from_memory, _sort_features

class HandlerMetamodel:

    def __init__(self, device = "cpu"):
        self.device = device
        self.metamodel = None 
        self.x = None 
        self.scalers = None

    def load_model_from_memory(self, trained_model, scalers):
        
        self.metamodel, self.scalers = _load_model_from_memory(trained_model = trained_model, scalers = scalers)


    def load_model_from_file(self, dir_metamodel, file_metamodel):
                     
        self.metamodel, self.scalers = _load_model_from_file(dir_metamodel = dir_metamodel, file_metamodel = file_metamodel)


    def load_data(self, data):

        if self.scalers:
            data, _, _ = _normalise_dataset(x=data, scalers=self.scalers)
        else:
            print("Data loaded without scaling features. Call load_model_from_file first to load data scalers.")    
        
        self.x = _generate_tensors(data=data, device=self.device)


    def predict(self, x = None, mc_samples = None):

        if x is None:
            _x = self.x
        else:
            if not isinstance(x[0], torch.Tensor):
                _x = _generate_tensors(x, device=self.device)

        if self.metamodel is None:
            raise ValueError("Load model before predict. Call load_model_from_memory or load_model_from_file.")
        else:
            if not mc_samples:
                if hasattr(self.metamodel, "predict"):
                    predictions = self.metamodel.predict(x = _x)
                else:
                    raise ValueError("Prediction not supported by the metamodel loaded. Implement a method")
            else:
                if hasattr(self.metamodel, "predict_mc_dropout"):
                    predictions = self.metamodel.predict_mc_dropout(x = _x, n_samples = mc_samples)
                else:
                    raise ValueError("Montecarlo dropout not supported by the metamodel loaded. Implement a method")

        return predictions

    
    def _normalise(self, x):

        norm_x = []

        for key, value in x.items():
            if key == 'features_rec':
                min_scaler = self.scalers['min_scaler_rec']
                max_scaler = self.scalers['max_scaler_rec']
                if min_scaler is None or max_scaler is None:
                    continue
                else:
                    norm_data = (value - min_scaler) / (max_scaler - min_scaler + 1e-8)
                    norm_x.append(norm_data)

            elif key == 'features_lin':
                min_scaler = self.scalers['min_scaler_lin']
                max_scaler = self.scalers['max_scaler_lin']
                if min_scaler is None or max_scaler is None:
                    continue
                else:
                    norm_data = (value - min_scaler) / (max_scaler - min_scaler + 1e-8)
                    norm_x.append(norm_data)
        
        return norm_x
    

    def instanciate_model(self, factory, model_hyperparameters):

        if model_hyperparameters is None:
            raise ValueError(f"Set model hyperparameters before istanciate model. Call set_model_hyperparameters()")
        else:
            factory.default_args['hyperparameters'] = model_hyperparameters
            model_instance = factory.create()
            model_instance.to(self.device)
        
        return factory.model_class, model_instance
        

