import os
import torch
import importlib
import joblib

def _load_model_from_memory(trained_model, scalers, device = "cpu"):

    metamodel = trained_model.to(device)
    metamodel.eval()
    
    return metamodel, scalers
    

def _load_model_from_file(dir_metamodel, file_metamodel, device = "cpu"):

    path_file_metamodel = os.path.join(dir_metamodel, file_metamodel)
    
    if not os.path.exists(path_file_metamodel):
        raise ValueError("File metamodel not found")

    with open(path_file_metamodel, 'rb') as f:
        partial_checkpoint = torch.load(f, map_location = device, weights_only = False)
    
    metadata = partial_checkpoint['metadata']              
    checkpoint = torch.load(path_file_metamodel, map_location = device, weights_only = False)

    model_state = checkpoint['model_state']
    metadata = checkpoint['metadata']
    module = importlib.import_module(metadata["model_module"])
    class_metamodel = getattr(module, metadata["model_class"])
    
    metamodel = None
    if hasattr(class_metamodel, "from_metadata"):
        metamodel = class_metamodel.from_metadata(metadata)
        metamodel.load_state_dict(model_state)
        metamodel.to(device).eval()

    # search data scalers
    found = False
    scalers = None
    for file in os.listdir(dir_metamodel):
        if "scaler" in file:
            found = True
            print(f'Found scaler {file}')
            path_file = os.path.join(dir_metamodel, file)
            if os.path.exists(path_file):
                scalers = joblib.load(path_file)

    if not found:
        print('Scaler not found')

    return metamodel, scalers