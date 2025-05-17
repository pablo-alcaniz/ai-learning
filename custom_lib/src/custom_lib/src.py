import torch

"""
How to implement this library in the desired working directory:
$ uv pip install -e /home/pablo/dev/ai-learning/00_custom_lib
$ uv add "custom_lib @ /home/pablo/dev/ai-learning/00_custom_lib/"

In this particular system an alias has been implemented: 
$ uv_install_cl ---> uv pip install...
$ uv_add_cl ---> uv add "...
"""



####################################################################################################
"""
Function to clear the weights of a PyTorch model. 
HTE (How to employ): 
    network.apply(clear_cache)
        -> network == model name
        -> clear_cache == function
"""
def clear_cache(model):
    reset_parameters = getattr(model, "reset_parameters", None)
    print(reset_parameters)
    if callable(reset_parameters):
        model.reset_parameters()

####################################################################################################

####################################################################################################
"""
Function to save a model parameters (weigths and biases).
NOTE_1: Variable SAVE_DIR should be an absolute path (if normal config is used) and must include the 
       desired filename with his extension (.pth)  
"""
def save_model(model, SAVE_DIR: str):
    torch.save(model.state_dict(), SAVE_DIR)

####################################################################################################

####################################################################################################
"""
Function to load model parameters (weights and biases)
NOTE_1: Variable SAVE_DIR should be an absolute path (if normal config is used) and must include the 
       desired filename with his extension (.pth)  
NOTE_2: The model architecture MUST BE the same as the one where the data come. 
"""
def load_model(model, SAVE_DIR: str):
    model.load_state_dict(torch.load(SAVE_DIR))

####################################################################################################




