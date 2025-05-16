# custom lib with utilities
"""
How to implement this library:
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
def test1():
    print("1")

def test2():
    print("2")

def test3():
    print("3")

def test4():
    print("Prueba correcta")