import imp

import numpy as np

def model_select_nb(size, data):
    from my_models.NaiveBayesGuassianPractice1.nb_main import model_select
    
    model, percentage = model_select(size, data)
    return model, percentage

def model_select_nn(size, data):
    from my_models.NeuralNetworkClass.nn_main import model_select
    
    model, percentage = model_select(size, data)
    return model, percentage

def model_select_svm(size, data):
    from my_models.SupportVectorMachineClass.svm_main import model_select
    
    model, percentage = model_select(size, data)
    return model, percentage

