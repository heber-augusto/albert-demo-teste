from source.code import *


def test_load_generative_model():
    # Teste de carga do modelo generative ai
    ret_model = load_generative_model(model_dir = None)
    assert ret_model != None


def test_fine_tune_model():
    # Teste de fine_tune do modelo
    model = None
    input_data = None
    output_dir = None    
    inference = fine_tune_model(model, input_data, output_dir)
    assert inference != None    