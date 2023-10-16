from source.code import *


def test_load_generative_model():
    model_dir = ''
    parameters = ''
    ret_model = load_generative_model(model_dir)
    assert ret_model != None


def test_perform_inference():
    # Teste de verificação de testes de InferenceJobType
    ret_model = None
    input_content = None
    inference = perform_inference(input_content, ret_model)
    assert inference != None    