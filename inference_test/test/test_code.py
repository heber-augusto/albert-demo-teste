from source.code import *


def test_load_generative_model():
    # Teste de verificação de testes de InferenceJobType
    model_dir = r'./source/'
    ret_model = load_generative_model(model_dir)
    assert ret_model != None


def test_perform_inference():
    # Teste de verificação de testes de InferenceJobType
    model_dir = r'./source/'
    ret_model = load_generative_model(model_dir)
    input_content = 'test'
    inference = perform_inference(input_content, ret_model)
    assert inference != None    