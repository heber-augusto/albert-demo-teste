import os

from transformers import T5Tokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import pipeline

def load_generative_model(model_dir):
    """
    Função a ser implementada pelo usuário para carregar o modelo de Generative AI.

    Returns:
        model: O modelo de Generative AI carregado.
    """

    source_lang = "en"
    target_lang = "pt"

    model_path = os.path.join(model_dir, 'finetuned_model')
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

    model_name = 'unicamp-dl/ptt5-small-portuguese-vocab'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    en_pt_translator_pipeline = pipeline(
        f"translation_{source_lang}_to_{target_lang}", 
        model=model,
        tokenizer=tokenizer,
        max_length=64)
    print(f'chegou ao fim: {en_pt_translator_pipeline}')
    return en_pt_translator_pipeline

def perform_inference(input_content, generative_model):
    """
    Função a ser implementada pelo usuário para realizar a inferência usando o modelo de Generative AI.

    Args:
        input_content (str): O conteúdo de entrada para a inferência.
        generative_model: O modelo de Generative AI carregado.

    Returns:
        generated_text (str): O texto gerado pelo modelo de Generative AI.
    """
    return generative_model(input_content)[0]['translation_text']


#if __name__ == "__main__":
#    en_pt_translator_pipeline  = load_generative_model(model_dir = None)
#    print(perform_inference('translation test', en_pt_translator_pipeline))





