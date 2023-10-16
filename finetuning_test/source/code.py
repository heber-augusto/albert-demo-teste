import os

from datasets import DatasetDict, Dataset, Features, Translation
from datasets import load_from_disk

from transformers import T5Tokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay





def preprocess_function(examples, tokenizer):
    source_lang = "en"
    target_lang = "pt"
    prefix = "translate English to Portuguese: "
    max_length=64

    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(
        inputs, 
        padding="max_length",
        max_length=max_length, 
        truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            padding="max_length",
            max_length=max_length, 
            truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_output_texts(model, tokenizer, tf_set, max_length):
    output_texts = []
    for batch_idx, batch in enumerate(tf_set):
        print(f'generating output: {batch_idx}')
        source_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_ids = batch['labels']
        outputs = model.generate(
            source_ids, 
            attention_mask=attention_mask, 
            do_sample=False, 
            max_length=max_length)
        text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_texts.extend(text)

    return output_texts

def load_tokenizer():
    """
    Carrega o modelo de generative AI a ser fine-tuned.

    Retorna:
        model: Modelo carregado
    """
    model_name = 'unicamp-dl/ptt5-small-portuguese-vocab'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    #with tokenizer.as_target_tokenizer():
    #    print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))

    return tokenizer



def load_generative_model(model_dir = None):
    """
    Carrega o modelo de generative AI a ser fine-tuned.

    Retorna:
        model: Modelo carregado
    """

    model_name = 'unicamp-dl/ptt5-small-portuguese-vocab'
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model


def fine_tune_model(model, input_data, output_dir, parameters):
    """
    Realiza o fine-tuning do modelo de generative AI.

    Args:
        model: Modelo de generative AI carregado
        input_data (str): Caminho para os dados de entrada
        output_dir (str): Caminho para o diretório de saída
    """

    max_length=64
    batch_size = 64

    hf_dataset = load_from_disk(input_data)
    train_test_dict = hf_dataset.train_test_split(test_size=0.2)
    train = train_test_dict['train']
    test_and_val = train_test_dict['test']
    train_test_dict = test_and_val.train_test_split(test_size=0.5)
    test = train_test_dict['train']
    val = train_test_dict['test']

    ds = DatasetDict({
        'train': train,
        'test': test,
        'valid': val})

    tokenizer = load_tokenizer()
    tokenized_texts = ds.map(
        preprocess_function, 
        fn_kwargs={"tokenizer": tokenizer,}, 
        batched=True)   


    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        return_tensors="tf",
        padding='max_length',
        max_length=max_length)


    tf_train_set = tokenized_texts["train"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    # tf_test_set = tokenized_texts["test"].to_tf_dataset(
    #     columns=["attention_mask", "input_ids", "labels"],
    #     shuffle=False,
    #     batch_size=batch_size,
    #     collate_fn=data_collator,
    # )

    tf_val_set = tokenized_texts["valid"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )
    
    # output_texts = get_output_texts(
    #     model, 
    #     tokenizer, 
    #     tf_val_set, 
    #     max_length)
    
    # ref_pt = []
    # for elem in ds['valid']:
    #     ref_pt.append(elem['translation']['pt'])
    # bleu = evaluate.load("bleu")
    # results = bleu.compute(
    #     predictions=output_texts, 
    #     references=ref_pt)
    
    # print(results)


    #optimizer = AdamWeightDecay(learning_rate=0.01)
    #model.compile(optimizer=optimizer)
    #model.fit(
    #    x=tf_train_set, 
    #    validation_data=tf_val_set, 
    #    epochs=10)
    
    model.save_pretrained(os.path.join(output_dir, 'saved_model'))
    

    return model

def main():

    # Carregue o modelo
    generative_model = load_generative_model(model_dir = None)
    
    # Defina o caminho para os dados de entrada
    input_data = "./input/traducoes_en_pt.hf"  # Altere conforme necessário

    # Defina o diretório de saída
    output_dir = "./output"  # Altere conforme necessário

    # Realize o fine-tuning
    fine_tune_model(generative_model, input_data, output_dir)

if __name__ == "__main__":
    os.chdir('/home/ubuntu/projetos/albert-test/finetuning_test/source')
    main()
