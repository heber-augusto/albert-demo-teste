
import pandas as pd
from datasets import DatasetDict, Dataset, Features, features, Translation
from datasets import load_from_disk
import os
import datasets
import random

# Leitura do dataset ParaCrawl99
traducoes_en_pt = pd.read_pickle(r'https://github.com/unicamp-dl/Lite-T5-Translation/raw/master/ParaCrawl99k/ParaCrawl99K_EnPt_PCrawlGoogleT.pkl')
df_traducoes_en_pt = pd.DataFrame(traducoes_en_pt, columns=['en','pt'])

# Função para converter o dataframe pandas em um Dataset do Hugging Face
# essa função foi criada com o apoio do chatgpt (não encontrei exemplos na documentação)
def pandas_to_huggingface_dataset(df):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda example: {'translation': {'en': example['en'], 'pt': example['pt']}}, remove_columns=['en', 'pt'])
    
    # Definir o esquema do Dataset
    dataset = dataset.cast(Features({'translation': Translation(languages=('en', 'pt'))}))
    
    return dataset

# salva arquivo de traducoes en_pt
hf_dataset = pandas_to_huggingface_dataset(df_traducoes_en_pt)
input_dir = os.path.dirname(os.path.abspath(__file__))
hf_dataset.save_to_disk(os.path.join(input_dir,"traducoes_en_pt.hf"))

# #1. Separação em treino, validação e teste
# Separe o conjunto de dados ParaCrawl99k em 80% para treino, 10% para validação e 10% para teste. 
# Todos os experimentos intermediários devem ser avaliados no conjunto de validação e, apenas ao final, reportar 
# o resultado do melhor modelo no conjunto de teste.

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


def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset)
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    return df


df_to_display = show_random_elements(ds["train"])

for row in df_to_display['translation']:
    print(row)