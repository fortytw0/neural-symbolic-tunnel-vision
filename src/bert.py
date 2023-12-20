import os
import pickle
import json
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer

'''
Variable Declaration
'''

train_data_path = 'data/docnli/train_5sent_50ksample.json'
dev_data_path = 'data/docnli/dev_5sent_10ksample.json'

save_dir = 'data/bert'
train_data_output_dir = 'data/bert/train'
dev_data_output_dir = 'data/bert/dev'
batch_size = 2

model_name = 'bert-base-cased'
tokenizer_name = 'bert-base-cased'

'''
Folder Creation
'''

if not os.path.exists(save_dir) : 
    os.mkdir(save_dir)

if not os.path.exists(train_data_output_dir) : 
    os.mkdir(train_data_output_dir)

if not os.path.exists(dev_data_output_dir) : 
    os.mkdir(dev_data_output_dir)


'''
Load Models
''' 

model = LongformerModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model = model.to("cuda")

'''
Read data
'''

train_data = json.load(open(train_data_path))
dev_data = json.load(open(dev_data_path))

'''
Function Definition
'''

def batch(iterable , batch_size) : 
    l = len(iterable)
    for i in range(0, l, batch_size) : 
        yield iterable[i:min(l, i+batch_size)]

def save_dict_as_pickle(d , save_path) : 
    with open(save_path, 'wb') as f : 
        pickle.dump(d, f)

def get_cls(text) : 

    tokens = tokenizer(text, padding='max_length' , truncation=True, return_tensors='pt')
    
    input_ids = tokens.input_ids.to("cuda")
    attention_mask = tokens.attention_mask.to("cuda")

    cls = model(input_ids=input_ids, attention_mask=attention_mask)[:,0,:].cpu().detach().numpy()

    return cls


'''
Train Loop
'''

start_index = int(len(os.listdir(train_data_output_dir))/batch_size)
try : 
    train_batches = [train_batch for train_batch in batch(train_data, batch_size)][start_index : ]

    for i , data in tqdm(enumerate(train_batches)) : 


        premise = [sent_tokenize(d['premise']) for d in data]
        hypothesis = [sent_tokenize(d['hypothesis']) for d in data]






        




        
        

        for i , d in enumerate(data) : 
            d['repr'] = longformer_repr[i]
            save_path = os.path.join(train_data_output_dir, f"{d['id']}.pkl")
            save_dict_as_pickle(d , save_path)

except Exception as e : 

    print(f"Error encountered in Train Loop Batch : {i+1}/{len(train_batches)} ")
    print(e)

'''
Dev Loop
'''

start_index = int(len(os.listdir(dev_data_output_dir))/batch_size)
try : 
    dev_batches = [dev_batch for dev_batch in batch(dev_data, batch_size)][start_index : ]

    for i , data in tqdm(enumerate(dev_batches)) : 

        premise_hypothesis_pairs = [(d['premise'] , d['hypothesis']) for d in data]

        tokens = tokenizer(premise_hypothesis_pairs, padding='max_length' , truncation=True, return_tensors='pt')
        
        input_ids = tokens.input_ids.to("cuda")
        attention_mask = tokens.attention_mask.to("cuda")

        longformer_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        longformer_repr = longformer_repr.last_hidden_state[:,0,:].cpu().detach().numpy()

        for i , d in enumerate(data) : 
            d['repr'] = longformer_repr[i]
            save_path = os.path.join(dev_data_output_dir, f"{d['id']}.pkl")
            save_dict_as_pickle(d , save_path)

except Exception as e : 

    print(f"Error encountered in Val Loop Batch : {i+1}/{len(train_batches)} ")
    print(e)