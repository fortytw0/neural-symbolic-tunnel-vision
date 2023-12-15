import os
import pickle
import json
from tqdm import tqdm
from transformers import LongformerModel, AutoTokenizer

'''
Variable Declaration
'''

train_data_path = 'data/docnli/train_5sent_50ksample.json'
dev_data_path = 'data/docnli/dev_5sent_10ksample.json'

train_data_output_dir = 'data/longformer/train'
dev_data_output_dir = 'data/longformer/dev'
batch_size = 2

'''
Folder Creation
'''

if not os.path.exists('data/longformer') : 
    os.mkdir('data/longformer')

if not os.path.exists(train_data_output_dir) : 
    os.mkdir(train_data_output_dir)

if not os.path.exists(dev_data_output_dir) : 
    os.mkdir(dev_data_output_dir)


'''
Load Models
''' 

model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

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

'''
Train Loop
'''

start_index = int(len(os.listdir(train_data_output_dir))/batch_size)
try : 
    train_batches = [train_batch for train_batch in batch(train_data, batch_size)][start_index : ]

    for data in tqdm(train_batches) : 

        premise_hypothesis_pairs = [(d['premise'] , d['hypothesis']) for d in data]

        tokens = tokenizer(premise_hypothesis_pairs, padding='max_length' , truncation=True, return_tensors='pt')
        
        input_ids = tokens.input_ids.to("cuda")
        attention_mask = tokens.attention_mask.to("cuda")

        longformer_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        longformer_repr = longformer_repr.last_hidden_state[:,0,:].cpu().detach().numpy()

        for i , d in enumerate(data) : 
            d['longformer_repr'] = longformer_repr[i]
            save_path = os.path.join(train_data_output_dir, f"{d['id']}.pkl")
            save_dict_as_pickle(d , save_path)

except Exception as e : 

    print(e)

'''
Dev Loop
'''

start_index = int(len(os.listdir(dev_data_output_dir))/batch_size)
try : 
    dev_batches = [dev_batch for dev_batch in batch(dev_data, batch_size)][start_index : ]

    for data in tqdm(dev_batches) : 

        premise_hypothesis_pairs = [(d['premise'] , d['hypothesis']) for d in data]

        tokens = tokenizer(premise_hypothesis_pairs, padding='max_length' , truncation=True, return_tensors='pt')
        
        input_ids = tokens.input_ids.to("cuda")
        attention_mask = tokens.attention_mask.to("cuda")

        longformer_repr = model(input_ids=input_ids, attention_mask=attention_mask)
        longformer_repr = longformer_repr.last_hidden_state[:,0,:].cpu().detach().numpy()

        for i , d in enumerate(data) : 
            d['longformer_repr'] = longformer_repr[i]
            save_path = os.path.join(dev_data_output_dir, f"{d['id']}.pkl")
            save_dict_as_pickle(d , save_path)

except Exception as e : 

    print(e)