from dataclasses import dataclass
from socketserver import DatagramRequestHandler
from torch.utils.data import Dataset
import os.path as osp
import os
import pandas as pd
from transformers import AutoTokenizer

class EnFr(Dataset):
    def __init__(self, data_folder, en_file, fr_file, s_key, t_key):
        self.data_folder = osp.abspath(data_folder)
        self.s_key = s_key
        self.t_key = t_key
        en_raw = osp.join(self.data_folder, en_file)
        fr_raw = osp.join(self.data_folder, fr_file)
        raw_data = {s_key: [line for line in en_raw], t_key: [line for line in fr_raw]}
        df = pd.DataFrame(raw_data, columns = [s_key, t_key])
        self.tokenizer = AutoTokenizer.from_pretrained('t5_small', model_max_length=512)
        self.df = self.tokenize(df)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        source = self.df[self.s_key][idx]
        target = self.df[self.t_key][idx]
        return source, target

    def tokenize(self, df, s_key='en', t_key='fr', max_length='2048'):
        s = [sentence for sentence in df[s_key]]
        t = [sentence for sentence in df[t_key]]
        
        data = self.tokenizer(s, max_length=max_length, truncation=True, padding=True, return_tensors='pt') 
        
        with self.tokenizer.as_target_tokenizer():
            target_tokens = self.tokenizer(t, truncation=True, padding=True, max_length=max_length, return_tensors='pt') 
            
        data['target'] = target_tokens['input_ids']
        
        return data