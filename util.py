import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
# import pytorch_forecasting.utils.create_mask as create_mask
import random
import copy
import time

tokenizer = AutoTokenizer.from_pretrained('t5_small', model_max_length=512)

def cloneLayers(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

def preprocess_data(df, s_key, t_key, max_length):
    s = [sentence for sentence in df[s_key]]
    t = [sentence for sentence in df[t_key]]
    
    model_input = tokenizer(s, max_length=max_length, truncation=True, padding=True, return_tensors='pt') 
    
    with tokenizer.as_target_tokenizer():
        target_tokens = tokenizer(t, truncation=True, padding=True, max_length=max_length, return_tensors='pt') 
        
    model_input['target'] = target_tokens['input_ids']
    
    return model_input

def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True

