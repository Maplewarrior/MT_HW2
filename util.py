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

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=d_model)


def preprocess_data(df, s_key, t_key, max_length):
    setup_seed(42)
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

def train_model(model, data, epochs, verbose=True):
    model.train()
    start = time.time()
    total_loss = 0
    
    source_all = data['input_ids']
    target_all = data['target']
    
    # loop over epochs
    for epoch in range(epochs):
        
        # loop over all sentences
        for i in range(len(source_all)):
            
            # unsqueeze to avoid dim mismatch between embedder and pe
            src = source_all[i].unsqueeze(0)
            trg = target_all[i].unsqueeze(0)
            size = len(trg)
            print("sizeeee", size)
            
            source_pad = source_all[i] == 0
            
            target_pad = target_all[i] == 0
            
            input_msk = (source_all[i] != source_pad).unsqueeze(1)
            
            # trg_ipt = trg[:, :-1]
            # targets = trg[:, 1:].contiguous().view(-1)
            
            nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
            nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)
            
            target_msk = (target_all[i] != target_pad).unsqueeze(1)
            target_msk = target_msk & nopeak_mask
            
            print("getting preds...")
            # preds = model.forward(src, trg , None, None)
            preds = model.forward(src, trg, input_msk, target_msk)
            print("preds gotten...")
            optim.zero_grad()    
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ignore_idx=target_pad)
            loss.backward()
            optim.step()
            total_loss += loss.data[0]
            if verbose:
                print("time =",time.time()-start, "\n loss:", loss.data[0], "\n total loss:", total_loss)