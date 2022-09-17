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


def setup_seed(seed):
    random.seed(seed)                          
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True  

#%%
setup_seed(42)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
### Load data ###
data_en = open("data/hansards.e", encoding='utf-8').read().split('\n')
data_fr = open("data/hansards.f", encoding='utf-8').read().split('\n')


raw_data = {'en': [line for line in data_en], 'fr': [line for line in data_fr]}

df = pd.DataFrame(raw_data, columns = ['en', 'fr'])

df_small = df[['en', 'fr']][:100]



print(df_small.head())


#%%
eng = df_small['en']

eng_list = [sentence for sentence in eng]

# model_input['input_ids'] #.unsqueeze(0)

#%%


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        setup_seed(42)
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=1024):
        super().__init__()
        setup_seed(42)
        self.d_model = d_model
        
        # Allocate memory to 
        pe = torch.zeros((max_seq_len, d_model))
        
        ### From attention is all you need ###
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos, i] = np.sin(pos/10000**(2*i/self.d_model))
                pe[pos, i+1] = np.cos(pos/10000**(2*i/self.d_model))
        # Fixed positional encoding
        pe.requires_grad = False
        pe = pe.unsqueeze(0) # Make pe into [batch size x seq_len x d_model]
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        setup_seed(42)
        # Make embeddings larger
        x = x*np.sqrt(self.d_model)
        # Get sequence length
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], 
        requires_grad=False)
        return x


def Attention(Q, K, V, d_k, mask=None, dropout=None):
    setup_seed(42)

    vals = (Q @ K.transpose(-2,-1))/np.sqrt(d_k)
    
    # Mask the scores if mask is specified. Model cannot see into future if masked.
    if mask is not None:
        mask = mask.unsqueeze(1)
        vals = vals.masked_fill(mask, 1e-5)
    # vals = vals if mask is None else vals.masked_fill_(mask, 1e-4)
    
    softmax = nn.Softmax(dim=-1)
    vals = softmax(vals)
    
    # apply dropout if specified
    vals = vals if dropout is None else dropout(vals)
    
    out =  vals @ V
    return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, dropout=.1, relative = False):
        super().__init__()
        setup_seed(42)
        self.n_heads = n_heads
        self.d_model = d_model
        # self.seq_len = seq_len
        self.d_k = d_k
        
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    # d_model = 512
    # n_heads = 8
    # seq_len = 20
    
    # [20,512] --> [20, 8, 64]
    ## If batch size is used, say of 128:
    ## out = [128, 20, 8, 64]
    
    # Input = Matrix of dim [bs x seq_len x d_model]
    def split_heads(self, t):
        return t.reshape(t.size(0), -1, self.n_heads, int(self.d_k))
    # Output = Matrix of dim [bs x seq_len x n_heads x d_k]
    
    def forward(self, Q, K, V, mask = None):
        setup_seed(42)
        Q = self.linear(Q)
        K = self.linear(K)
        V = self.linear(V)
        
        Q, K, V = [self.split_heads(t) for t in (Q, K, V)] 
        Q, K, V = [t.transpose(1,2) for t in (Q, K, V)] # reshape to [bs x n_heads x seq_len x d_k]
        
        # Compute Attention
        vals = Attention(Q, K, V, self.d_k, mask, self.dropout)
        
        # Reshape to [bs x seq_len x d_model]
        vals = vals.transpose(1,2).contiguous().view(vals.size(0), -1, self.d_model)
       
        out = self.out(vals) # linear
        return out
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super().__init__()
        setup_seed(42)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        setup_seed(42)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

        
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, dropout=.1):
        setup_seed(42)
        super().__init__()
        self.MHA = MultiHeadAttention(n_heads, d_model, d_k, dropout)
        self.FFN = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        setup_seed(42)
        
        # See "Attention is all you need" to follow code structure
        
        x2 = self.dropout1(self.MHA(x, x, x, mask))
        x = self.norm1(x) + self.norm1(x2)
        
        x2 = self.dropout2(self.FFN(x))
        x = x + self.norm2(x2)
    
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, dropout=.1):
        setup_seed(42)
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.MHA = MultiHeadAttention(n_heads, d_model, d_k, dropout)
        self.FFN = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Batch Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # self.linear = nn.Linear()
        
    def forward(self, x, e_out, source_mask, target_mask):
        setup_seed(42)
        
        # See "Attention is all you need" to follow code structure
        ## part 1
        x2 = self.norm1(x) # Norm
        x = self.dropout1(self.MHA.forward(x2, x2, x2, target_mask)) # Masked MHA, target
        x = x2 + self.norm1(x) # Add & Norm
        
        ## part 2
        x3 = self.dropout2(self.MHA.forward(x, e_out, e_out, source_mask)) # MHA on encoder output
        x2 = self.dropout2(self.MHA.forward(x, x, x)) #MHA continued in decoder
        x = self.norm2(x3) + self.norm2(x2) + self.norm2(x) # Add & Norm
        
        ## part 3
        x2 = self.dropout3(self.FFN.forward(x)) ## Feed forward
        
        x = x + self.norm3(x2) # add
        # x = self.norm3(x) # norm (!!!CHECK IF THIS IS EQUIVALENT!!!)
        return x

def cloneLayers(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, n_layers, n_heads, dropout=.1):
        super().__init__()
        self.n_layers = n_layers
        self.embedder = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.e_layers = cloneLayers(EncoderLayer(n_heads, d_model, d_ff, d_k), n_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, source, mask=None):
        x = self.embedder.forward(source)
        x = self.pe.forward(x)
        for i in range(self.n_layers):
            x = self.e_layers[i](x, mask)
        
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, n_layers, n_heads, dropout=.1):
        super().__init__()
        self.n_layers = n_layers
        self.embedder = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.d_layers = cloneLayers(DecoderLayer(n_heads, d_model, d_ff, d_k), n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, trg, e_out, source_mask, target_mask):
        x = self.embedder.forward(trg)
        x = self.pe.forward(x)
        
        for i in range(self.n_layers):
            x = self.d_layers[i](x, e_out, source_mask, target_mask)
        
        return self.norm(x)
        
    
    
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model,d_ff, d_k, n_layers, n_heads):
        super().__init__()
        self.e = Encoder(source_vocab_size, d_model,d_ff, d_k, n_layers, n_heads)
        self.d = Decoder(target_vocab_size, d_model,d_ff, d_k, n_layers, n_heads)
        self.linear_f = nn.Linear(d_model, target_vocab_size)
        
    def forward(self, source, target, source_mask, target_mask):
        e_out = self.e.forward(source, source_mask)
        d_out = self.d.forward(target, e_out, source_mask, target_mask)
        
        out = self.linear_f(d_out)
        return out
        


#%%
### Define arguments ### (same as in "Attention is all you need")
d_model = 512 # Dimension of embeddings
d_k = 64 # dimension of keys (d_model / n_heads)
d_ff = 2048
vocab_size = len(df) # Number of (unique) words in dataset
n_heads = 8 # Number of heads for MHA
n_layers = 6 # Number of model layers
train_iter = 5
model_checkpoint = 't5-small'


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=vocab_size)


# Function for mapping data from strings to tokens
# s_key = source key, t_key = target_key
def preprocess_data(df, s_key, t_key, max_length):
    setup_seed(42)
    s = [sentence for sentence in df[s_key]]
    t = [sentence for sentence in df[t_key]]
    
    model_input = tokenizer(s, max_length=max_length, truncation=True, padding=True, return_tensors='pt') 
    
    with tokenizer.as_target_tokenizer():
        target_tokens = tokenizer(t, truncation=True, padding=True, max_length=max_length, return_tensors='pt') 
        
    model_input['target'] = target_tokens['input_ids']
    
    return model_input

ipt = preprocess_data(df_small, 'en', 'fr', max_length=36)



src_vocab_size = [word for sentence in ipt['input_ids'] for word in sentence]
src_vocab_size = len(np.unique(src_vocab_size))

trg_vocab_size = [word for sentence in ipt['target'] for word in sentence]
trg_vocab_size = len(np.unique(trg_vocab_size))






"""
Note to self:
    Adjust get_target_mask to find where padding occurs first and ignore 
    everything that comes after that.
    
    --> I.e. find the i where trg[i] == 0 the first time and ignore i+1
"""

def get_target_mask(size, target):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(0)).masked_fill(mask == 1, float(1))
    
    # Find out where the target pad starts
    trg_pad = (target==0).nonzero()
    
    # Check if there is no padding in sentence
    if len(trg_pad) == 0:
        stop_idx = size
        
    
    else:
        stop_idx = trg_pad[0][1].item()
        mask[stop_idx:, :] = -69
    
    
    return mask.unsqueeze(0) > 0, stop_idx

"""
Functionality:
    Find where padding starts in source.
    Generate mask such that everything is ignored after the first padding seen.
"""
def get_source_mask(size, source):
    src_pad = (source==0).nonzero()
    
    if len(src_pad == 0):
        stop_idx = size
        
    else:
        stop_idx = src_pad[0][1].item()
    
    mask = source.clone()
    # Mask all padding with -inf
    mask[:, stop_idx:] = 0
    # Convert everything before stop_idx to zero
    mask[:,:stop_idx] = 1
    
    mask = mask.unsqueeze(0) > 0
    
    return mask




model = Transformer(vocab_size, vocab_size, d_model, d_ff, d_k, n_layers, n_heads)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
        
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Test for a single sentence
# out = model.forward(en1, trg_input, en1_msk, fr1_msk)

# loss = F.cross_entropy(out.view(-1, out.size(-1)), y)

#%%
print_every = 10
def train_model(model, data, epochs, verbose=False):
    model.train()
    start = time.time()
    total_loss = 0
    
    source_all = data['input_ids']
    target_all = data['target']
    
    # loop over epochs
    for epoch in range(epochs):
        print("epoch", epoch+1)
        # loop over all sentences
        for i in range(len(source_all)):
            if i % print_every == 0:
                print("sentece", i+1)
            
            # unsqueeze to avoid dim mismatch between embedder and pe
            src = source_all[i].unsqueeze(0) 
            trg = target_all[i].unsqueeze(0)
            
            # target input, remove last word
            trg_input = trg[:, :-1]
            
            # get targets
            y = trg[:, 1:].contiguous().view(-1)
            
            src_mask = get_source_mask(src.size(1), src)
            trg_mask, trg_stop_idx = get_target_mask(trg_input.size(1), trg_input)
            
            preds = model.forward(src, trg_input, src_mask, trg_mask)
            optim.zero_grad()    
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            
            if i % print_every ==0:
                
                if verbose:
                    print("time:",np.round(time.time()-start, 2), "\n loss:", loss.item(), "\n total loss:", total_loss)

#%%
epochs = 1
train_model(model, ipt, epochs, True)             



