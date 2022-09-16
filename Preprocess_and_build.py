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

model_input['input_ids'] #.unsqueeze(0)

#%%


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        setup_seed(42)
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        setup_seed(42)
        # print("Embedder in shape: \n", x.size())
        out = self.embed(x)
        # print("After embed shape:\n", out.size())
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
        print("seq_len:", seq_len)
        v = torch.autograd.Variable(self.pe[:,:seq_len], 
        requires_grad=False)
        print("pe_shape:", v.size())
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], 
        requires_grad=False)
        return x


def Attention(Q, K, V, d_k, mask=None, dropout=None):
    setup_seed(42)

    
    vals = (Q @ K.transpose(-2,-1))/np.sqrt(d_k)
    # apply softmax
    softmax = nn.Softmax(dim=-1)
    vals = softmax(vals)
    
    # Mask the scores if mask is specified. Model cannot see into future if masked.
    vals = vals if mask is None else vals.masked_fill_(mask, 1e-4)
    
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
        # print("t size", t.size())
        return t.reshape(t.size(0), -1, self.n_heads, self.d_k)
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
        
        
        # print("before norm", x.size())
        # x2 = self.norm1(x) # norm
        # print("after norm", x2.size())
        # x = x + self.dropout1(self.MHA.forward(x2,x2,x2, mask)) #MHA
        # print("after MHA", x.size())
     
        # x2 = self.norm2(x)
        # x = self.dropout2(self.FFN.forward(x2))
        
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
        
    def forward(self, x, e_out, target_mask, source_mask):
        setup_seed(42)
        
        # See "Attention is all you need" to follow code structure
        
        # part 1
        x2 = self.norm1(x) # Norm
        x = self.dropout1(self.MHA.forward(x2, x2, x2, target_mask)) # Masked MHA, target
        x = x2 + self.norm1(x) # Add & Norm
        
        
        ## part 2
        x3 = self.dropout2(self.MHA.forward(x, e_out, e_out)) # MHA on encoder output
        x2 = self.dropout2(self.MHA.forward(x, x, x)) #MHA continued in decoder
        x = self.norm2(x3) + self.norm2(x2) + self.norm2(x) # Add & Norm
        
        ## part 3
        x2 = self.dropout3(self.FFN.forward(x)) ## Feed forward
        
        x = x + self.norm3(x2) # add
        # x = self.norm3(x) # norm (!!!CHECK IF THIS IS EQUIVALENT!!!)
        
        # x = self.linear(x) # DIMS???
        # x = self.softmax(x) #IMPLEMENT??
        
        return x

def cloneLayers(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, n_layers, n_heads, dropout=.1):
        super().__init__()
        self.n_layers = n_layers
        self.embedder = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.e_layers = cloneLayers(EncoderLayer(n_heads, d_model, d_ff, dropout), n_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, source, mask=None):
        print("INPUTT", source.size())
        x = self.embedder.forward(source)
        print("EMBEDD", x.size())
        
        print("PEE", self.pe.forward(x).size())
        x = self.pe.forward(x)
        
        
        for i in range(self.n_layers):
            print(self.e_layers[i](x, mask))
            x = self.e_layers[i](x, mask)
        
        print("Layers in Encoder:\n", x.size())
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, n_layers, n_heads, dropout=.1):
        super().__init__()
        self.n_layers = n_layers
        self.embedder = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.d_layers = cloneLayers(DecoderLayer(n_heads, d_model, d_ff, d_k), n_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, source, e_out, source_mask, target_mask):
        x = self.embedder.forward(source)
        x = self.pe.forward(x)
        
        for i in range(self.n_layers):
            x = self.d_layers[i](x, e_out, source_mask, target_mask)
        
        return self.norm(x)
        
    
    
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.e = Encoder(source_vocab_size, d_model,d_ff, n_layers, n_heads)
        self.d = Decoder(target_vocab_size, d_model,d_ff, d_k, n_layers, n_heads)
        
        self.linear_f = nn.Linear(d_model, target_vocab_size)
        
    def forward(self, source, target, source_mask, target_mask):
        e_out = self.e(source, source_mask)
        d_out = self.d(target, e_out, source_mask, target_mask)
        
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
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=d_model)



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

# flat_list = [item for sublist in l for item in sublist]



#%%
"""
Note to self:
    Adjust get_target_mask to find where padding occurs first and ignore 
    everything that comes after that.
    
    --> I.e. find the i where trg[i] == 0 the first time and ignore i+1
"""

def get_target_mask(size, target):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

"""
Find where padding starts in source.
Generate mask such that everything is ignored after the first padding seen.

"""
def get_source_mask(size, source):
    
    
    return mask

source_all = ipt['input_ids']
target_all = ipt['target']


src = source_all[0].unsqueeze(0)

trg = target_all[0].unsqueeze(0)

size = trg.size(1)

trg_mask = get_target_mask(size)


#%%

EMB = Embedder(vocab_size, d_model)
pe = PositionalEncoder(d_model, max_seq_len=d_model)
MHA = MultiHeadAttention(n_heads, d_model, d_k)
FFN = FeedForwardNetwork(d_model, d_ff)

encoderLayer = EncoderLayer(n_heads, d_model, d_ff, d_k)
# encoderLayer2 = EncoderLayer1(n_heads, d_model, d_ff)

decoderLayer1 = DecoderLayer(n_heads, d_model, d_ff, d_k)

encoder = Encoder(vocab_size, d_model,d_ff, n_layers, n_heads)

setup_seed(42)




#%%

sent1 = df_small['en'][0]
print(sent1)

e1 = ipt['input_ids'][0].unsqueeze(0)
f1 = ipt['target'][0].unsqueeze(0)
print(e1)
print("before embed", e1.size())
e1 = EMB.forward(e1)
f1 = EMB.forward(f1)
print("after embed", e1.size())

e1 = pe.forward(e1)
f1 = pe.forward(f1)
print("positional encode \n", e1.size())


Q, K, V = e1, e1, e1

print("performing MHA on source")
out = MHA.forward(Q, K, V)
print("out shape:", out.size())


out = FFN.forward(out)
print("FFN:\n", out.size())
out_e1 = encoderLayer(out)

print("encoderLayer out:\n",out_e1.size())



# out_d1 = decoderLayer1.forward(f1, out_e1, None, None)






#%%
print("method 1:")
print((ipt['input_ids'].unsqueeze(1).shape))
print((ipt['target'].unsqueeze(1).shape))


print(ipt['input_ids'].unsqueeze(1)[0].shape)
print("method2:")


# print("##### Example shown here #####")
EMB = Embedder(vocab_size, d_model)
pe = PositionalEncoder(d_model, max_seq_len=36)

e1 = df_small['en'][0]
f1 = df_small['fr'][0]


e1 = tokenizer(e1, truncation=True, padding=True, return_tensors='pt')
f1 = tokenizer(f1, truncation=True, padding=True, return_tensors='pt')

print("before embed:\n", e1['input_ids'].shape)

e1 = EMB.forward(e1['input_ids'])
print("after embed:\n", e1.size())
f1 = EMB.forward(f1['input_ids'])


print(f1.size())

p_e1 = pe(e1)






pe = PositionalEncoder(d_model, max_seq_len=d_model)
MHA = MultiHeadAttention(n_heads, d_model, d_k)
FFN = FeedForwardNetwork(d_model, d_ff)

encoderLayer = EncoderLayer(n_heads, d_model, d_ff, d_k)
# encoderLayer2 = EncoderLayer1(n_heads, d_model, d_ff)

decoderLayer1 = DecoderLayer(n_heads, d_model, d_ff, d_k)

encoder = Encoder(vocab_size, d_model,d_ff, n_layers, n_heads)

setup_seed(42)


### define model and initialize params ###
model = Transformer(vocab_size, vocab_size, d_model, n_layers, n_heads)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

### define optimizer ###
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


epochs = 5
batch_size = 10

# src = ipt['input_ids']
# trg = ipt['target']

# def batch(ipt, n=5):
#     l = len(ipt)
#     for ndx in range(0, l, n):
#         yield ipt[ndx:min(ndx + n, l)]


def get_mask(trg, trg_mask):
    return torch.masked_select(trg, trg_mask)

# target_seq = trg[0]
# size = len(target_seq)

# nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
# nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)


#%%
# for epoch in range(epochs):
    
T = Transformer(vocab_size, vocab_size, d_model, n_layers, n_heads)

preds = T.forward(ipt['input_ids'][0], ipt['target'][0], None, None)

#%%
def train_model(model_input, epochs, verbose=True):
    model.train()
    start = time.time()
    total_loss = 0
    
    source_all = model_input['input_ids']
    target_all = model_input['target']
    
    # loop over epochs
    for epoch in range(epochs):
        
        # loop over all sentences
        for i in range(len(source_all)):
            
            # unsqueeze to avoid dim mismatch between embedder and pe
            src = torch.tensor(source_all[i].unsqueeze(1)) 
            trg = torch.tensor(target_all[i].unsqueeze(1))
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



train_model(ipt, epochs)             
#%%
EMB = Embedder(vocab_size, d_model)
e1 = EMB(ipt['input_ids'][0].unsqueeze(1))
f1 = EMB(ipt['target'][0].unsqueeze(1))



print(e1.size())
print(f1.size())


print(ipt['target'][1]==0)
#%%
# 
# p_e1 = pe(e1)


# print("##### Example shown here #####")
# EMB = Embedder(vocab_size, d_model)

# e1 = df_small['en'][0]
# e2 = df_small['en'][2]
# f1 = df_small['fr'][0]

# e = tokenizer(e1, return_tensors='pt')
# e1 = tokenizer(e1, return_tensors='pt')
# e2 = tokenizer(e2, return_tensors='pt')

# # print(e1)

# e1 = EMB.forward(e1['input_ids'])
# p_e1 = pe(e1)
#%%
# def train_model(epochs, print_every=100):
    
#     model.train()
    
#     start = time.time()
#     temp = start
    
#     total_loss = 0
    
#     for epoch in range(epochs):
       
#         for i, batch in enumerate(train_iter):
#             src = batch.English.transpose(0,1)
#             trg = batch.French.transpose(0,1)
#             # the French sentence we input has all words except
#             # the last, as it is using each word to predict the next
            
#             trg_input = trg[:, :-1]
            
#             # the words we are trying to predict
            
#             targets = trg[:, 1:].contiguous().view(-1)
            
#             # create function to make masks using mask code above
            
#             src_mask, trg_mask = create_masks(src, trg_input)
            
#             preds = model(src, trg_input, src_mask, trg_mask)
            
#             optim.zero_grad()
            
#             loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
#             results, ignore_index=target_pad)
#             loss.backward()
#             optim.step()
            
#             total_loss += loss.data[0]
#             if (i + 1) % print_every == 0:
#                 loss_avg = total_loss / print_every
#                 print("time = %dm, epoch %d, iter = %d, loss = %.3f,%ds per %d iters" 
#                       % ((time.time() - start) // 60,
#                 epoch + 1, i + 1, loss_avg, time.time() - temp,
#                 print_every))
#                 total_loss = 0
#                 temp = time.time()
                
#%%




print("##### Example shown here #####")
EMB = Embedder(vocab_size, d_model)

e1 = df_small['en'][0]
e2 = df_small['en'][2]
f1 = df_small['fr'][0]

e = tokenizer(e1, return_tensors='pt')
e1 = tokenizer(e1, return_tensors='pt')
e2 = tokenizer(e2, return_tensors='pt')

# print(e1)

e1 = EMB.forward(e1['input_ids'])
p_e1 = pe(e1)

#%%
# f1 = tokenizer(f1, return_tensors='pt')

# print("embedding")
# # e1 = EMB.forward(e1['input_ids'])
# e2 = EMB.forward(e2['input_ids'])
# f1 = EMB.forward(f1['input_ids'])

# print("f1:", f1.size())
# out_e = encoder.forward(e['input_ids'])
# #%%

# print("positional encoding")
# # print(e1.size())

# e1 = pe.forward(e1)
# Q, K, V = e1, e1, e1

# out = MHA.forward(Q, K, V)

# out = FFN.forward(out)

# out_e1 = encoderLayer(out)
# print("encoderLayer out:\n",out_e1.size())



# out_d1 = decoderLayer1.forward(f1, out_e1, None, None)





#%%

"""
20 letters

how many combinations of 20 characters with length 20 are possible?

abc 
acb
cba 
cab
bac
bca


"""

