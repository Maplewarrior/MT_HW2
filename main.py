import Embedder
from Encoder import Encoder, EncoderLayer
from Decoder import Decoder, DecoderLayer
import PositionalEncoder
import MultiHeadAttention
import FeedForwardNetwork
import Transformer
from util import setup_seed, preprocess_data, train_model
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random

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

# Function for mapping data from strings to tokens
# s_key = source key, t_key = target_key

ipt = preprocess_data(df_small, 'en', 'fr', max_length=36)

src_vocab_size = len(np.unique([word for sentence in ipt['input_ids'] for word in sentence]))
trg_vocab_size = len(np.unique([word for sentence in ipt['target'] for word in sentence]))

# flat_list = [item for sublist in l for item in sublist]
pe = PositionalEncoder(d_model, max_seq_len=d_model)
MHA = MultiHeadAttention(n_heads, d_model, d_k)
FFN = FeedForwardNetwork(d_model, d_ff)
encoder = Encoder(vocab_size, d_model, n_layers, n_heads)

### define model and initialize params ###
model = Transformer(vocab_size, vocab_size, d_model, n_layers, n_heads)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

### define optimizer ###
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
epochs = 5
batch_size = 10

preds = model.forward(ipt['input_ids'][0], ipt['target'][0], None, None)

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

