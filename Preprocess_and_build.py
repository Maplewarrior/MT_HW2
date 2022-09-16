from util import setup_seed, preprocess_data
import torch
import numpy as np
import random
import Transformer


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



src_vocab_size = [word for sentence in ipt['input_ids'] for word in sentence]
src_vocab_size = len(np.unique(src_vocab_size))

trg_vocab_size = [word for sentence in ipt['target'] for word in sentence]
trg_vocab_size = len(np.unique(trg_vocab_size))

# flat_list = [item for sublist in l for item in sublist]

print((ipt['input_ids'].unsqueeze(1).shape))
print((ipt['target'].unsqueeze(1).shape))




def cloneLayers(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

pe = PositionalEncoder(d_model, max_seq_len=d_model)
MHA = MultiHeadAttention(n_heads, d_model, d_k)
FFN = FeedForwardNetwork(d_model, d_ff)

encoderLayer = EncoderLayer(n_heads, d_model, d_ff)
# encoderLayer2 = EncoderLayer1(n_heads, d_model, d_ff)

decoderLayer1 = DecoderLayer(n_heads, d_model, d_ff, d_k)

encoder = Encoder(vocab_size, d_model, n_layers, n_heads)
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


# def get_mask(trg, trg_mask):
#     return torch.masked_select(trg, trg_mask)

# target_seq = trg[0]
# size = len(target_seq)

# nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
# nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0)


#%%
# for epoch in range(epochs):
    
T = Transformer(vocab_size, vocab_size, d_model, n_layers, n_heads)

preds = T.forward(ipt['input_ids'][0], ipt['target'][0], None, None)




train_model(ipt, epochs)             
#%%
EMB = Embedder(vocab_size, d_model)
e1 = EMB(ipt['input_ids'][0].unsqueeze(1))
f1 = EMB(ipt['target'][0].unsqueeze(1))



print(e1.size())
print(f1.size())


print(ipt['target'][1]==0)

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

