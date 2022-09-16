import torch.nn as nn
import MultiHeadAttention
import FeedForwardNetwork
import Embedder
import PositionalEncoder
from util import cloneLayers, setup_seed


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dropout=.1):
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

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, dropout=.1):
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