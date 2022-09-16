import Embedder
import PositionalEncoder
import MultiHeadAttention
import FeedForwardNetwork
from util import cloneLayers, setup_seed
import torch.nn as nn

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

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, dropout=.1):
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