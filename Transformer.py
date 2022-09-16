import Decoder
import Encoder
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.e = Encoder(source_vocab_size, d_model, n_layers, n_heads)
        self.d = Decoder(target_vocab_size, d_model, n_layers, n_heads)
        
        self.linear_f = nn.Linear(d_model, target_vocab_size)
        
    def forward(self, source, target, source_mask, target_mask):
        e_out = self.e(source, source_mask)
        d_out = self.d(target, e_out, source_mask, target_mask)
        
        out = self.linear_f(d_out)
        return out