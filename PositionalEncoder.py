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
