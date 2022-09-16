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
        vals = Attention(Q, K, V, d_k, mask, self.dropout)
        
        # Reshape to [bs x seq_len x d_model]
        vals = vals.transpose(1,2).contiguous().view(vals.size(0), -1, self.d_model)
       
        out = self.out(vals) # linear
        return out
    
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