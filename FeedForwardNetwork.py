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