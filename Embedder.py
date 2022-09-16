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