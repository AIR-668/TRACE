from modules import *

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_heads, dim_output):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            SAB(dim_input, dim_output, num_heads),
            SAB(dim_output, dim_output, num_heads))
        self.dec = nn.Sequential(
            PMA(dim_output, num_heads, 1),
            nn.Linear(dim_output, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))

