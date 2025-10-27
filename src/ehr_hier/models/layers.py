import torch.nn as nn

class SimpleTransformerLayer(nn.Module):
    #nn.TransformerEncoderLayer (batch_first=True). Mask ignored in this stub
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
    def forward(self, x, attn_keep_mask=None):
        return self.layer(x)