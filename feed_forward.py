# feed_forward.py

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: aynı boyutta çıktı [batch_size, seq_len, d_model]
        """
        return self.dropout(self.linear2(self.relu(self.linear1(x))))


# ✅ Test bloğu
if __name__ == "__main__":
    # Test parametreleri
    batch_size = 2
    seq_len = 10
    d_model = 64
    hidden_dim = 256  # genelde 4 * d_model alınır

    # Rastgele sahte veri
    x = torch.randn(batch_size, seq_len, d_model)

    # FFN modülü
    ffn = FeedForward(d_model, hidden_dim)

    # Çıktı al
    out = ffn(x)

    # Yazdır
    print("Input shape :", x.shape)   # [2, 10, 64]
    print("Output shape:", out.shape) # [2, 10, 64]
    print("Output dtype:", out.dtype)
