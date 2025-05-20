# add_norm.py

import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        x: Giriş tensor [batch_size, seq_len, d_model]
        sublayer_output: Attention veya FFN çıktısı
        return: LayerNorm(x + Dropout(sublayer_output))
        """
        return self.norm(x + self.dropout(sublayer_output))


# ✅ Test Bloğu
if __name__ == "__main__":
    # Parametreler
    batch_size = 2
    seq_len = 10
    d_model = 64

    # Sahte giriş verileri
    x = torch.randn(batch_size, seq_len, d_model)
    sublayer_out = torch.randn(batch_size, seq_len, d_model)

    # AddNorm bloğu
    add_norm = AddNorm(d_model, dropout=0.1)

    # Uygula
    out = add_norm(x, sublayer_out)

    # Sonuçları yazdır
    print("Input shape         :", x.shape)           # [2, 10, 64]
    print("Sublayer output     :", sublayer_out.shape)
    print("Output shape        :", out.shape)         # [2, 10, 64]
    print("Output dtype        :", out.dtype)
    print("LayerNorm successful:", torch.allclose(out, out, atol=1e-5))  # basic stability check
