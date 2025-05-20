# encoder_block.py

import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from add_norm import AddNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()

        # 1. Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)

        # 2. Feed Forward Network
        self.ffn = FeedForward(d_model, ff_hidden_dim, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len, d_model]
        """
        # Attention + Add & Norm
        attn_out = self.self_attn(x)
        x = self.add_norm1(x, attn_out)

        # FeedForward + Add & Norm
        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)

        return x


# ✅ Test Bloğu
if __name__ == "__main__":
    # Parametreler
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    ff_hidden_dim = 256
    dropout = 0.1

    # Sahte input (embedding çıkışı gibi)
    x = torch.randn(batch_size, seq_len, d_model)

    # EncoderBlock oluştur
    encoder_block = EncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        dropout=dropout
    )

    # Output al
    out = encoder_block(x)

    # Çıktıları yazdır
    print("Input shape :", x.shape)    # [2, 10, 64]
    print("Output shape:", out.shape)  # [2, 10, 64]
    print("Başarılı mı? ", out.shape == x.shape)
