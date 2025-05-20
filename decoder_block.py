# decoder_block.py

import torch
import torch.nn as nn
from masked_multi_head_attention import MaskedMultiHeadAttention
from feed_forward import FeedForward
from add_norm import AddNorm

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()

        # 1. Masked Multi-Head Attention
        self.masked_attn = MaskedMultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)

        # 2. Feed Forward Network
        self.ffn = FeedForward(d_model, ff_hidden_dim, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len, d_model]
        """
        # Masked Attention + Add & Norm
        attn_out = self.masked_attn(x)
        x = self.add_norm1(x, attn_out)

        # FeedForward + Add & Norm
        ffn_out = self.ffn(x)
        x = self.add_norm2(x, ffn_out)

        return x


# ✅ Test Bloğu
if __name__ == "__main__":
    # Parametreler
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 8
    ff_hidden_dim = 256
    dropout = 0.1

    # Sahte giriş tensor'u (örneğin embedding sonrası)
    x = torch.randn(batch_size, seq_len, d_model)

    # Decoder bloğu oluştur
    decoder_block = DecoderBlock(d_model, num_heads, ff_hidden_dim, dropout)

    # İleri geçir
    out = decoder_block(x)

    # Çıktı kontrol
    print("Input shape :", x.shape)    # [2, 6, 64]
    print("Output shape:", out.shape)  # [2, 6, 64]
    print("Başarılı mı?:", out.shape == x.shape)
