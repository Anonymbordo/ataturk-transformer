# multi_head_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model, num_heads'e tam bölünmeli"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Lineer projeksiyonlar
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Çıkış projeksiyonu
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()

        # Q, K, V hesapla
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Head'lere böl ve transpoze et
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attention = torch.matmul(weights, V)

        # Head'leri birleştir
        concat = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Çıkış projeksiyonu
        return self.out_proj(concat)


# ✅ Test bloğu
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8

    # Rastgele giriş verisi (örneğin embedding çıkışı gibi)
    x = torch.randn(batch_size, seq_len, d_model)

    # Katmanı oluştur
    mha = MultiHeadAttention(d_model, num_heads)

    # Çıktı al
    out = mha(x)

    print("Input shape :", x.shape)    # [2, 10, 64]
    print("Output shape:", out.shape)  # [2, 10, 64]
