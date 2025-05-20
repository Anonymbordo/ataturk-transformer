# encoder.py

import torch
import torch.nn as nn
from encoder_block import EncoderBlock
from input_embedding import InputEmbedding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Giriş embedding katmanı (token + positional)
        self.embedding = InputEmbedding(vocab_size, d_model, max_seq_len)

        # Birden fazla EncoderBlock
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [batch_size, seq_len] — token ID'leri
        return: [batch_size, seq_len, d_model]
        """
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        return x


# ✅ Test Bloğu
if __name__ == "__main__":
    # Parametreler
    batch_size = 2
    seq_len = 10
    vocab_size = 80
    d_model = 64
    max_seq_len = 100
    num_heads = 8
    ff_hidden_dim = 256
    num_layers = 4
    dropout = 0.1

    # Sahte giriş (token ID'leri)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Encoder modelini oluştur
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # Çalıştır
    out = encoder(x)

    # Sonuçları yazdır
    print("Input shape :", x.shape)    # [2, 10]
    print("Output shape:", out.shape)  # [2, 10, 64]
