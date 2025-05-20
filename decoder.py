# decoder.py

import torch
import torch.nn as nn
from input_embedding import InputEmbedding
from decoder_block import DecoderBlock

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, num_heads, ff_hidden_dim, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        # 1. Giriş embedding (token + positional)
        self.embedding = InputEmbedding(vocab_size, d_model, max_seq_len)

        # 2. Birden fazla DecoderBlock
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 3. Çıkış katmanı (d_model → vocab_size)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len] — token ID'leri
        return: [batch_size, seq_len, vocab_size]
        """
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        return self.output_layer(x)


# ✅ Test Bloğu
if __name__ == "__main__":
    # Parametreler
    batch_size = 2
    seq_len = 6
    vocab_size = 80
    d_model = 64
    max_seq_len = 100
    num_heads = 8
    ff_hidden_dim = 256
    num_layers = 4
    dropout = 0.1

    # Sahte giriş verisi (token ID'leri)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Transformer decoder oluştur
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )

    # Modelden çıktıyı al
    out = decoder(x)

    # Kontroller
    print("Input shape :", x.shape)    # [2, 6]
    print("Output shape:", out.shape)  # [2, 6, 80]
    print("Başarılı mı?:", out.shape == (batch_size, seq_len, vocab_size))
