# input_embedding.py

import torch
import torch.nn as nn
from positional_encoding import get_positional_encoding


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Positional encoding hesaplanıp buffer olarak saklanır
        pos_enc = get_positional_encoding(max_seq_len, d_model)
        self.register_buffer("positional_encoding", pos_enc)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: [batch_size, seq_len] şeklinde token ID tensor'u
        return: [batch_size, seq_len, d_model] tensor
        """
        seq_len = x.size(1)
        device = x.device

        # Embedding + d_model'e göre ölçekleme
        emb = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)).to(device)

        # Positional encoding uygun boyutta kesilip batch'e uygulanır
        pos_enc = self.positional_encoding[:seq_len, :].unsqueeze(0).to(device)

        return self.dropout(emb + pos_enc)


# ✅ Test bloğu (direkt çalıştırmak istersen)
if __name__ == "__main__":
    vocab_size = 80        # tokenizer.vocab_size kadar
    d_model = 64
    max_seq_len = 100
    batch_size = 2
    seq_len = 10

    # Rastgele sahte token ID'leri
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Model
    embed = InputEmbedding(vocab_size, d_model, max_seq_len)
    output = embed(x)

    print("Input shape:", x.shape)       # [2, 10]
    print("Output shape:", output.shape) # [2, 10, 64]
