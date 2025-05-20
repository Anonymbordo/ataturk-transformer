# model_word.py

import torch
import torch.nn as nn
import math

def get_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)  # [1, max_len, d_model]
    return pe

class WordTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=256, num_heads=4, ff_hidden=512, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = get_positional_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model, num_heads, ff_hidden, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: [batch, seq_len]
        """
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)

        emb = self.embedding(x) + pos_enc
        emb = self.dropout(emb)

        # Geleceği görmemesi için mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        output = self.transformer_decoder(emb, emb, tgt_mask=tgt_mask)
        return self.output_proj(output)  # [batch, seq_len, vocab_size]

# Test
if __name__ == "__main__":
    model = WordTransformerDecoder(vocab_size=1000)
    x = torch.randint(0, 1000, (2, 16))  # [batch, seq_len]
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
