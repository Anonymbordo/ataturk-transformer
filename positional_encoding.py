# positional_encoding.py

import torch
import math

def get_positional_encoding(seq_len, d_model):
    """
    seq_len: Maksimum pozisyon (örneğin 100)
    d_model: Embedding boyutu (örneğin 64)
    return: [seq_len, d_model] boyutunda positional encoding tensor'ü
    """
    pos_enc = torch.zeros(seq_len, d_model)

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = 10000 ** ((2 * i) / d_model)
            pos_enc[pos, i] = math.sin(pos / div_term)
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = math.cos(pos / div_term)

    return pos_enc


# ✅ Test bloğu
if __name__ == "__main__":
    seq_len = 10
    d_model = 16
    pe = get_positional_encoding(seq_len, d_model)

    print("Positional Encoding shape:", pe.shape)  # [10, 16]
    print(pe)
