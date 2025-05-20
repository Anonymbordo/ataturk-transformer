# dataset_word.py

import torch
from torch.utils.data import Dataset

class WordDataset(Dataset):
    def __init__(self, encoded_data, seq_len):
        self.data = encoded_data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# Test
if __name__ == "__main__":
    from tokenizer_word import WordTokenizer

    with open("data/ataturk.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = WordTokenizer(text)
    encoded = tokenizer.encode(text)
    print("Toplam token:", len(encoded))

    dataset = WordDataset(encoded, seq_len=16)
    print("Veri uzunluğu:", len(dataset))

    x, y = dataset[0]
    print("X:", x)
    print("Y:", y)
