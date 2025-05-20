# tokenizer_word.py

class WordTokenizer:
    def __init__(self, text):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"

        tokens = set(text.strip().split())
        self.vocab = [self.pad_token, self.unk_token, self.sos_token, self.eos_token] + sorted(tokens)
        self.word2id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id2word = {idx: word for word, idx in self.word2id.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        words = text.strip().split()
        return (
            [self.word2id[self.sos_token]] +
            [self.word2id.get(word, self.word2id[self.unk_token]) for word in words] +
            [self.word2id[self.eos_token]]
        )

    def decode(self, ids):
        words = [self.id2word.get(idx, self.unk_token) for idx in ids]
        return ' '.join(words).replace(self.sos_token, '').replace(self.eos_token, '').strip()

# Test
if __name__ == "__main__":
    with open("data/ataturk.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = WordTokenizer(text)

    sample = "cumhuriyet fazilettir"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)

    print("Sample:", sample)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
    print("Vocab size:", tokenizer.vocab_size)
