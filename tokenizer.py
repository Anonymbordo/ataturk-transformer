# tokenizer.py

class CharTokenizer:
    def __init__(self, text):
        # Vocab: benzersiz karakterlerin sıralı listesi
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)

        # Karakter → ID ve ID → karakter eşlemeleri
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}  # string to index
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}  # index to string

    def encode(self, text):
        """Metni karakter ID listesine çevir"""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        """ID listesini metne çevir"""
        return ''.join([self.itos[i] for i in ids])


# Aşağıdaki blok sadece test içindir.
# Bu kısmı istersen `train.py` gibi ayrı bir dosyaya taşıyabilirsin.
if __name__ == "__main__":
    # Atatürk metnini oku
    with open("data/ataturk.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenizer'ı oluştur
    tokenizer = CharTokenizer(text)

    # Test örneği
    sample_text = "ey türk gençliği"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print("Sample:", sample_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)
