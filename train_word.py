# train_word.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizer_word import WordTokenizer
from dataset_word import WordDataset
from model_word import WordTransformerDecoder

# 🔧 Ayarlar
batch_size = 32
seq_len = 16
d_model = 128
num_heads = 4
ff_hidden = 512
num_layers = 4
dropout = 0.1
epochs = 3
lr = 1e-3
max_len = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Cihaz:", device)

# 📂 Veriyi yükle ve tokenize et
with open("data/ataturk.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = WordTokenizer(text)
encoded = tokenizer.encode(text)
vocab_size = tokenizer.vocab_size

print(f"🔠 Vocab size: {vocab_size}")
print(f"🔢 Toplam token sayısı: {len(encoded)}")

# 📚 Dataset ve DataLoader
dataset = WordDataset(encoded, seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 🧠 Model
model = WordTransformerDecoder(
    vocab_size=vocab_size,
    d_model=d_model,
    max_len=max_len,
    num_heads=num_heads,
    ff_hidden=ff_hidden,
    num_layers=num_layers,
    dropout=dropout
).to(device)

# 🎯 Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 🚀 Eğitim döngüsü (CTRL+C ile kayıt destekli)
print("🚀 Eğitime başlıyoruz...\n")

try:
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)  # [batch, seq_len, vocab_size]
            out = out.view(-1, vocab_size)
            y = y.view(-1)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"\n📉 Epoch {epoch+1}/{epochs} tamamlandı - Avg Loss: {avg_loss:.4f}\n")

    torch.save(model.state_dict(), "word_model.pth")
    print("✅ Model başarıyla kaydedildi: word_model.pth")

except KeyboardInterrupt:
    torch.save(model.state_dict(), "word_model_interrupt.pth")
    print("\n🛑 Eğitim erken durduruldu! 💾 Model kaydedildi: word_model_interrupt.pth")
