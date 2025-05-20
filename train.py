# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import CharTokenizer
from decoder import TransformerDecoder
import time

# ✅ Hyperparametreler
batch_size = 512
seq_len = 32
d_model = 128
num_heads = 8
ff_hidden_dim = 512
num_layers = 4
dropout = 0.1
learning_rate = 1e-3
epochs = 1
max_seq_len = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Cihaz:", device)

# ✅ Veri okuma ve tokenizer
print("📂 Veri okunuyor...")
with open("data/ataturk.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
encoded = tokenizer.encode(text)
vocab_size = tokenizer.vocab_size
print("🔤 Tokenizer hazır")
print("📏 Karakter sayısı:", len(encoded))
print("🔠 Vocab size:", vocab_size)

# ✅ Dataset
class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len])
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1])
        return x, y

dataset = CharDataset(encoded, seq_len)
print("📚 Dataset oluşturuldu. Örnek sayısı:", len(dataset))

if len(dataset) == 0:
    raise ValueError("❌ Dataset boş! seq_len çok büyük olabilir.")

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("🧃 DataLoader hazır")

# ✅ Model
print("🧠 Model oluşturuluyor...")
model = TransformerDecoder(
    vocab_size=vocab_size,
    d_model=d_model,
    max_seq_len=max_seq_len,
    num_heads=num_heads,
    ff_hidden_dim=ff_hidden_dim,
    num_layers=num_layers,
    dropout=dropout
).to(device)
print("✅ Model hazır")

# ✅ Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Eğitim döngüsü
print("🚀 Eğitime başlıyoruz...\n")

try:
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)

            output = output.view(-1, vocab_size)
            y = y.view(-1)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}] Batch {batch_idx} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"\n📈 Epoch {epoch+1}/{epochs} tamamlandı - Ortalama Loss: {avg_loss:.4f}\n")

    # Normal kayıt
    torch.save(model.state_dict(), "ataturk_model.pth")
    print("✅ Model başarıyla kaydedildi: ataturk_model.pth")

except KeyboardInterrupt:
    # CTRL+C ile kayıt
    torch.save(model.state_dict(), "ataturk_model_interrupt.pth")
    print("\n🛑 Eğitim elle durduruldu! 💾 Model kaydedildi: ataturk_model_interrupt.pth")
