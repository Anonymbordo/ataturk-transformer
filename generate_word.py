# generate_word.py

import torch
import torch.nn.functional as F
from tokenizer_word import WordTokenizer
from model_word import WordTransformerDecoder
import random

# 📂 Metni ve tokenizer'ı yükle
with open("data/ataturk.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = WordTokenizer(text)
vocab_size = tokenizer.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Modeli tanımla ve ağırlıkları yükle
model = WordTransformerDecoder(
    vocab_size=vocab_size,
    d_model=128,
    max_len=256,
    num_heads=4,
    ff_hidden=512,
    num_layers=4,
    dropout=0.1
).to(device)

# Eğitim tamamlandıysa bu dosya, erken durdurduysan "word_model_interrupt.pth"
model.load_state_dict(torch.load("word_model.pth", map_location=device))
model.eval()

# 🔮 Üretim fonksiyonu (top-k sampling)
def generate_text(prompt, max_len=50, top_k=5, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt)
        generated = input_ids[:-1]  # [SOS kelime1 kelime2 ...]

        for _ in range(max_len):
            x = torch.tensor([generated], device=device)
            out = model(x)
            logits = out[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k=top_k)
            next_token = random.choices(topk_indices[0].tolist(), weights=topk_probs[0].tolist(), k=1)[0]

            if tokenizer.id2word[next_token] == "<EOS>":
                break

            generated.append(next_token)

        return tokenizer.decode(generated)

# 🎯 Kullanıcıdan prompt al ve üret
print("\nÖrnek: Cumhuriyet nedir? Atatürk şöyle der:\n")
prompt = input("📝 Başlangıç metni: ").strip()

output = generate_text(prompt, max_len=50, top_k=5, temperature=1.0)

print("\n🧠 Üretilen metin:\n")
print(output)
