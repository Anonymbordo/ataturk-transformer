# generate_sampling.py

import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer
from decoder import TransformerDecoder
import random

# 📂 Model ve tokenizer'ı yükle
with open("data/ataturk.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerDecoder(
    vocab_size=vocab_size,
    d_model=128,
    max_seq_len=256,
    num_heads=8,
    ff_hidden_dim=512,
    num_layers=4,
    dropout=0.1
).to(device)

model.load_state_dict(torch.load("ataturk_model.pth", map_location=device))
model.eval()

# 🔮 Top-k Sampling ile üretim (tekrar engelleyici ve yaratıcı)
def generate_sampling(prompt, max_len=150, temperature=1.0, top_k=5, block_ngram=3):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt)
        generated = input_ids[:]

        for _ in range(max_len):
            x = torch.tensor([generated], device=device)
            out = model(x)
            logits = out[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k=top_k)
            topk_probs = topk_probs.squeeze()
            topk_indices = topk_indices.squeeze()

            next_token = random.choices(topk_indices.tolist(), weights=topk_probs.tolist(), k=1)[0]

            # 🔁 Tekrar kontrolü
            if len(generated) >= block_ngram:
                last_ngram = generated[-(block_ngram - 1):]
                if all(t == next_token for t in last_ngram):
                    continue  # Tekrarı engelle

            generated.append(next_token)

        return tokenizer.decode(generated)

# 🎯 Prompt girişi ve üretim
print("\n📥 Örnek: Cumhuriyet nedir? Atatürk şöyle der:\n")
prompt = input("📝 Başlangıç metni: ").strip()

output = generate_sampling(prompt)

print("\n🧠 Üretilen metin:\n")
print(output)
