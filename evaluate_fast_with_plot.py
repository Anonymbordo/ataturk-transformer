
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt
from tokenizer import CharTokenizer
from tokenizer_word import WordTokenizer
from dataset_word import WordDataset
from model_word import WordTransformerDecoder
from decoder import TransformerDecoder

def evaluate(model, tokenizer, encoded_data, seq_len, vocab_size, device, batch_size=32, max_batches=100):
    subset = encoded_data[:10000]
    dataset = WordDataset(subset, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= max_batches:
                break

            x, y = x.to(device), y.to(device)
            out = model(x)
            out = out.view(-1, vocab_size)
            y = y.view(-1)

            loss = criterion(out, y)
            total_loss += loss.item()

            predictions = out.argmax(dim=1)
            correct += (predictions == y).sum().item()
            total_tokens += y.numel()

            if batch_idx % 10 == 0:
                print(f"🔁 Batch {batch_idx+1}/{max_batches} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / max_batches
    perplexity = math.exp(avg_loss)
    accuracy = correct / total_tokens

    return avg_loss, perplexity, accuracy

if __name__ == "__main__":
    seq_len = 16
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/ataturk.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print("\n🔵 Kelime Temelli Model Değerlendiriliyor...")
    word_tokenizer = WordTokenizer(text)
    encoded_word = word_tokenizer.encode(text)
    vocab_size_word = word_tokenizer.vocab_size

    word_model = WordTransformerDecoder(
        vocab_size=vocab_size_word,
        d_model=128,
        max_len=256,
        num_heads=4,
        ff_hidden=512,
        num_layers=4,
        dropout=0.1
    ).to(device)

    word_model.load_state_dict(torch.load("word_model.pth", map_location=device))
    w_loss, w_ppl, w_acc = evaluate(word_model, word_tokenizer, encoded_word, seq_len, vocab_size_word, device)

    print("\n🟢 Karakter Temelli Model Değerlendiriliyor...")
    char_tokenizer = CharTokenizer(text)
    encoded_char = char_tokenizer.encode(text)
    vocab_size_char = char_tokenizer.vocab_size

    char_model = TransformerDecoder(
        vocab_size=vocab_size_char,
        d_model=128,
        max_seq_len=256,
        num_heads=8,
        ff_hidden_dim=512,
        num_layers=4,
        dropout=0.1
    ).to(device)

    char_model.load_state_dict(torch.load("ataturk_model.pth", map_location=device))
    c_loss, c_ppl, c_acc = evaluate(char_model, char_tokenizer, encoded_char, seq_len, vocab_size_char, device)

    print("\n📊 Hızlı Performans Karşılaştırması (İlk 100 Batch)")
    print("--------------------------------------------------")
    print(f"🔵 Kelime Modeli -> Loss: {w_loss:.4f}, PPL: {w_ppl:.2f}, Accuracy: {w_acc:.2%}")
    print(f"🟢 Karakter Modeli -> Loss: {c_loss:.4f}, PPL: {c_ppl:.2f}, Accuracy: {c_acc:.2%}")

    models = ["Kelime Temelli", "Karakter Temelli"]
    loss = [w_loss, c_loss]
    perplexity = [w_ppl, c_ppl]
    accuracy = [w_acc, c_acc]

    plt.figure(figsize=(6, 4))
    plt.bar(models, loss, color='orange')
    plt.title("Loss Karşılaştırması")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(models, perplexity, color='red')
    plt.title("Perplexity Karşılaştırması")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracy, color='green')
    plt.title("Accuracy Karşılaştırması")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
