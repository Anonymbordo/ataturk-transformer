import fitz  # PyMuPDF
import os

# PDF dosya yolu
pdf_path = "data/nutuk.pdf"
output_path = "data/ataturk.txt"

# PDF'i aç
doc = fitz.open(pdf_path)
full_text = ""

# Her sayfanın metnini al
for page in doc:
    full_text += page.get_text()

# Küçük harfe çevir, boşlukları sadeleştir
full_text = full_text.lower()
full_text = full_text.replace("\n", " ")
full_text = " ".join(full_text.split())  # fazladan boşlukları siler

# Kaydet
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("PDF başarıyla data/ataturk.txt dosyasına yazıldı.")
