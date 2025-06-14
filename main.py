# Retrieval-Augmented Generation (RAG) System for Hematologic Cancers: Robust Data Preparation Script

import os
import re
import nltk
import torch
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

# Download punkt for sentence tokenization (if needed)
nltk.download('punkt', quiet=True)

# -----------------------------
# Configuration
# -----------------------------
input_root = './assignment_corpus/en/mayoclinic'
output_root = './assignment_corpus/en/translated'
translate_all = True          # Set to False to filter only hematologic docs
max_tokens = 512
stride = 128                  # overlap for token-wise splitting
model_name = 'Helsinki-NLP/opus-mt-en-el'

os.makedirs(output_root, exist_ok=True)

# -----------------------------
# Optional Filtering setup
# -----------------------------
filter_terms = [
    "hematologic", "blood cancer", "hematological neoplasm", "leukemia", "lymphoma", "myeloma",
    "acute myeloid", "chronic lymphocytic", "CLL", "acute lymphoblastic", "chronic myelogenous",
    "Hodgkin", "nonHodgkin", "diffuse large B-cell", "myelodysplastic", "myeloproliferative",
    "monoclonal gammopathy", "Waldenström", "CAR-NK", "CAR-T", "stem cell transplant",
    "minimal residual disease", "hematopoiesis", "bone marrow"
]
pattern = re.compile(r"\b(?:" + r"|".join(map(re.escape, filter_terms)) + r")\b", re.IGNORECASE)

def contains_hematologic(text):
    return bool(pattern.search(text))

# -----------------------------
# Load model
# -----------------------------
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Token-wise Chunking & Translation Helpers
# -----------------------------
def split_text_tokenwise(text, max_tokens=max_tokens, stride=stride):
    """Split text into overlapping token chunks to cover entire document."""
    tokens = tokenizer(text, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'][0]
    total_len = input_ids.size(0)
    chunks = []
    step = max_tokens - stride
    for start in range(0, total_len, step):
        end = min(start + max_tokens, total_len)
        chunk_ids = input_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if end == total_len:
            break
    return chunks


def translate_en_to_el(text):
    """Translate full text by token-wise chunking."""
    chunks = split_text_tokenwise(text)
    translated = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model.generate(**inputs)
        translated_chunk = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        translated.append(translated_chunk)
    return '\n'.join(translated)

# -----------------------------
# Process files recursively
# -----------------------------
for subdir, dirs, files in os.walk(input_root):
    rel_path = os.path.relpath(subdir, input_root)
    out_dir = os.path.join(output_root, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for filename in tqdm(files, desc=f"Processing {rel_path}"):
        if not filename.lower().endswith('.txt'):
            continue

        input_path = os.path.join(subdir, filename)
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Skipping {input_path}, read error: {e}")
            continue

        if not translate_all and not contains_hematologic(content):
            continue

        try:
            translated_text = translate_en_to_el(content)
        except Exception as e:
            print(f"Error translating {filename}: {e}")
            continue

        base, ext = os.path.splitext(filename)
        out_en = os.path.join(out_dir, f"{base}_en{ext}")
        out_el = os.path.join(out_dir, f"{base}_el{ext}")

        with open(out_en, 'w', encoding='utf-8') as f_en:
            f_en.write(content)
        with open(out_el, 'w', encoding='utf-8') as f_el:
            f_el.write(translated_text)

print("\n✅ Finished translating documents with robust chunking and filtering.")
