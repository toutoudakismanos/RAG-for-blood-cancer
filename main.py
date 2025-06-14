# Retrieval-Augmented Generation (RAG) System for Hematologic Cancers: Data Preparation Notebook

"""
This notebook guides you through the first phase (50%) of building a RAG system for hematologic cancers:
1. Loading and structuring the dataset
2. Filtering for hematologic cancer-related documents
3. Translating English content to Greek
4. Preserving directory structure and exporting processed data

Requirements:
- Python 3.8+
- pandas
- tqdm
- regex
- Hugging Face `transformers` and `datasets`
- SentencePiece

"""

# 1. Εγκατάσταση απαραίτητων βιβλιοθηκών
# !pip install pandas tqdm regex transformers datasets sentencepiece

# 2. Import libraries
import os
import re
import pandas as pd
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

# 3. Ορισμός όρων φιλτραρίσματος
filter_terms = [
    "hematologic", "blood cancer", "hematological neoplasm", "leukemia", "lymphoma", "myeloma",
    "acute myeloid", "chronic lymphocytic", "CLL", "acute lymphoblastic", "chronic myelogenous",
    "Hodgkin", "nonHodgkin", "diffuse large B-cell", "myelodysplastic", "myeloproliferative",
    "monoclonal gammopathy", "Waldenström", "CAR-NK", "CAR-T", "stem cell transplant",
    "minimal residual disease", "hematopoiesis", "bone marrow"
]

# Compile regex pattern for efficient matching
pattern = re.compile(r"\b(?:" + r"|".join(map(re.escape, filter_terms)) + r")\b", re.IGNORECASE)

# 4. Συνάρτηση για φιλτράρισμα κειμένου

def contains_hematologic(text):
    return bool(pattern.search(text))

# 5. Συνάρτηση μετάφρασης (EN → EL)
model_name = 'Helsinki-NLP/opus-mt-en-el'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_en_to_el(texts, batch_size=8):
    translations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        generated = model.generate(**encoded)
        translations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return translations

# 6. Επεξεργασία φακέλων και εξαγωγή
input_root = './assignment_corpus/en/mayoclinic'
output_root = './assignment_corpus/en/translated'

for subdir, dirs, files in os.walk(input_root):
    rel_path = os.path.relpath(subdir, input_root)
    out_dir = os.path.join(output_root, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    for file in files:
        if not file.lower().endswith('.txt'):
            continue
        input_path = os.path.join(subdir, file)
        with open(input_path, encoding='utf-8') as f:
            content = f.read()
        # Φιλτράρισμα
        if contains_hematologic(content):
            # Μετάφραση
            greek = translate_en_to_el([content])[0]
            # Διατήρηση δομής
            base, ext = os.path.splitext(file)
            out_en = os.path.join(out_dir, f"{base}_en{ext}")
            out_el = os.path.join(out_dir, f"{base}_el{ext}")
            with open(out_en, 'w', encoding='utf-8') as f_en, open(out_el, 'w', encoding='utf-8') as f_el:
                f_en.write(content)
                f_el.write(greek)

print("Finished processing hematologic cancer documents.")
