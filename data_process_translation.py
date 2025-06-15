import os
import re
import shutil
import nltk
from transformers import MarianMTModel, MarianTokenizer

# ===== DOWNLOAD REQUIRED NLTK RESOURCES =====
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Critical for tokenization

# ===== PATHS =====
input_dir = './assignment_corpus/en/mayoclinic'
filtered_dir = './assignment_corpus/en/filtered'
translated_dir = './assignment_corpus/en/translated'

# Create directories if missing
os.makedirs(filtered_dir, exist_ok=True)
os.makedirs(translated_dir, exist_ok=True)

# Λίστα κλειδιών για φιλτράρισμα (Αγγλικά)
keywords_en = [
    r"\bhematologic\b", r"\bblood cancer\b", r"\bhematological neoplasm\b", 
    r"\bleukemia\b", r"\blymphoma\b", r"\bmyeloma\b", r"\bacute myeloid\b", 
    r"\bchronic lymphocytic\b", r"\bCLL\b", r"\bacute lymphoblastic\b", 
    r"\bchronic myelogenous\b", r"\bHodgkin\b", r"\bnonHodgkin\b", 
    r"\bdiffuse large B-cell\b", r"\bmyelodysplastic\b", 
    r"\bmyeloproliferative\b", r"\bmonoclonal gammopathy\b", 
    r"\bWaldenström\b", r"\bCAR-NK\b", r"\bCAR-T\b", 
    r"\bstem cell transplant\b", r"\bminimal residual disease\b", 
    r"\bhematopoiesis\b", r"\bbone marrow\b"
]
pattern_en = re.compile('|'.join(keywords_en), re.IGNORECASE)

# Λεξικό για αντικατάσταση ιατρικών όρων (ICD-10)
medical_dict = {
    "leukemia": "λευχαιμία",
    "lymphoma": "λέμφωμα",
    "myeloma": "μυέλωμα",
    "bone marrow": "μυελός των οστών",
    "hematologic": "αιματολογικός",
    "blood cancer": "καρκίνος του αίματος",
    "acute myeloid": "οξεία μυελογενής λευχαιμία",
    "CLL": "Χρόνια Λεμφοκυτταρική Λευχαιμία",
    "CAR-T": "CAR-T κυτταρική θεραπεία",
    "stem cell transplant": "μεταμόσχευση βλαστοκυττάρων"
}

# ===== TRANSLATION SETUP =====
model_name = 'Helsinki-NLP/opus-mt-en-el'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# ===== IMPROVED TRANSLATION FUNCTION =====
def translate_text(text):
    # Verify NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Replace medical terms first
    for en, gr in medical_dict.items():
        text = re.sub(rf'\b{re.escape(en)}\b', gr, text, flags=re.IGNORECASE)
    
    # Tokenize with explicit language handling
    sentences = nltk.sent_tokenize(text, language='english')
    
    translated = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs)
        translated.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    return ' '.join(translated)

# ===== MAIN PROCESS =====
# 1. Filter files
for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    if pattern_en.search(content):
        shutil.copy(filepath, os.path.join(filtered_dir, filename))

# 2. Translate filtered files
for filename in os.listdir(filtered_dir):
    filepath = os.path.join(filtered_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        translated_content = translate_text(content)
        with open(os.path.join(translated_dir, filename), 'w', encoding='utf-8') as f:
            f.write(translated_content)
        print(f"Translated: {filename}")
    except Exception as e:
        print(f"Error translating {filename}: {str(e)}")

print("Processing complete!")