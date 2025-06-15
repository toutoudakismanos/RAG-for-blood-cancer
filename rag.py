import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.tokenize import word_tokenize

# ===== PATHS =====
input_dir = './assignment_corpus/en/mayoclinic'
filtered_dir = './assignment_corpus/en/filtered'
translated_dir = './assignment_corpus/en/translated'


# 1. Προετοιμασία δεδομένων για RAG
def load_documents(lang='en'):
    path = filtered_dir if lang == 'en' else translated_dir
    documents = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            documents.append(f.read())
    return documents

# 2. Δημιουργία Vector DB
def create_vector_db(documents, lang):
    model_name = 'sentence-transformers/all-MiniLM-L6-v2' if lang == 'en' \
        else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, model, documents

# 3. RAG Pipeline
class RAGSystem:
    def __init__(self, lang='en'):
        self.lang = lang
        self.documents = load_documents(lang)
        self.index, self.retriever, _ = create_vector_db(self.documents, lang)
        self.generator = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base",
            max_length=512
        )
    
    def retrieve(self, query, k=5):
        query_embedding = self.retriever.encode([query])
        _, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.documents[i] for i in indices[0]]
    
    def generate_answer(self, question):
        context = " ".join(self.retrieve(question))
        input_text = f"question: {question} context: {context}"
        return self.generator(input_text)[0]['generated_text']

# 4. Δοκιμή συστήματος
def test_rag():
    # Δοκιμή στα Αγγλικά
    en_rag = RAGSystem(lang='en')
    en_questions = [
        "What is the treatment for leukemia?",
        "How is lymphoma diagnosed?",
        "What are the symptoms of myeloma?",
        "Explain stem cell transplantation.",
        "What is CAR-T therapy?"
    ]
    
    # Δοκιμή στα Ελληνικά
    el_rag = RAGSystem(lang='el')
    el_questions = [
        "Ποια είναι η θεραπεία για τη λευχαιμία;",
        "Πώς διαγιγνώσκεται το λέμφωμα;",
        "Ποια είναι τα συμπτώματα του μυελώματος;",
        "Εξηγήστε τη μεταμόσχευση βλαστοκυττάρων.",
        "Τι είναι η θεραπεία CAR-T;"
    ]
    
    # Απαντήσεις και αξιολόγηση
    for q in en_questions:
        answer = en_rag.generate_answer(q)
        print(f"Q: {q}\nA: {answer}\n")
    
    for q in el_questions:
        answer = el_rag.generate_answer(q)
        print(f"Q: {q}\nA: {answer}\n")

# Εκτέλεση δοκιμών
test_rag()