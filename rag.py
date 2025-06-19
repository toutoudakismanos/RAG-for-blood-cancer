from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import re

# Configuration
EN_FILTERED = Path('./assignment_corpus/en/filtered')
EL_FILTERED = Path('./assignment_corpus/en/translated')
MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"

def is_greek(text):
    """Check if text contains Greek characters"""
    return bool(re.search(r'[α-ωΑ-Ω]', text))

def load_documents():
    """Load and combine documents from both directories"""
    documents = []
    
    # Load English documents
    for file_path in EN_FILTERED.glob('*.txt'):
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata["language"] = "english"
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Load translated Greek documents
    for file_path in EL_FILTERED.glob('*.txt'):
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata["language"] = "greek"
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(documents)} documents")
    return documents

def create_vector_store(documents):
    """Create FAISS vector store with optimized CPU embeddings"""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Create embeddings optimized for CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully")
    return vector_store

def initialize_llm():
    """Initialize Llama with CPU optimizations"""
    if not os.path.exists(MODEL_NAME):
        print(f"Error: Model file {MODEL_NAME} not found!")
        print("Please download from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        return None
    
    return LlamaCpp(
        model_path=MODEL_NAME,
        n_ctx=2048,
        n_threads=max(1, os.cpu_count() - 1),  # Reserve one core
        n_gpu_layers=0,
        temperature=0.1,
        max_tokens=200,  # Shorter responses
        verbose=False
    )

def create_qa_chain(vector_store, llm):
    """Create retrieval QA chain with medical prompt template"""
    prompt_template = """
    [INST] <<SYS>>
    You are an oncology medical assistant. Follow these rules:
    1. Answer ONLY based on the provided context
    2. Be concise and factual (1-2 sentences max)
    3. If context doesn't contain answer, say "I don't have information about this"
    4. Respond in the SAME LANGUAGE as the question
    <</SYS>>
    
    Context: {context}
    Question: {question}
    Answer: [/INST]
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 3, "score_threshold": 0.7}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

def main():
    # Step 1: Load documents
    documents = load_documents()
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    # Step 2: Create vector store
    vector_store = create_vector_store(documents)
    
    # Step 3: Initialize Llama
    llm = initialize_llm()
    if not llm:
        return
    
    # Step 4: Create QA chain
    qa_chain = create_qa_chain(vector_store, llm)
    
    # Interactive questioning
    while True:
        question = input("\nAsk a cancer-related question (type 'exit' to quit):\n")
        if question.lower() in ['exit', 'quit']:
            break
        
        try:
            # Detect language for debugging
            lang = "Greek" if is_greek(question) else "English"
            print(f"\nDetected question language: {lang}")
            
            # Get answer
            result = qa_chain.invoke({"query": question})
            response = result["result"]
            
            # Post-process to ensure conciseness
            if len(response.split('. ')) > 2:
                response = '. '.join(response.split('. ')[:2]) + '.'
            
            print(f"\nAnswer: {response}")
            
            # Show sources for verification
            print("\nSource documents:")
            for i, doc in enumerate(result['source_documents'][:2]):  # Top 2 only
                print(f"{i+1}. {os.path.basename(doc.metadata['source'])}")
                print(f"   Language: {doc.metadata.get('language', 'unknown')}")
                print(f"   Content: {doc.page_content[:150]}...")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Cancer Information RAG System")
    print("-----------------------------")
    print("Note: This system will respond in the same language as your question")
    print("and provide concise answers (1-2 sentences maximum).")
    main()