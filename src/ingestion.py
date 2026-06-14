import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_and_ingest_data(file_path, file_type='pdf'):
    # Carrega o documento com base no tipo especificado
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Divide o texto em pedaços menores para processamento
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Inicializa o modelo de embeddings do HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Cria e retorna o vectorstore Chroma
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore