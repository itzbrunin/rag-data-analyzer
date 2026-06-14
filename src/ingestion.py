import os
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_and_ingest_data(file_path, file_type):
    """
    Carrega um arquivo PDF ou CSV, divide em pedaços (chunks) e salva no ChromaDB.
    
    Args:
        file_path (str): Caminho para o arquivo.
        file_type (str): 'pdf' ou 'csv'.
    """
    # 1. Carrega o doumento com base no tipo
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError("Tipo de arquivo não suportado. Use 'pdf' or 'csv'.")
    
    documents = loader.load()
    
    #2. Divide o texto em pedaços menores (chunks) para não estourar o limite do LLM
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    #3. Configura o modelo de embeddings local da HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    #4. Cria o banco de dados de vetores e salva os chunks no diretório local
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Nota: Em versões muito recentes do ChromaDB, o persist() é feito automaticamente.
    # ao passar o persist_directory, mas mantemos aqui por compatibilidade.
    vectorstore.persist()
    
    return vectorstore
