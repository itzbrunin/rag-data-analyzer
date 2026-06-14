import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# Carrega variáveis do arquivo .env (como GROQ_API_KEY)
load_dotenv()


def create_rag_agent(vectorstore):
    # Obtém a chave da Groq do ambiente
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("A variável GROQ_API_KEY não foi encontrada no .env")

    # Inicializa o modelo LLM da Groq (Llama 3 / Mixtral)
    llm = ChatGroq(
        temperature=0,                    # Respostas mais estáveis
        groq_api_key=groq_api_key,        # Autenticação
        model_name="mixtral-8x7b-32768"   # Modelo rápido e compatível com LangChain
    )

    # Template básico para orientar o modelo
    template = """
    Use o contexto abaixo para responder à pergunta.
    Se a resposta não estiver no contexto, diga que não sabe.

    {context}

    Pergunta: {question}
    Resposta:
    """

    # Monta o objeto PromptTemplate para LangChain
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Cria o retriever baseado no ChromaDB
    retriever = vectorstore.as_retriever()

    # Cria a cadeia RAG (retrieval + geração)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                 # Método simples: junta chunks e envia ao LLM
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain