import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_rag_agent(vectorstore):
    # Inicializa o modelo ChatGroq com o modelo atualizado e suportado
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    # Define o template de prompt
    template = """Use as seguintes partes do contexto para responder à pergunta ao final. Se você não souber a resposta, apenas diga que não sabe, não tente inventar uma resposta.

    Contexto:
    {context}

    Pergunta: {question}
    Resposta útil:"""
    
    prompt = PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()

    # Função auxiliar para formatar os documentos recuperados em um único texto
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Cria a cadeia RAG usando a sintaxe moderna do LangChain (LCEL)
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["query"])), 
            "question": lambda x: x["query"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain