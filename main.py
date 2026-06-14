import sys
from src.ingestion import load_document, create_vectorstore
from src.agent import create_rag_agent

def main():
    # Caminho para o documento de exemplo
    file_path = "data/document.pdf"
    
    print("Iniciando o processamento do documento...")
    
    # Carrega o documento e cria o banco de vetores
    try:
        docs = load_document(file_path)
        vectorstore = create_vectorstore(docs)
        
        # Cria o agente RAG
        agent = create_rag_agent(vectorstore)
        
        print("Sistema pronto! Digite 'sair' para encerrar.")
        
        # Loop principal para perguntas do usuário
        while True:
            query = input("\nPergunta: ")
            if query.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o programa.")
                break
            
            # Executa a consulta no agente
            response = agent.invoke({"input": query})
            print(f"Resposta: {response['output']}")
            
    except Exception as e:
        print(f"Erro ao executar o sistema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()