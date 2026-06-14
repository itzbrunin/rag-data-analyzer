import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env de qualquer outra coisa
load_dotenv()


from src.ingestion import load_and_ingest_data
from src.agent import get_rag_agent

def main():
    # Caminho para o arquivo PDF
    file_path = 'data/documento.pdf'
    
    print('Iniciando a ingestão de dados...')
    # Carrega e ingere os dados do arquivo
    vectorstore = load_and_ingest_data(file_path, file_type='pdf')
    
    print('Configurando o agente RAG...')
    # Inicializa o agente RAG com o vectorstore
    agent = get_rag_agent(vectorstore)
    
    print('Sistema pronto! Digite "sair" para encerrar.')
    
    # Loop principal para interação com o usuário
    while True:
        user_query = input('\nPergunta: ')
        
        if user_query.lower() in ['sair', 'exit', 'quit']:
            print('Encerrando o programa.')
            break
            
        # Invoca o agente para processar a pergunta
        response = agent.invoke({'query': user_query})
        print(f'Resposta: {response}')

if __name__ == '__main__':
    main()