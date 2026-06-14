RAG Data Analyzer
=================

Um sistema profissional de RAG (Retrieval-Augmented Generation) desenvolvido em Python para análise de documentos extensos e extração de insights precisos utilizando Inteligência Artificial. O sistema permite conversar com PDFs de centenas de páginas, mantendo o contexto semântico e garantindo respostas baseadas estritamente nos dados fornecidos.

Recursos Principais
-------------------
* Arquitetura Moderna: Construído com LCEL (LangChain Expression Language), garantindo um pipeline de dados limpo, modular e escalável, substituindo as antigas cadeias legadas.
* Inferência de Alta Velocidade: Integração nativa com a API da Groq, utilizando o modelo Llama-3.3-70b-versatile para processamento de linguagem natural com latência ultrabaixa.
* Gestão de Ambiente Segura: Gerenciamento de credenciais e variáveis de ambiente isoladas utilizando python-dotenv.
* Processamento Robusto de PDFs: Ingestão, divisão de texto (chunking) e vetorização otimizadas para documentos complexos.

Pré-requisitos
--------------
* Python 3.8 ou superior
* Chave de API da Groq (GROQ_API_KEY)

Instalação e Configuração
-------------------------
1. Clone o repositório e acesse a pasta do projeto.

2. Crie e ative um ambiente virtual:
   python -m venv venv
   .\venv\Scripts\activate  (Windows)
   source venv/bin/activate (Linux/Mac)

3. Instale as dependências do projeto:
   pip install langchain langchain-core langchain-community langchain-groq chromadb sentence-transformers pypdf python-dotenv

4. Na raiz do projeto, crie um arquivo chamado .env e adicione sua chave de API:
   GROQ_API_KEY=sua_chave_da_groq_aqui

Como Usar
---------
1. Insira o arquivo PDF que deseja analisar dentro da pasta data/ e certifique-se de que o nome corresponde ao configurado no código (ex: documento.pdf).

2. Execute o arquivo principal da aplicação:
   python main.py

3. O sistema realizará a leitura e vetorização do documento. Em seguida, um terminal interativo será aberto para você fazer perguntas em linguagem natural sobre o conteúdo do arquivo. Para encerrar, digite "sair".

Estrutura do Projeto
--------------------
* data/ : Diretório destinado aos documentos de entrada (ex: PDFs).
* src/ : Contém os módulos principais do sistema.
  * agent.py : Configuração do modelo de linguagem (Groq) e construção da cadeia RAG utilizando a sintaxe moderna LCEL.
  * ingestion.py : Lógica de carregamento de documentos, divisão em chunks e armazenamento no banco de dados vetorial.
* main.py : Ponto de entrada da aplicação, responsável por orquestrar a ingestão e o loop de interação com o usuário.
* .env : Arquivo de configuração de variáveis de ambiente (deve ser ignorado no controle de versão).

Tecnologias Utilizadas
----------------------
* Python
* LangChain (Core, Community e Groq)
* Groq API (Llama 3.3 70B)
* ChromaDB (Banco de dados vetorial)
* HuggingFace (Sentence Transformers para embeddings)
* PyPDF
