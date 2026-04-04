# RAG Data Analyzer 🤖📊

Este projeto consiste em um **Agente de IA** especializado em análise de dados e documentos, utilizando a arquitetura **RAG (Retrieval-Augmented Generation)**. 

O objetivo é permitir que usuários interajam com grandes volumes de dados de forma semântica, obtendo insights precisos e fundamentados em fontes específicas.

## 🚀 Tech Stack
- **Linguagem:** Python 3.10+
- **Orquestração:** LangChain / LangGraph
- **LLM:** Llama 3 (via Groq API) - *Foco em baixa latência*
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)

## 🛠️ Arquitetura do Sistema
1. **Ingestão:** Processamento de arquivos (PDF/CSV) e quebra em pedaços (Chunks).
2. **Vetorização:** Geração de embeddings locais para eficiência de custo e memória.
3. **Retrieval:** Busca semântica no banco de vetores ChromaDB.
4. **Geração:** Processamento da consulta via Llama 3 para resposta contextualizada.

## 🔧 Como rodar
1. Clone o repositório.
2. Crie um ambiente virtual: `python -m venv venv`.
3. Instale as dependências: `pip install -r requirements.txt`.
4. Configure sua `GROQ_API_KEY` no arquivo `.env`.
5. Execute `python src/agent.py`.
