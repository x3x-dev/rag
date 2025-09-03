# RAG:

## 1. What is RAG?
- LLMs are trained on static data and usually lacks access to real time information
- RAG Combines Information Retrival + Text Generation
- Helps reduce model hallucinations
- Useful for domain-specific adaptation of LLMs compared to fine-tuning

## 2. RAG Architecture
Two core components
- Retrieval: Vectore store, embeddings, search method
- Generator: LLM for generating final responses
- Retrieval flow: Query → Embedding → Vector DB → Top-k chunks → LLM with context
  - **Query:** The user’s question or prompt
  - **Embedding:** The query is converted into a dense vector using an embedding model
  - **Vector DB:** A database that stores embeddings of documents and allows similarity search
  - **Top-k chunks:** The most relevant document chunks are retrieved based on similarity
  - **LLM with context:** Retrieved chunks are passed along with the query into the LLM to generate a grounded answer

## 3. Implementation Tools: Libraries

### 3.1 Libraries
- **LangChain:** https://python.langchain.com/docs/
- **LlamaIndex:** https://docs.llamaindex.ai/en/stable/
- **Haystack:** https://docs.haystack.deepset.ai/docs

### 3.2 Vector Stores
- FAISS (local, simple)
- Chroma (lightweight, Pythonic)
- Pinecone, Weaviate, Milvus (cloud/enterprise)

## 5. MingLibAI Demo

## 5.1 What is MingLib?

MingLib is a fictional Python library for quantitative finance and investment banking, created specifically for testing RAG (Retrieval-Augmented Generation) systems. It provides comprehensive documentation covering:

- **Risk Management** - VaR calculation, stress testing, Monte Carlo simulations
- **Portfolio Optimization** - Markowitz, Black-Litterman, risk parity strategies  
- **Derivatives Pricing** - Options, fixed income, credit risk models
- **Performance Analytics** - Attribution analysis, benchmarking, risk metrics
- **Data Management** - Validation, backtesting, reporting tools

## 5.2 MingLibAI RAG System

MingLibAI implements a comprehensive RAG pipeline using `LlamaIndex` with multiple models and retrieval strategies for testing and demonstration purposes.
 
### Models Used
- **Embedding Model:** `BAAI/bge-large-en-v1.5` (Hugging Face)
- **Generative Model:** `gpt-3.5-turbo` (OpenAI)


### Web Application

I have also created a simple web app that provides a simple chat interface for querying the rag system.

More information: [`README.md`](minglib_ai/README.md)

## 6. Resources
- **Documentation:** [`docs/`](docs/) - Complete MingLib API reference (540+ pages)
- **Test Questions:** [`test_questions.md`](test_questions.md) - 10 evaluation questions with expected outputs
- **Web Interface:** [http://localhost:3000](http://localhost:3000) - Live chat interface
- **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs) - FastAPI Swagger UI 


