# MingLib Query Web Application

A web application that provides a chat a chat interface for querying the minglib rag systems...


**Frontend:** React + TypeScript + Tailwind CSS (`minglib_ai/client/`)
- Modern chat interface with markdown support
- Model switcher (Simple/Enhanced/No RAG)
- Clickable example questions
- Persistent chat history with backend sync

**Backend:** FastAPI + Python (`minglib_ai/server/`)
- RESTful API endpoints (`/api/chat`, `/api/chat/history`)
- In-memory chat storage
- Unified model switching
- Structured JSON responses

### Quick Start

```bash
# Backend
cd minglib_ai/server
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python main.py

# Frontend  
cd minglib_ai/client
npm install
npm run dev
```