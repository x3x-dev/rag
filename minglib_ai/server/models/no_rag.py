import os
from llama_index.llms.openai import OpenAI
from utils import parse_response
import time

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable")
    exit(1)

# Initialize LLM without any RAG
llm = OpenAI(model="gpt-4o-mini", temperature=0)

def get_response(question: str) -> dict:
    """Get a response from GPT-3.5 without RAG in expected API format"""
    start_time = time.time()
    try:
        response = llm.complete(question)
        raw_response = str(response.text)
        return parse_response(raw_response, start_time, "no_rag")
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": str(e),
                "code": "NO_RAG_ERROR",
                "details": {"model_type": "no_rag", "question": question}
            }
        }
