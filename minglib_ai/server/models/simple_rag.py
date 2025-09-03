"""
Simple RAG Pipeline for MingLib Documentation
"""

import os
import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from utils import parse_response
import time
import os
from pathlib import Path

# Turn off verbose logging & warnings
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable")
    exit(1)

# Load docs from project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent  # Go up from models/ -> server/ -> minglib_ai/ -> rag/
docs_path = project_root / "docs" / "api"

# Load documents
documents = SimpleDirectoryReader(
    str(docs_path), recursive=True, required_exts=[".md"]
).load_data()


# Build index
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",  # Best quality embedding model
    trust_remote_code=True,
)
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Set up generative models
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)


# Custom system prompts for different model types
simple_system_prompt = """You are an expert assistant for MingLib, a Python library for quantitative finance and investment banking.

Your role is to provide accurate, helpful, and detailed responses about MingLib's functions, classes, and capabilities using ONLY the provided documentation context.

CRITICAL REQUIREMENTS:
1. ALWAYS include practical CODE EXAMPLES with proper imports and usage
2. ALWAYS end your response with "REFERENCE: [filename].md" in UPPERCASE
3. Include specific function names, class names, and module paths
4. Provide complete, runnable code snippets with realistic parameters
5. Show return values and expected outputs when available
6. Focus on practical usage patterns and best practices
7. If context is insufficient, state limitations clearly

RESPONSE FORMAT:
- Start with a clear explanation
- Include a complete CODE EXAMPLE with imports
- Show expected outputs/return values
- End with "REFERENCE: [filename].md"

Context information is below:
---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, answer the query about MingLib.
Query: {query_str}
Answer: """

# Create templates for both models
simple_template = PromptTemplate(simple_system_prompt)

# Create query engines for both models
engine = index.as_query_engine(
    response_mode="compact",
    verbose=False,
    text_qa_template=simple_template,
)

def get_simple_response(question: str) -> dict:
    """Return API-ready response for frontend"""
    start_time = time.time()
    try:
        model_type = "simple"
        raw_response = str(engine.query(question))
        return parse_response(raw_response, start_time,  "simple")
    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": str(e),
                "code": "PROCESSING_ERROR",
                "details": {"model_type": model_type, "question": question}
            }
        }

