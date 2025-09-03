"""
Simple RAG for MingLib Documentation

The simplest possible implementation
"""

import os
import logging
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from llama_index.core import PromptTemplate

load_dotenv()

# Turn off verbose logging & warnings
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable")
    exit(1)

# Load docs
documents = SimpleDirectoryReader(
    "../docs/", recursive=True, required_exts=[".md"]
).load_data()


# Load embedding model
print("Loading Hugging Face embedding model...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5",  # Best quality embedding model
    trust_remote_code=True,
)

# Build index
print("Building index...")
index = VectorStoreIndex.from_documents(documents, show_progress=True)

print("Setting up generative models...")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)


# Custom system prompt for better RAG responses
system_prompt = """You are an expert assistant for MingLib, a Python library for quantitative finance and investment banking.

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

text_qa_template = PromptTemplate(system_prompt)

# Query engine with optimized settings and custom prompt
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Retrieve fewer chunks to reduce API calls
    response_mode="compact",  # More efficient response synthesis
    verbose=False,  # Turn off query engine verbosity
    text_qa_template=text_qa_template,  # Custom system prompt
)


def get_response(question: str) -> str:
    """Get a response from the query engine."""
    return str(query_engine.query(question))


if __name__ == "__main__":
    # Interactive mode
    print("\n" + "=" * 70)
    print("Interactive mode - ask your own questions!")
    print("Commands:")
    print("  <question>     - Get response")
    print("  quit/exit      - Exit program")

    while True:
        user_input = input("\nYour input: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if user_input:
            print("-" * 50)
            response = get_response(user_input)
            print(f"Answer: {response}")
