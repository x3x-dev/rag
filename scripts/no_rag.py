"""
Test LLM knowledge of MingLib without RAG

This script tests how well a plain LLM (without RAG) can answer
questions about our fictional minglib financial library.
"""

import os
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("Please set your OPENAI_API_KEY environment variable")
    exit(1)

# Initialize LLM without any RAG
llm = OpenAI(model="gpt-4o-mini", temperature=0)


def get_response(question: str) -> str:
  """Retrieve and answer a question using the LLM."""
  response = llm.complete(question)
  return response.text


if __name__ == "__main__":
      
  print("=" * 60)
  print("TESTING LLM KNOWLEDGE OF MINGLIB (NO RAG)")
  print("=" * 60)
  print("Ask questions about the fictional minglib library.")
  print("Type 'quit' to exit.\n")

  while True:
      question = input("Your question: ").strip()
      
      if question.lower() in ['quit', 'exit', 'q']:
          print("Goodbye!")
          break
      
      if question:
          print("-" * 50)
          
          # Query the LLM directly without any context
          response = llm.complete(question)
          
          print(f"Answer: {response.text}")
          print("-" * 50)
      else:
          print("Please enter a question.")