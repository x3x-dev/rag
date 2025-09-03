"""
MingLib RAG — clean, hybrid, and fast

- Loads Markdown docs recursively once, with correct metadata.
- Persistent VectorStoreIndex with semantic/sentence chunking.
- Retrieval modes: "vector", "keyword" (BM25), and true "hybrid" (union).
- Optional LLM-based reranking.
- Deterministic REFERENCES footer from used nodes.

Prereqs:
  pip install llama-index llama-index-llms-openai llama-index-embeddings-huggingface
  pip install llama-index-retrievers-bm25 transformers sentencepiece python-dotenv

Env:
  export OPENAI_API_KEY=...

Run:
  python enhanced_llamaindex.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

from dotenv import load_dotenv

# LlamaIndex core
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.loading import load_indices_from_storage

# Models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# ---------- Config ----------
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
INDEX_ID = "minglib-index"

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("minglib_rag")

# ---------- Prompt ----------
SYSTEM_PROMPT = """You are an expert assistant for MingLib, a Python library for quantitative finance and investment banking.

Provide accurate, helpful, and detailed responses using ONLY the provided documentation context.

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
Answer:
"""
TEXT_QA_TEMPLATE = PromptTemplate(SYSTEM_PROMPT)

# ---------- Hybrid Retriever (Union) ----------
class UnionHybridRetriever(BaseRetriever):
    """
    Union hybrid retriever:
      - Retrieves from vector and BM25.
      - Merges by node id, keeps highest score.
      - Returns top_k by score.
    """
    def __init__(self, vector_retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever, top_k: int = 10):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k

    def _merge_nodes(self, lists: Iterable[List[NodeWithScore]]) -> List[NodeWithScore]:
        pool: Dict[str, NodeWithScore] = {}
        for lst in lists:
            for nws in lst:
                if hasattr(nws, "node") and hasattr(nws.node, "node_id"):
                    nid = nws.node.node_id
                elif hasattr(nws, "id_"):
                    nid = nws.id_
                else:
                    nid = repr(getattr(nws, "node", nws))[:64]
                prev = pool.get(nid)
                if prev is None or (nws.score or 0.0) > (prev.score or 0.0):
                    pool[nid] = nws
        merged = list(pool.values())
        merged.sort(key=lambda x: (x.score or 0.0), reverse=True)
        return merged[: self.top_k]

    # Required BaseRetriever methods
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vec_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        return self._merge_nodes([vec_nodes, bm25_nodes])

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)

# ---------- Main RAG Class ----------
class MingLibRAG:
    def __init__(
        self,
        docs_path: str = "./docs",
        persist_dir: str = "./storage",
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        similarity_cutoff: float = 0.6,
        enable_rerank: bool = False,
        top_k: int = 10,
    ):
        self.docs_path = Path(docs_path)
        self.persist_dir = Path(persist_dir)
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_cutoff = similarity_cutoff
        self.enable_rerank = enable_rerank
        self.top_k = top_k

        self.llm = None
        self.embed_model = None
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None

        self.debug_handler = LlamaDebugHandler(print_trace_on_end=True)
        self.callback_manager = CallbackManager([self.debug_handler])

        self.doc_metadata: Dict[str, Dict[str, Any]] = {
            "risk_management.md": {"module": "minglib.risk", "category": "risk_analysis"},
            "portfolio_optimization.md": {"module": "minglib.portfolio", "category": "optimization"},
            "market_data.md": {"module": "minglib.market_data", "category": "data_processing"},
            "options_pricing.md": {"module": "minglib.options", "category": "derivatives"},
            "fixed_income.md": {"module": "minglib.fixed_income", "category": "bonds"},
            "credit_risk.md": {"module": "minglib.credit", "category": "credit_analysis"},
            "performance_analytics.md": {"module": "minglib.performance", "category": "analytics"},
            "data_validation.md": {"module": "minglib.validation", "category": "data_quality"},
            "backtesting.md": {"module": "minglib.backtesting", "category": "strategy_testing"},
            "reporting.md": {"module": "minglib.reporting", "category": "report_generation"},
        }

    # ----- Models -----
    def setup_models(self) -> None:
        self.llm = OpenAI(model=self.model_name, temperature=0.1, max_tokens=2048)
        self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.callback_manager = self.callback_manager
        logger.info(f"Models ready: LLM={self.model_name}, Embeddings={self.embedding_model}")

    # ----- Documents -----
    def load_documents(self) -> List[Document]:
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_path.resolve()}")
        logger.info(f"Loading documents from {self.docs_path} (recursive)")
        reader = SimpleDirectoryReader(input_dir=str(self.docs_path), recursive=True, required_exts=[".md"])
        raw_docs = reader.load_data()
        docs: List[Document] = []
        for d in raw_docs:
            filename = Path(d.metadata.get("file_path", "")).name
            md = dict(d.metadata) if d.metadata else {}
            md["filename"] = filename or "UNKNOWN.md"
            md.setdefault("doc_type", "documentation")
            if filename in self.doc_metadata:
                md.update(self.doc_metadata[filename])
            preface_lines = [f"Document Type: {md.get('doc_type', 'documentation')}"]
            if "module" in md:
                preface_lines.append(f"Module: {md['module']}")
            if "category" in md:
                preface_lines.append(f"Category: {md['category']}")
            preface = "\n".join(preface_lines)
            docs.append(Document(text=f"{preface}\n\n{d.text}", metadata=md))
        logger.info(f"Total documents loaded: {len(docs)}")
        return docs

    # ----- Indexing -----
    def _has_persisted_index(self) -> bool:
        """Check if there's a valid persisted index (only if JSON files exist)"""
        return self.persist_dir.exists() and any(self.persist_dir.glob("*.json"))

    def _load_existing_index(self) -> Optional[VectorStoreIndex]:
        """Try to load existing index with fallback handling"""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            idx = load_index_from_storage(storage_context, index_id=INDEX_ID)
            logger.info("Loaded index by ID '%s'", INDEX_ID)
            return idx
        except Exception as e:
            logger.warning("Load by ID failed (%s). Trying fallback...", e)
            try:
                all_idx = load_indices_from_storage(storage_context)
                if all_idx:
                    idx = next(iter(all_idx.values()))
                    idx.set_index_id(INDEX_ID)
                    idx.storage_context.persist(persist_dir=str(self.persist_dir))
                    logger.info("Loaded existing index and re-tagged to '%s'", INDEX_ID)
                    return idx
            except Exception as e2:
                logger.warning("Fallback load failed (%s).", e2)
        return None

    def create_index(self, documents: List[Document], force_rebuild: bool = False) -> None:
        if self._has_persisted_index() and not force_rebuild:
            logger.info("Loading existing index from storage...")
            idx = self._load_existing_index()
            if idx is not None:
                self.index = idx
                return

        logger.info("Creating new index...")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        try:
            nodes = splitter.get_nodes_from_documents(documents)
            logger.info("Created %d nodes (semantic splitter)", len(nodes))
        except Exception as e:
            logger.warning("Semantic splitter failed (%s). Using sentence splitter.", e)
            sentence_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator="\n\n")
            nodes = sentence_splitter.get_nodes_from_documents(documents)
            logger.info("Created %d nodes (sentence splitter)", len(nodes))

        self.index = VectorStoreIndex(nodes, show_progress=True)
        self.index.set_index_id(INDEX_ID)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(self.persist_dir))
        logger.info("Index created and persisted")

    # ----- Query Engine -----
    def _build_retriever(self, retrieval_mode: str) -> BaseRetriever:
        mode = retrieval_mode.lower().strip()
        if mode == "vector":
            return VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        if mode == "keyword":
            return BM25Retriever.from_defaults(docstore=self.index.docstore, similarity_top_k=self.top_k)
        vec = VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k)
        bm25 = BM25Retriever.from_defaults(docstore=self.index.docstore, similarity_top_k=self.top_k)
        return UnionHybridRetriever(vec, bm25, top_k=self.top_k)

    def setup_query_engine(self, retrieval_mode: str = "hybrid") -> None:
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        retriever = self._build_retriever(retrieval_mode)
        postprocessors = [SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)]
        if self.enable_rerank:
            try:
                from llama_index.core.postprocessor import LLMRerank
                postprocessors.append(LLMRerank(choice_batch_size=10, top_n=min(5, self.top_k), llm=self.llm))
                logger.info("LLM reranker enabled")
            except Exception as e:
                logger.warning("LLM reranker unavailable: %s", e)

        synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=TEXT_QA_TEMPLATE,
            llm=self.llm,
        )

        # Build RetrieverQueryEngine manually to avoid duplicate 'retriever' arg
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=postprocessors,
            callback_manager=self.callback_manager,
        )
        logger.info("Query engine ready (mode=%s)", retrieval_mode)

    # ----- Query API -----
    def _build_references_footer(self, source_nodes: Optional[List[NodeWithScore]]) -> str:
        if not source_nodes:
            return "REFERENCE: UNKNOWN.MD"
        files = []
        for n in source_nodes:
            md = getattr(n, "metadata", None)
            if not md and hasattr(n, "node") and hasattr(n.node, "metadata"):
                md = n.node.metadata
            filename = None if not md else md.get("filename")
            if filename:
                files.append(filename)
        files = sorted({f for f in files if f})
        return f"REFERENCE: {', '.join(f.upper() for f in files)}" if files else "REFERENCE: UNKNOWN.MD"

    def query(self, question: str, retrieval_mode: str = "hybrid") -> Dict[str, Any]:
        if self.query_engine is None:
            self.setup_query_engine(retrieval_mode)
        enhanced_question = (
            f"Question about MingLib financial library: {question}\n"
            "Provide code examples and module paths."
        )
        response = self.query_engine.query(enhanced_question)

        source_nodes = getattr(response, "source_nodes", None)
        source_info: List[Dict[str, Any]] = []
        if source_nodes:
            for node in source_nodes:
                md = getattr(node, "metadata", {})
                if not md and hasattr(node, "node") and hasattr(node.node, "metadata"):
                    md = node.node.metadata
                source_info.append(
                    {
                        "filename": md.get("filename", "UNKNOWN.md"),
                        "doc_type": md.get("doc_type", "Unknown"),
                        "module": md.get("module", md.get("Module", "Unknown")),
                        "score": getattr(node, "score", "N/A"),
                    }
                )

        answer_text = str(response).rstrip()
        footer = self._build_references_footer(source_nodes)
        if not answer_text.upper().endswith(footer):
            answer_text = f"{answer_text}\n\n{footer}"

        return {
            "question": question,
            "answer": answer_text,
            "sources": source_info,
            "retrieval_mode": retrieval_mode,
            "response_metadata": getattr(response, "metadata", {}),
        }

    # ----- Initialization -----
    def initialize(self, force_rebuild: bool = False, retrieval_mode: str = "hybrid") -> None:
        logger.info("Initializing MingLib RAG...")
        self.setup_models()
        docs = self.load_documents()
        self.create_index(docs, force_rebuild=force_rebuild)
        self.setup_query_engine(retrieval_mode=retrieval_mode)
        logger.info("MingLib RAG initialized")

    # ----- Search (no synthesis) -----
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not created. Call initialize() first.")
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        results = []
        for n in nodes:
            md = getattr(n, "metadata", {})
            if not md and hasattr(n, "node"):
                md = getattr(n.node, "metadata", {})
            text = n.node.get_content() if hasattr(n, "node") else str(getattr(n, "text", ""))
            results.append(
                {
                    "content": (text[:500] + "...") if len(text) > 500 else text,
                    "metadata": md,
                    "score": getattr(n, "score", "N/A"),
                }
            )
        return results

# ---------- CLI Demo ----------
def main() -> None:
    rag = MingLibRAG(
        docs_path="../docs/api",   # adjust to your docs root
        persist_dir="./storage",
        model_name=DEFAULT_MODEL,
        embedding_model=DEFAULT_EMBED_MODEL,
        chunk_size=1024,
        chunk_overlap=200,
        similarity_cutoff=0.6,
        enable_rerank=False,   # True = better precision, slower/costlier
        top_k=10,
    )
    try:
        rag.initialize(force_rebuild=False, retrieval_mode="hybrid")
        print("\n" + "=" * 80)
        print("MINGLIB RAG — INTERACTIVE DEMO")
        print("=" * 80)
        while True:
            print("\nOptions:")
            print("1. Ask a custom question")
            print("2. Run an example query")
            print("3. Search documents (no synthesis)")
            print("4. Rebuild index")
            print("5. Exit")
            choice = input("\nEnter choice (1-5): ").strip()
            if choice == "1":
                question = input("\nEnter your question about MingLib: ").strip()
                if question:
                    print("\nProcessing...")
                    res = rag.query(question, retrieval_mode="hybrid")
                    print("\n--- Answer ---\n")
                    print(res["answer"])
                    if res["sources"]:
                        print("\n--- Sources ---")
                        for i, s in enumerate(res["sources"], 1):
                            print(f"{i}. {s['filename']} ({s['module']}) — score={s['score']}")
            elif choice == "2":
                q = "How to calculate Value at Risk (historical simulation) in MingLib?"
                print(f"\nExample: {q}\n")
                res = rag.query(q, retrieval_mode="hybrid")
                print("\n--- Answer (first 800 chars) ---\n")
                print(res["answer"][:800] + ("..." if len(res["answer"]) > 800 else ""))
                print("\n--- Sources ---")
                for i, s in enumerate(res["sources"], 1):
                    print(f"{i}. {s['filename']} ({s['module']}) — score={s['score']}")
            elif choice == "3":
                sq = input("\nEnter search terms: ").strip()
                if sq:
                    out = rag.search_documents(sq, top_k=5)
                    print(f"\nSearch results for '{sq}':")
                    for i, r in enumerate(out, 1):
                        md = r["metadata"]
                        print(f"\n{i}. File: {md.get('filename', 'UNKNOWN.md')}")
                        print(f"   Module: {md.get('module', 'Unknown')}")
                        print(f"   Score: {r['score']}")
                        print(f"   Snippet: {r['content']}")
            elif choice == "4":
                print("\nRebuilding index...")
                rag.initialize(force_rebuild=True, retrieval_mode="hybrid")
                print("Rebuilt.")
            elif choice == "5":
                print("\nGoodbye.")
                break
            else:
                print("Invalid choice. Try again.")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Error: {e}")

if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        sys.exit(1)
    main()
