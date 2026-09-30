"""Hybrid Retriever — BM25 + Semantic + Cross-Encoder Reranking"""
import json
from pathlib import Path
from typing import List, Dict
from rich.console import Console
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

console = Console()


class HybridRetriever:
    """
    Combines:
    1. Semantic search (ChromaDB)
    2. Keyword search (BM25)
    3. RRF fusion
    4. Cross-Encoder reranking
    """

    def __init__(
        self,
        db_dir: str = "./data/chroma_db",
        collection_name: str = "legal_docs",
        dataset_file: str = "data/datasets/legal_qa_dataset.jsonl",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        console.print("[cyan]🔧 Initializing Hybrid Retriever...[/cyan]")

        # ── ChromaDB (semantic) ──
        self.client = chromadb.PersistentClient(path=db_dir)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

        # ── BM25 (keyword) — load dataset for indexing ──
        self.docs = []
        self.metadatas = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                p = json.loads(line)
                text = f"Q: {p['instruction']}\nA: {p['output']}"
                if p.get("input"):
                    text = f"Q: {p['instruction']}\nContext: {p['input']}\nA: {p['output']}"
                self.docs.append(text)
                self.metadatas.append({
                    "instruction": p["instruction"][:500],
                    "output": p["output"][:1000],
                    "input": (p.get("input") or "")[:300],
                    "source": p.get("source", "unknown"),
                })

        console.print(f"   📚 BM25 index: {len(self.docs)} docs")

        # Tokenize for BM25 (simple word-level)
        tokenized_corpus = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # ── Cross-Encoder Reranker ──
        console.print("   🤖 Loading reranker...")
        self.reranker = CrossEncoder(reranker_model, max_length=512)

        console.print("[green]✅ Hybrid Retriever ready[/green]")

    def _semantic_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """ChromaDB semantic search"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        docs = []
        for i, (doc, meta) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
        )):
            docs.append({
                "text": doc,
                "metadata": meta,
                "id": results["ids"][0][i],
                "semantic_rank": i,
            })
        return docs

    def _bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Top K indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        docs = []
        for rank, idx in enumerate(top_indices):
            docs.append({
                "text": self.docs[idx],
                "metadata": self.metadatas[idx],
                "id": f"doc_{idx}",
                "bm25_rank": rank,
                "bm25_score": float(scores[idx]),
            })
        return docs

    def _rrf_fusion(
        self,
        semantic_docs: List[Dict],
        bm25_docs: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Reciprocal Rank Fusion"""
        scores = {}

        for doc in semantic_docs:
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + doc["semantic_rank"] + 1)
            scores[doc_id + "_data"] = doc

        for doc in bm25_docs:
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + doc["bm25_rank"] + 1)
            if doc_id + "_data" not in scores:
                scores[doc_id + "_data"] = doc

        # Sort by fused score
        ranked_ids = sorted(
            [id for id in scores.keys() if not id.endswith("_data")],
            key=lambda x: scores[x],
            reverse=True,
        )

        fused = []
        for rid in ranked_ids:
            if rid + "_data" in scores:
                fused.append(scores[rid + "_data"])
        return fused

    def _rerank(
        self,
        query: str,
        docs: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Cross-Encoder reranking"""
        if not docs:
            return []

        pairs = [(query, d["text"]) for d in docs]
        scores = self.reranker.predict(pairs)

        # Add scores and sort
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        docs_sorted = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return docs_sorted[:top_k]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Full hybrid retrieval:
        1. Semantic search (top 20)
        2. BM25 search (top 20)
        3. RRF fusion
        4. Cross-Encoder rerank → top K
        """
        # 1. Semantic
        semantic_docs = self._semantic_search(query, top_k=20)

        # 2. BM25
        bm25_docs = self._bm25_search(query, top_k=20)

        # 3. RRF Fusion
        fused = self._rrf_fusion(semantic_docs, bm25_docs)

        # 4. Rerank (top 20 fused → top K)
        final = self._rerank(query, fused[:20], top_k=top_k)

        return final


if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]Hybrid Retriever Test[/bold magenta]")
    console.print("="*70)

    retriever = HybridRetriever()

    # Test queries
    queries = [
        "What is Section 302 IPC?",
        "How to file an FIR?",
        "What are grounds for divorce?",
    ]

    for query in queries:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results, 1):
            console.print(f"\n  [bold]{i}. Source: {r['metadata']['source']}[/bold]")
            console.print(f"     {r['text'][:250]}...")
            console.print(f"     [dim]Rerank score: {r['rerank_score']:.3f}[/dim]")