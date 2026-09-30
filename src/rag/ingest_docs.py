"""Ingest legal Q&A dataset into ChromaDB"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
console = Console()

# ─────────────────────────────────────────
# Setup
# ─────────────────────────────────────────
DB_DIR = "./data/chroma_db"
COLLECTION_NAME = "legal_docs"
DATASET_FILE = "data/datasets/legal_qa_dataset.jsonl"

Path(DB_DIR).mkdir(parents=True, exist_ok=True)


def main():
    console.print("\n" + "="*70)
    console.print("[bold magenta]LexiRAG — Ingest Legal Docs[/bold magenta]")
    console.print("="*70)

    # ── Load dataset ──
    if not Path(DATASET_FILE).exists():
        console.print(f"[red]❌ Dataset not found: {DATASET_FILE}[/red]")
        return

    pairs = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    console.print(f"\n📚 Loaded: {len(pairs)} pairs")

    # ── ChromaDB client ──
    console.print("[cyan]🔧 Initializing ChromaDB...[/cyan]")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Use sentence-transformers embedding (local, free)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # small, fast, good quality
    )

    # Delete existing collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        console.print("[yellow]   Old collection deleted[/yellow]")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Ingest in batches ──
    console.print("[cyan]📥 Ingesting documents...[/cyan]")
    BATCH_SIZE = 100

    total = 0
    for i in track(range(0, len(pairs), BATCH_SIZE), description="Ingesting"):
        batch = pairs[i:i + BATCH_SIZE]

        documents = []
        metadatas = []
        ids = []

        for j, p in enumerate(batch):
            # Combine instruction + output for better retrieval
            text = f"Q: {p['instruction']}\nA: {p['output']}"
            if p.get("input"):
                text = f"Q: {p['instruction']}\nContext: {p['input']}\nA: {p['output']}"

            documents.append(text)
            metadatas.append({
                "instruction": p["instruction"][:500],
                "output": p["output"][:1000],
                "input": (p.get("input") or "")[:300],
                "source": p.get("source", "unknown"),
            })
            ids.append(f"doc_{i + j}")

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        total += len(documents)

    console.print(f"\n[green]✅ Ingested: {total} docs[/green]")
    console.print(f"[green]💾 DB saved at: {DB_DIR}[/green]")

    # ── Test query ──
    console.print("\n[bold cyan]🔍 Test Query:[/bold cyan]")
    query = "What is Section 302 IPC?"
    results = collection.query(query_texts=[query], n_results=2)

    console.print(f"\nQuery: {query}")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        console.print(f"\n[bold]{i}. Source: {meta['source']}[/bold]")
        console.print(f"   {doc[:300]}...")

    console.print(f"\n[bold green]✅ RAG pipeline ready![/bold green]")


if __name__ == "__main__":
    main()