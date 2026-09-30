"""RAG QA Pipeline — Retrieval + Generation with citations"""
import os
import json
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from groq import Groq

from src.rag.hybrid_retriever import HybridRetriever

load_dotenv()
console = Console()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


PROMPT_TEMPLATE = """You are a helpful Indian legal assistant. Answer the question based ONLY on the provided context.

Rules:
1. Answer ONLY using information from the context below
2. Cite the specific Section/Article (e.g., "Section 302 IPC")
3. If the context doesn't contain the answer, say "I don't have enough information"
4. Keep answers clear and concise (100-200 words)
5. Do NOT hallucinate or add information not in context

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


class RAGPipeline:
    """End-to-end RAG QA system"""

    def __init__(self, top_k: int = 5):
        console.print("[cyan]🔧 Initializing RAG Pipeline...[/cyan]")
        self.retriever = HybridRetriever()
        self.top_k = top_k
        console.print("[green]✅ RAG Pipeline ready[/green]")

    def _build_context(self, docs: List[Dict]) -> str:
        """Format retrieved docs into context"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            text = doc["text"]
            source = doc["metadata"].get("source", "unknown")
            score = doc.get("rerank_score", 0)
            context_parts.append(f"[{i}] {text}\n(Source: {source}, relevance: {score:.2f})")
        return "\n\n".join(context_parts)

    def _call_llm(self, context: str, question: str) -> str:
        """Call Groq LLM"""
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile" if False else "openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def ask(self, question: str) -> Dict:
        """Full RAG: retrieve + generate"""
        # 1. Retrieve
        docs = self.retriever.retrieve(question, top_k=self.top_k)

        if not docs:
            return {
                "answer": "No relevant legal documents found.",
                "sources": [],
                "confidence": 0.0,
            }

        # 2. Build context
        context = self._build_context(docs)

        # 3. Generate
        answer = self._call_llm(context, question)

        # 4. Confidence: based on top rerank score
        top_score = docs[0].get("rerank_score", 0)
        # Normalize: scores typically range -10 to 10
        confidence = min(max((top_score + 5) / 15, 0.0), 1.0)

        return {
            "answer": answer,
            "sources": [
                {
                    "snippet": d["text"][:200],
                    "source": d["metadata"].get("source", "unknown"),
                    "score": d.get("rerank_score", 0),
                }
                for d in docs
            ],
            "confidence": confidence,
        }


if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]RAG QA Pipeline Test[/bold magenta]")
    console.print("="*70)

    pipeline = RAGPipeline(top_k=5)

    test_questions = [
        "What is Section 302 IPC?",
        "How to file an FIR?",
    ]

    for q in test_questions:
        console.print(f"\n[bold cyan]❓ Question:[/bold cyan] {q}")
        result = pipeline.ask(q)
        console.print(f"\n[bold green]✅ Answer:[/bold green]")
        console.print(f"{result['answer']}")
        console.print(f"\n[dim]Confidence: {result['confidence']:.2f}[/dim]")
        console.print(f"[dim]Sources: {len(result['sources'])}[/dim]")

        console.print(f"\n[dim]Top sources:[/dim]")
        for i, s in enumerate(result["sources"][:3], 1):
            console.print(f"  [{i}] {s['source']} (score: {s['score']:.2f})")
            console.print(f"      {s['snippet'][:150]}...")
        console.print("\n" + "-"*70)