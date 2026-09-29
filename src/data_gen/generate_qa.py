"""Generate Indian legal Q&A dataset using Groq"""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────
# Legal topics (IPC, CrPC, Constitution)
# ─────────────────────────────────────────
LEGAL_TOPICS = [
    # IPC sections
    "Section 302 IPC - Murder punishment",
    "Section 304 IPC - Culpable homicide",
    "Section 307 IPC - Attempt to murder",
    "Section 376 IPC - Rape",
    "Section 420 IPC - Cheating",
    "Section 499 IPC - Defamation",
    "Section 506 IPC - Criminal intimidation",
    # CrPC
    "Section 154 CrPC - FIR registration",
    "Section 161 CrPC - Examination of witnesses",
    "Section 167 CrPC - Remand procedure",
    "Section 173 CrPC - Police report",
    # Constitution
    "Article 21 - Right to life and liberty",
    "Article 14 - Equality before law",
    "Article 19 - Freedom of speech",
    "Article 32 - Constitutional remedies",
    "Article 226 - Writ jurisdiction of High Court",
    # Contract Act
    "Section 10 Contract Act - Valid contract",
    "Section 23 Contract Act - Unlawful agreements",
    "Section 56 Contract Act - Impossibility of performance",
    # Family Law
    "Hindu Marriage Act - Divorce grounds",
    "Section 125 CrPC - Maintenance",
    # Property
    "Transfer of Property Act - Sale deed",
    "Section 53A TPA - Part performance",
    # Company Law
    "Companies Act 2013 - Director duties",
]


def generate_qa_pair(topic: str) -> dict:
    """Generate 3 Q&A pairs for a legal topic"""
    
    prompt = f"""You are an Indian legal expert. Generate 3 question-answer pairs about: {topic}

Rules:
- Questions should be practical (what lawyers actually ask)
- Answers must be accurate per Indian law
- Cite specific sections/articles
- Include exceptions/caveats if applicable
- Answer in 100-200 words

Return ONLY valid JSON:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "answer": "...",
      "citation": "Section X of Y",
      "topic": "{topic}"
    }}
  ]
}}

Start with {{ and end with }}."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    
    content = response.choices[0].message.content
    
    try:
        data = json.loads(content)
        return data.get("qa_pairs", [])
    except json.JSONDecodeError:
        return []


def main():
    output_file = Path("data/datasets/legal_qa_dataset.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_pairs = []
    failed = []
    
    console.print(f"\n[bold cyan]Generating Q&A for {len(LEGAL_TOPICS)} topics...[/bold cyan]")
    console.print("Target: 3 pairs per topic = ~75 pairs\n")
    
    for i, topic in enumerate(track(LEGAL_TOPICS, description="Generating...")):
        try:
            pairs = generate_qa_pair(topic)
            all_pairs.extend(pairs)
            time.sleep(1)  # Rate limit
        except Exception as e:
            failed.append((topic, str(e)[:100]))
            console.print(f"[red]❌ {topic}: {e}[/red]")
    
    # Save JSONL
    with output_file.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    console.print(f"\n[bold green]✅ Generated: {len(all_pairs)} pairs[/bold green]")
    console.print(f"[yellow]Failed: {len(failed)} topics[/yellow]")
    console.print(f"[green]💾 Saved: {output_file}[/green]")


if __name__ == "__main__":
    main()