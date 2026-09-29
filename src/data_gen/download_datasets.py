"""Download ready-made Indian legal datasets from HuggingFace"""
import json
from pathlib import Path
from datasets import load_dataset
from rich.console import Console

console = Console()


def download_mani_dataset():
    """RMani1/indian-legal-dataset-indian-law — 1500 pairs, SFT format"""
    console.print("\n[cyan]📥 Downloading RMani1/indian-legal-dataset-indian-law...[/cyan]")
    
    try:
        ds = load_dataset(
            "RMani1/indian-legal-dataset-indian-law",
            split="train",
        )
        console.print(f"   [green]✅ Loaded: {len(ds)} pairs[/green]")
        console.print(f"   Columns: {ds.column_names}")
        return ds
    except Exception as e:
        console.print(f"   [red]❌ Failed: {str(e)[:150]}[/red]")
        return None


def download_techmaestro_dataset():
    """Techmaestro369 — loads each legal file separately (mismatched schemas)"""
    console.print("\n[cyan]📥 Downloading Techmaestro369/indian-legal-texts-finetuning...[/cyan]")
    
    files = ["constitution_qa.json", "crpc_qa.json", "ipc_qa.json"]
    all_rows = []
    
    for fname in files:
        try:
            ds = load_dataset(
                "Techmaestro369/indian-legal-texts-finetuning",
                data_files=fname,
                split="train",
            )
            console.print(f"   [green]✅ {fname}: {len(ds)} pairs[/green]")
            all_rows.extend([dict(ex) for ex in ds])
        except Exception as e:
            console.print(f"   [yellow]⚠️ {fname} failed: {str(e)[:120]}[/yellow]")
    
    if not all_rows:
        console.print("   [red]❌ No data loaded[/red]")
        return None
    
    console.print(f"   [green]✅ Total Techmaestro: {len(all_rows)} pairs[/green]")
    return all_rows


def normalize_mani(example):
    """Convert RMani format to standard instruction/input/output"""
    return {
        "instruction": example.get("instruction", "").strip(),
        "input": example.get("input", "").strip(),
        "output": example.get("output", "").strip(),
        "source": "RMani1/indian-legal",
    }


def normalize_techmaestro(example):
    """Convert Techmaestro format to standard instruction/input/output"""
    question = (example.get("question") or "").strip()
    answer = (example.get("answer") or "").strip()
    section = (example.get("section") or "").strip()
    source_doc = (example.get("source") or "").strip()
    
    # Combine source doc + section as input
    input_text = f"{source_doc} — {section}".strip(" —")
    
    return {
        "instruction": question,
        "input": input_text,
        "output": answer,
        "source": "Techmaestro369/indian-legal-texts",
    }


def save_jsonl(pairs, output_file):
    """Save as JSONL"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    valid = 0
    seen = set()
    with output_file.open("w", encoding="utf-8") as f:
        for pair in pairs:
            # Skip empty
            if not pair["instruction"] or not pair["output"]:
                continue
            # Deduplicate by instruction
            key = pair["instruction"][:100].lower()
            if key in seen:
                continue
            seen.add(key)
            
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            valid += 1
    
    console.print(f"\n[green]💾 Saved: {output_file} ({valid} valid pairs)[/green]")
    return valid


def main():
    console.print("\n" + "="*70)
    console.print("[bold magenta]LexiRAG — Dataset Download[/bold magenta]")
    console.print("="*70)
    
    all_pairs = []
    
    # 1. RMani dataset
    mani_ds = download_mani_dataset()
    if mani_ds:
        mani_pairs = [normalize_mani(ex) for ex in mani_ds]
        all_pairs.extend(mani_pairs)
    
    # 2. Techmaestro dataset
    tech_ds = download_techmaestro_dataset()
    if tech_ds:
        tech_pairs = [normalize_techmaestro(ex) for ex in tech_ds]
        all_pairs.extend(tech_pairs)
    
    # 3. Save combined
    if all_pairs:
        console.print(f"\n[bold]Total raw pairs: {len(all_pairs)}[/bold]")
        save_jsonl(all_pairs, "data/datasets/legal_qa_dataset.jsonl")
    else:
        console.print("\n[red]❌ No datasets downloaded[/red]")
        return
    
    # 4. Show samples
    console.print("\n[bold cyan]Sample pairs:[/bold cyan]")
    for i, pair in enumerate(all_pairs[:3], 1):
        console.print(f"\n[bold]{i}.[/bold]")
        console.print(f"  [yellow]Q:[/yellow] {pair['instruction'][:150]}")
        console.print(f"  [green]A:[/green] {pair['output'][:200]}...")
        console.print(f"  [dim]Source: {pair['source']}[/dim]")
    
    console.print("\n[bold green]✅ Dataset ready![/bold green]")


if __name__ == "__main__":
    main()