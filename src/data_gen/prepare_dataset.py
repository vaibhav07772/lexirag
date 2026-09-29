"""Prepare dataset: stats, split, upload-ready format"""
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


def load_jsonl(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def analyze(pairs):
    """Show dataset statistics"""
    console.print("\n[bold cyan]📊 Dataset Statistics[/bold cyan]")
    
    # Length stats
    instr_lens = [len(p["instruction"]) for p in pairs]
    out_lens = [len(p["output"]) for p in pairs]
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Instruction", justify="right")
    table.add_column("Output", justify="right")
    
    table.add_row("Count", str(len(pairs)), str(len(pairs)))
    table.add_row("Min length", str(min(instr_lens)), str(min(out_lens)))
    table.add_row("Max length", str(max(instr_lens)), str(max(out_lens)))
    table.add_row("Avg length", f"{sum(instr_lens)/len(instr_lens):.1f}", f"{sum(out_lens)/len(out_lens):.1f}")
    
    console.print(table)
    
    # Sources breakdown
    console.print("\n[bold]Source breakdown:[/bold]")
    sources = {}
    for p in pairs:
        s = p.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    for s, count in sorted(sources.items(), key=lambda x: -x[1]):
        console.print(f"  • {s}: {count}")


def save_split(pairs, train_file, val_file, val_ratio=0.1):
    """Split into train/val"""
    import random
    random.seed(42)
    random.shuffle(pairs)
    
    n_val = int(len(pairs) * val_ratio)
    val = pairs[:n_val]
    train = pairs[n_val:]
    
    for data, fname in [(train, train_file), (val, val_file)]:
        with open(fname, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    console.print(f"\n[green]✅ Train: {len(train)} → {train_file}[/green]")
    console.print(f"[green]✅ Val:   {len(val)} → {val_file}[/green]")


def main():
    console.print("\n" + "="*70)
    console.print("[bold magenta]LexiRAG — Dataset Prep[/bold magenta]")
    console.print("="*70)
    
    input_file = "data/datasets/legal_qa_dataset.jsonl"
    
    if not Path(input_file).exists():
        console.print(f"[red]❌ Not found: {input_file}[/red]")
        return
    
    pairs = load_jsonl(input_file)
    console.print(f"\n[bold]Loaded: {len(pairs)} pairs[/bold]")
    
    analyze(pairs)
    
    save_split(
        pairs,
        "data/datasets/lexirag_train.jsonl",
        "data/datasets/lexirag_val.jsonl",
        val_ratio=0.1,
    )
    
    console.print("\n[bold green]✅ Dataset ready for Kaggle upload![/bold green]")


if __name__ == "__main__":
    main()