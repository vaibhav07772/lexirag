"""Output Guardrail — Citation validation + PII leak + Toxicity check"""
import re
from typing import Dict, List
from rich.console import Console

console = Console()


# Citation patterns (Indian legal)
CITATION_PATTERNS = [
    r"Section\s+\d+[A-Z]?\s+(?:of\s+)?(?:IPC|CrPC|CPC|Indian\s+Penal\s+Code|Code\s+of\s+Criminal\s+Procedure)",
    r"Article\s+\d+[A-Z]?\s+(?:of\s+)?(?:the\s+)?Constitution",
    r"(?:IPC|CrPC|CPC)\s+Section\s+\d+[A-Z]?",
    r"Section\s+\d+[A-Z]?",
    r"Article\s+\d+[A-Z]?",
]

# Toxicity keywords (basic)
TOXIC_PATTERNS = [
    r"\b(kill|murder|rape|terrorist|bomb)\s+(?:you|him|her|them|everyone)",
    r"\b(hate|stupid|idiot|moron)\s+(?:you|him|her)",
    r"\b(how\s+to\s+make\s+(?:a\s+)?bomb)",
]

# PII patterns (same as input for leak check)
PII_LEAK_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE_IN": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
}

MIN_CITATIONS = 1  # Legal answers should have at least 1 citation


class OutputGuard:
    """Output validation and safety"""

    def __init__(self):
        self.citation_compiled = [re.compile(p, re.IGNORECASE) for p in CITATION_PATTERNS]
        self.toxic_compiled = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

    def extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from output"""
        citations = []
        for pattern in self.citation_compiled:
            matches = pattern.findall(text)
            citations.extend(matches)
        # Dedupe while preserving order
        seen = set()
        unique = []
        for c in citations:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)
        return unique

    def validate_citations(self, text: str, retrieved_docs: List[Dict]) -> Dict:
        """Check if citations are valid (exist in retrieved docs)"""
        citations = self.extract_citations(text)

        if not citations:
            return {
                "valid": False,
                "citations": [],
                "invalid_citations": [],
                "reason": "No legal citation found in answer",
            }

        # Build combined context from retrieved docs
        context = " ".join(
            d.get("text", "") + " " + str(d.get("metadata", {}))
            for d in retrieved_docs
        ).lower()

        valid = []
        invalid = []
        for citation in citations:
            # Extract section number
            num_match = re.search(r"\d+[A-Z]?", citation)
            if num_match and num_match.group(0).lower() in context:
                valid.append(citation)
            elif citation.lower() in context:
                valid.append(citation)
            else:
                invalid.append(citation)

        return {
            "valid": len(valid) > 0,
            "citations": valid,
            "invalid_citations": invalid,
            "reason": f"{len(valid)} valid, {len(invalid)} not in context",
        }

    def check_toxicity(self, text: str) -> List[str]:
        """Detect toxic content"""
        detected = []
        for pattern in self.toxic_compiled:
            match = pattern.search(text)
            if match:
                detected.append(match.group(0)[:50])
        return detected

    def check_pii_leak(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in output (should not appear)"""
        found = {}
        for pii_type, pattern in PII_LEAK_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found[pii_type] = matches
        return found

    def validate(self, text: str, retrieved_docs: List[Dict] = None) -> Dict:
        """Full output validation"""
        result = {
            "valid": True,
            "issues": [],
            "citations": [],
            "invalid_citations": [],
            "toxicity": [],
            "pii_leaks": {},
        }

        # 1. Citation check
        if retrieved_docs:
            citation_result = self.validate_citations(text, retrieved_docs)
            result["citations"] = citation_result["citations"]
            result["invalid_citations"] = citation_result["invalid_citations"]

            if not citation_result["valid"]:
                result["valid"] = False
                result["issues"].append(f"Citation issue: {citation_result['reason']}")
            elif citation_result["invalid_citations"]:
                result["issues"].append(
                    f"Hallucinated citations detected: {citation_result['invalid_citations']}"
                )

        # 2. Toxicity check
        toxic = self.check_toxicity(text)
        if toxic:
            result["valid"] = False
            result["toxicity"] = toxic
            result["issues"].append(f"Toxic content: {toxic}")

        # 3. PII leak check
        pii = self.check_pii_leak(text)
        if pii:
            result["valid"] = False
            result["pii_leaks"] = pii
            result["issues"].append(f"PII leak: {list(pii.keys())}")

        return result


if __name__ == "__main__":
    console.print("\n" + "="*70)
    console.print("[bold magenta]Output Guardrail Test[/bold magenta]")
    console.print("="*70)

    guard = OutputGuard()

    # Fake retrieved docs
    fake_docs = [
        {"text": "Section 302 of IPC deals with punishment for murder."},
        {"text": "Section 154 CrPC is about filing FIR."},
    ]

    test_cases = [
        (
            "Section 302 of IPC deals with punishment for murder.",
            "valid",
        ),
        (
            "Under Section 154 CrPC, any person can file an FIR.",
            "valid",
        ),
        (
            "According to Section 999 IPC, XYZ is illegal.",  # Hallucinated
            "hallucinated",
        ),
        (
            "Just answer without any citation.",  # No citation
            "no_citation",
        ),
        (
            "Contact test@example.com for more info.",  # PII leak
            "pii_leak",
        ),
    ]

    for text, label in test_cases:
        console.print(f"\n[cyan]Case ({label}):[/cyan] {text[:80]}")
        result = guard.validate(text, fake_docs)
        console.print(f"[bold]Valid:[/bold] {result['valid']}")
        if result["citations"]:
            console.print(f"  [green]Citations:[/green] {result['citations']}")
        if result["invalid_citations"]:
            console.print(f"  [red]Invalid:[/red] {result['invalid_citations']}")
        if result["issues"]:
            for issue in result["issues"]:
                console.print(f"  [red]⚠️  {issue}[/red]")