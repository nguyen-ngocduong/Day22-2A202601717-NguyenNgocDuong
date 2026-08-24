from __future__ import annotations
import json
import os
from pathlib import Path
import typer
from rich import print
from groq import Groq

# Load .env if present
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    with env_file.open() as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

app = typer.Typer(help="Synthetic Data Generation for Preference Alignment")

SYSTEM_PROMPT = """You are an AI data engineer specializing in preference alignment (DPO/ORPO).
Your task is to generate high-quality preference pairs in JSONL format.
Each pair must have:
1. 'prompt': A clear instruction or question.
2. 'chosen': A high-quality, accurate, and helpful response.
3. 'rejected': A plausible but lower-quality response (e.g., contains a subtle error, hallucination, or poor formatting).
4. 'metadata': A dictionary with 'domain' and 'rubric'.

Output ONLY the JSONL lines, one per line. Do not include markdown formatting or extra text."""

USER_PROMPT_TEMPLATE = """Generate {count} new preference pairs about {domain}.
Use the following examples as a style guide:
{examples}

Focus on: {focus}"""

@app.command()
def generate(
    count: int = 5,
    domain: str = "machine learning",
    focus: str = "technical accuracy and safety",
    output_file: Path = Path("data/synthetic_preferences.jsonl"),
    seed_file: Path = Path("data/sample_preferences.jsonl"),
    model: str = "openai/gpt-oss-20b",
) -> None:
    """Generate synthetic preference pairs using Groq."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[red]Error: GROQ_API_KEY environment variable not set.[/red]")
        raise typer.Exit(1)

    client = Groq(api_key=api_key)

    # Load some examples from seed file
    examples_str = ""
    if seed_file.exists():
        with seed_file.open("r") as f:
            lines = [line.strip() for line in f if line.strip()][:3]
            examples_str = "\n".join(lines)

    print(f"Generating [blue]{count}[/blue] pairs for domain: [green]{domain}[/green] using [cyan]{model}[/cyan]...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                count=count, domain=domain, examples=examples_str, focus=focus
            )}
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    if not content:
        print("[red]Error: Received empty response from API.[/red]")
        raise typer.Exit(1)

    # Simple validation and write
    valid_lines = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip markdown code blocks if the model included them
        if line.startswith("```"):
            continue
        try:
            json.loads(line)
            valid_lines.append(line)
        except json.JSONDecodeError:
            print(f"[yellow]Skipping invalid JSON line: {line[:50]}...[/yellow]")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")

    print(f"[green]Successfully added {len(valid_lines)} pairs to {output_file}[/green]")

if __name__ == "__main__":
    app()
