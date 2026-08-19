"""CLI entry point for ConvFinQA evaluation."""

import asyncio

import typer
from dotenv import load_dotenv
from rich import print as rich_print

load_dotenv(override=True)

from src.eval import run_evaluation
from src.ingest import ingest_dataset
from src.models import AblationMode, ModelName

app = typer.Typer(
    name="main",
    help="LLM-driven conversational QA over financial documents (ConvFinQA).",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def ingest(
    data_path: str = typer.Option("data/convfinqa_dataset.json", help="Path to dataset JSON"),
    db_path: str = typer.Option("convfinqa.db", help="Output SQLite path"),
) -> None:
    """Parse the dataset JSON and load into SQLite. Run once before eval."""
    ingest_dataset(data_path=data_path, db_path=db_path)
    rich_print("[green]Ingestion complete.[/green]")


@app.command()
def eval(
    split: str = typer.Option("dev", help="Dataset split: train/dev/test"),
    limit: int = typer.Option(100, help="Max number of conversations to evaluate"),
    db_path: str = typer.Option("convfinqa.db", help="SQLite database path"),
    ablation: str = typer.Option("full", help="Ablation: full/no_sql/no_sandbox/no_history/truncated_history"),
    output: str = typer.Option("results.json", help="Path to write results JSON"),
    data_path: str = typer.Option("data/convfinqa_dataset.json", help="Path to dataset JSON"),
    model: str = typer.Option("gemini-2.5-flash", help="Gemini model: gemini-2.5-flash"),
    feature_hints: str = typer.Option("none", help="Feature hints: none, duplicate_cols, non_numeric, type2, type2_dupcols, all"),
) -> None:
    """Run evaluation suite and report accuracy metrics."""
    try:
        AblationMode(ablation)
    except ValueError:
        rich_print(f"[red]Unknown ablation mode: {ablation}[/red]")
        raise typer.Exit(1)

    model_map = {
        "gemini-2.5-flash": ModelName.GEMINI_25_FLASH,
        "gemini-3.5-flash": ModelName.GEMINI_35_FLASH,
        "gemini-2.5-pro": ModelName.GEMINI_25_PRO,
        "gemini-hybrid": ModelName.GEMINI_HYBRID,
    }
    model_choice = model_map.get(model.lower(), ModelName.GEMINI_25_FLASH)

    summary = asyncio.run(run_evaluation(
        split=split,
        limit=limit,
        db_path=db_path,
        ablation=ablation,
        output_path=output,
        data_path=data_path,
        model=model_choice,
        feature_hints=feature_hints,
    ))

    rich_print("\n[bold]Evaluation Results[/bold]")
    rich_print(f"  Execution accuracy: [green]{summary.execution_accuracy:.1%}[/green] ({summary.correct}/{summary.total})")
    rich_print(f"  By turn:  {summary.by_turn_index}")
    rich_print(f"  By type:  {summary.by_conversation_type}")
    rich_print(f"  By source: {summary.by_answer_source}")
    rich_print(f"  Tokens — in: {summary.total_input_tokens:,} | out: {summary.total_output_tokens:,}")
    rich_print(f"  Est. cost: ${summary.estimated_cost_usd:.4f}")


if __name__ == "__main__":
    app()
