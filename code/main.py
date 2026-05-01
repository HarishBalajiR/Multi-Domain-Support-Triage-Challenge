"""Terminal CLI for the multi-domain support triage agent."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from agent import run_dataframe
from config import INPUT_CSV_PATH, OUTPUT_CSV_PATH, SAMPLE_CSV_PATH
from ingest import build_index, run_smoke_test

console = Console()

OUTPUT_COLUMNS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


def _normalize_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["status"] = out["status"].astype(str).str.lower().replace(
        {"reply": "replied", "escalate": "escalated"}
    )
    out["request_type"] = out["request_type"].astype(str).str.lower()
    return out[OUTPUT_COLUMNS]


def _score(pred: pd.DataFrame, gt: pd.DataFrame) -> tuple[int, int, int, int]:
    total = len(gt)
    p_status = pred["status"].astype(str).str.lower()
    p_req = pred["request_type"].astype(str).str.lower()
    p_area = pred["product_area"].astype(str).str.lower()
    g_status = gt["Status"].astype(str).str.lower()
    g_req = gt["Request Type"].astype(str).str.lower()
    g_area = gt["Product Area"].astype(str).str.lower()
    return (
        int((p_status == g_status).sum()),
        int((p_req == g_req).sum()),
        int((p_area == g_area).sum()),
        total,
    )


def ingest_command(smoke: bool) -> None:
    """Build retrieval index artifacts from local corpus."""
    build_index()
    if smoke:
        run_smoke_test()


def run_command(input_csv: Path, output_csv: Path, max_rows: int) -> None:
    """Run triage and write output CSV in required schema."""
    start = time.time()
    df = pd.read_csv(input_csv)
    if max_rows > 0:
        df = df.head(max_rows)
    out = _normalize_output(run_dataframe(df))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    replied = int((out["status"] == "replied").sum())
    escalated = int((out["status"] == "escalated").sum())
    elapsed = time.time() - start
    console.print(
        f"[green]Done.[/green] rows={len(out)} replied={replied} escalated={escalated} "
        f"elapsed={elapsed:.1f}s output={output_csv}"
    )


def eval_sample_command(max_rows: int) -> None:
    """Evaluate agent against sample_support_tickets.csv labels."""
    df = pd.read_csv(SAMPLE_CSV_PATH)
    if max_rows > 0:
        df = df.head(max_rows)
    pred = _normalize_output(run_dataframe(df))
    status_ok, req_ok, area_ok, total = _score(pred, df)

    table = Table(title="Sample Evaluation")
    table.add_column("Metric")
    table.add_column("Score")
    table.add_row("status", f"{status_ok}/{total}")
    table.add_row("request_type", f"{req_ok}/{total}")
    table.add_row("product_area", f"{area_ok}/{total}")
    console.print(table)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Support triage agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Build retrieval index")
    ingest.add_argument("--smoke", action="store_true", help="Run retrieval smoke test after build")

    run = sub.add_parser("run", help="Run triage on input CSV")
    run.add_argument("--input-csv", type=Path, default=INPUT_CSV_PATH, help="Input ticket CSV path")
    run.add_argument("--output-csv", type=Path, default=OUTPUT_CSV_PATH, help="Output CSV path")
    run.add_argument("--max-rows", type=int, default=0, help="Limit rows for quick test (0=all)")

    eval_cmd = sub.add_parser("eval-sample", help="Evaluate against sample CSV")
    eval_cmd.add_argument("--max-rows", type=int, default=0, help="Limit sample rows for quick check")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    if args.command == "ingest":
        ingest_command(smoke=args.smoke)
    elif args.command == "run":
        run_command(input_csv=args.input_csv, output_csv=args.output_csv, max_rows=args.max_rows)
    elif args.command == "eval-sample":
        eval_sample_command(max_rows=args.max_rows)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
