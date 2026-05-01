# HackerRank Orchestrate
## Multi-Domain Support Triage
### Harish Balaji R

Terminal-based AI support triage agent built for the HackerRank Orchestrate challenge.

Problem Statement: [HackerRank Support Agent Problem Statement](https://github.com/HarishBalajiR/Multi-Domain-Support-Triage-Challenge/blob/main/problem_statement.md)

## Project Overview

This project processes support tickets across three ecosystems:

- HackerRank
- Claude
- Visa

The agent pipeline:

1. Classifies the domain and request type
2. Applies safety/escalation rules
3. Retrieves relevant local support docs (RAG)
4. Generates grounded responses with Ollama
5. Produces a strict output CSV schema

## Repository Structure

```text
.
├── code/                    # Core agent implementation
├── data/                    # Local support corpus used for retrieval
├── support_tickets/         # Input tickets + generated predictions
├── artifacts/               # Submission-friendly outputs (output.csv, log.txt)
├── AGENTS.md
├── problem_statement.md
├── evalutation_criteria.md
└── README.md
```

## How to Clone and Run on a Windows PC

### 1) Clone the repository

```powershell
git clone https://github.com/HarishBalajiR/Multi-Domain-Support-Triage-Challenge.git
cd "Multi-Domain-Support-Triage-Challenge"
```

### 2) Create and activate a virtual environment

```powershell
cd code
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

### 4) Pull and run local LLM (Ollama)

```powershell
ollama pull llama3.2:3b
```

### 5) Build a retrieval index from the local corpus

```powershell
python main.py ingest
```

### 6) Run on sample set (for benchmark)

```powershell
python main.py eval-sample
```

### 7) Generate final predictions CSV

```powershell
python main.py run --input-csv ..\support_tickets\support_tickets.csv --output-csv ..\support_tickets\output.csv
```


## Metrics History (Project Journey)

The following scores were observed during iterative development and debugging:

| Stage | status | request_type | product_area | Notes |
|---|---:|---:|---:|---|
| Early sample pipeline checks | 10/10 | 10/10 | 3/10 | Product area mapping was weak initially |
| Product-area calibration pass 1 | 10/10 | 10/10 | 6/10 | Weighted vote + canonical mapping improved area |
| Product-area calibration pass 2 | 10/10 | 10/10 | 8/10 | Deterministic area selection in orchestrator |
| Misaligned comparison snapshot | 1/10 | 2/10 | 1/10 | Not apples-to-apples (full output compared to sample labels) |
| Status regression snapshot | 2/10 | 10/10 | 8/10 | Over-escalation due to LLM timeout fallback |
| Final stabilized sample benchmark | 10/10 | 10/10 | 8/10 | Timeout/fallback fixes + prompt optimization |

## Key Fixes That Improved Results

- Added robust NaN/empty value normalization for ticket fields
- Reduced context payload pressure (`top_k=3`, snippet size truncated)
- Set prompt snippet cap to `700` chars for faster CPU inference
- Increased Ollama timeout to reduce invalid/empty model outputs
- Added grounded retrieval fallback for safe tickets instead of blind escalation
- Added status guardrail to prevent safe tickets from being escalated

## Notes

- Inference is local and deterministic (`temperature=0`, fixed seed).
- No external knowledge is used during triage response generation.
- Safety rules explicitly escalate high-risk cases.
