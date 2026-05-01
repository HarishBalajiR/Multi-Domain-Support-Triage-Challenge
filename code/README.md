# Support Triage Agent (Code)

## Setup

1. Create and activate a virtual environment:
   - PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `python -m pip install --upgrade pip`
   - `python -m pip install -r requirements.txt`
3. Ensure Ollama is running and pull the model:
   - `ollama pull llama3.2:3b`

## Run

Entry point is `main.py` (Typer CLI).

### 1) Build retrieval index

```powershell
.venv\Scripts\python.exe main.py ingest
```

Optional smoke check:

```powershell
.venv\Scripts\python.exe main.py ingest --smoke
```

### 2) Evaluate on sample tickets

```powershell
.venv\Scripts\python.exe main.py eval-sample
```

Quick test on first 3 rows:

```powershell
.venv\Scripts\python.exe main.py eval-sample --max-rows 3
```

### 3) Run on input tickets and write output

```powershell
.venv\Scripts\python.exe main.py run --input-csv ..\support_tickets\support_tickets.csv --output-csv ..\support_tickets\output.csv
```

Quick test on first 5 rows:

```powershell
.venv\Scripts\python.exe main.py run --max-rows 5
```

The output CSV schema is always:

`issue,subject,company,response,product_area,status,request_type,justification`
