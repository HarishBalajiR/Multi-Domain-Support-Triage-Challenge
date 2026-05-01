"""Configuration values for the support triage agent."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUPPORT_TICKETS_DIR = PROJECT_ROOT / "support_tickets"
OUTPUT_CSV_PATH = SUPPORT_TICKETS_DIR / "output.csv"
SAMPLE_CSV_PATH = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"
INPUT_CSV_PATH = SUPPORT_TICKETS_DIR / "support_tickets.csv"
INDEX_DIR = PROJECT_ROOT / "code" / ".index"
INDEX_VECTORS_PATH = INDEX_DIR / "vectors.npy"
INDEX_META_PATH = INDEX_DIR / "meta.jsonl"
INDEX_MANIFEST_PATH = INDEX_DIR / "manifest.json"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RANDOM_SEED = 42
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
RETRIEVAL_CANDIDATES = 20
RETRIEVAL_TOP_K = 4
MMR_LAMBDA = 0.5
