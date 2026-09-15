import os
import logging
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / '.env')

# CPU Environment & Memory Optimization for constrained environments (e.g. Render 512MB)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


DATA_DIR = BASE_DIR / 'data'
DOCS_DIR = DATA_DIR / 'documents'
INDEX_DIR = DATA_DIR / 'index'
UPLOADS_DIR = DATA_DIR / 'uploads'
DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained AI Models
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
NLI_MODEL = os.getenv('NLI_MODEL', 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')

# Backend-only LLM Configuration
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')

# Retrieval & Verification Thresholds
TOP_K = int(os.getenv('TOP_K', '5'))
SIMILARITY_THRESHOLD = float(os.getenv('RETRIEVAL_THRESHOLD', os.getenv('SIMILARITY_THRESHOLD', '0.48')))
NLI_ENTAILMENT_THRESHOLD = float(os.getenv('NLI_ENTAILMENT_THRESHOLD', '0.50'))
NLI_CONTRADICTION_THRESHOLD = float(os.getenv('NLI_CONTRADICTION_THRESHOLD', '0.45'))

# Security & Upload Constraints
FRONTEND_ORIGIN = os.getenv('FRONTEND_ORIGIN', 'http://localhost:5173')
CORS_ORIGINS_RAW = os.getenv('CORS_ORIGINS', '')
CORS_ORIGINS = [orig.strip() for orig in CORS_ORIGINS_RAW.split(',') if orig.strip()] if CORS_ORIGINS_RAW else [
    FRONTEND_ORIGIN,
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:3000',
    '*'
]
MAX_FILE_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', os.getenv('MAX_FILE_SIZE_MB', '25')))
MAX_UPLOAD_SIZE_MB = MAX_FILE_SIZE_MB
MAX_CHUNKS_PER_BATCH = int(os.getenv('MAX_CHUNKS_PER_BATCH', '8'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '2'))
ALLOWED_EXTENSIONS = {'.pdf'}

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger('GovVerify')


def get_process_memory_mb() -> float:
    """Returns current process RSS memory in megabytes (MB)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


