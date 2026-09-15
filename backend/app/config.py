import os
import logging
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / '.env')

DATA_DIR = BASE_DIR / 'data'
DOCS_DIR = DATA_DIR / 'documents'
INDEX_DIR = DATA_DIR / 'index'
DOCS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

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
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '25'))
ALLOWED_EXTENSIONS = {'.pdf'}

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger('GovVerify')

