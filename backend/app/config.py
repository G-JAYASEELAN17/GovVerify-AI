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
    'https://govverify-ai-nu.vercel.app',
    'https://govverify-ai.vercel.app',
    FRONTEND_ORIGIN,
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:3000',
    'http://127.0.0.1:3000'
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


class InsufficientMemoryError(Exception):
    """Raised when available container/host RAM is insufficient to safely load large AI models without crashing."""
    pass


def get_process_memory_mb() -> float:
    """Returns current process RSS memory in megabytes (MB)."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


def get_container_memory_limit_mb() -> float:
    """Returns the container/cgroup memory limit in MB if available, or total host RAM."""
    # Check cgroup v2
    cgroup_v2 = Path('/sys/fs/cgroup/memory.max')
    if cgroup_v2.exists():
        try:
            val = cgroup_v2.read_text().strip()
            if val != 'max':
                return round(int(val) / (1024 * 1024), 2)
        except Exception:
            pass

    # Check cgroup v1
    cgroup_v1 = Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    if cgroup_v1.exists():
        try:
            val = cgroup_v1.read_text().strip()
            num = int(val)
            if num < (1 << 50):  # Ignore unlimited (e.g. >1PB)
                return round(num / (1024 * 1024), 2)
        except Exception:
            pass

    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 * 1024), 2)
    except Exception:
        return 4096.0


def check_safe_memory_for_model(model_name: str, required_headroom_mb: float = 350.0):
    """
    Checks if there is safe memory headroom to load a model on CPU.
    If running in a <=512MB container (e.g. Render Free) or process memory is too high,
    raises InsufficientMemoryError to prevent an unhandled OS SIGKILL.
    """
    limit_mb = get_container_memory_limit_mb()
    current_rss = get_process_memory_mb()

    # BGE-M3 (560M parameters) requires >1.2GB RAM in float32
    if limit_mb <= 550.0 and 'bge-m3' in model_name.lower():
        logger.warning(
            f"[MEMORY_GUARD] Container memory ceiling is {limit_mb} MB (Render Free ~512MB). "
            f"Loading '{model_name}' (560M params) would trigger kernel OOM SIGKILL. "
            f"Halting load gracefully to protect server uptime."
        )
        raise InsufficientMemoryError(
            f"The verification model '{model_name}' requires more memory than the current deployment instance provides ({limit_mb} MB limit)."
        )

    # General headroom check for small memory instances
    if limit_mb <= 1024.0 and (limit_mb - current_rss) < required_headroom_mb:
        logger.warning(
            f"[MEMORY_GUARD] Current RSS is {current_rss} MB with container limit {limit_mb} MB. "
            f"Insufficient headroom ({limit_mb - current_rss:.1f} MB < {required_headroom_mb} MB) for '{model_name}'."
        )
        raise InsufficientMemoryError(
            "The verification model requires more memory than the current deployment instance provides."
        )


