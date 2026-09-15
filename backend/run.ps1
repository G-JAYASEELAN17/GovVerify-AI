$ErrorActionPreference = 'Stop'
if (Test-Path ..\.venv\Scripts\Activate.ps1) {
    ..\.venv\Scripts\Activate.ps1
} elseif (Test-Path .venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
} else {
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
}
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
