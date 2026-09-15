@echo off
echo Running GovVerify AI Automated Tests...
cd /d "%~dp0"
if exist ..\.venv (
    call ..\.venv\Scripts\activate.bat
) else (
    call .\.venv\Scripts\activate.bat
)
python -m pytest tests/ -v
