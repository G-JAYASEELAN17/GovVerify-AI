@echo off
echo Running GovVerify AI Automated Tests...
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python -m pytest tests/ -v
