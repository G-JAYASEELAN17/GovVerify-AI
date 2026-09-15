# GovVerify AI

## Verify every government AI claim with official evidence.

GovVerify AI is an evidence-grounded AI verification platform for **GOV-16 — Unsupported Government Claim Detector**. It analyzes government-related AI-generated information, extracts atomic factual claims, retrieves relevant evidence from indexed official government documents, and verifies each claim using semantic retrieval, deterministic numeric/condition analysis, and Natural Language Inference (NLI).

---

## Claim Statuses

Every extracted claim is evaluated against official indexed evidence and classified into one of four deterministic states:

- **SUPPORTED**: The claim is directly corroborated by retrieved official government evidence with high entailment confidence.
- **UNCERTAIN**: Relevant government evidence exists but is ambiguous, incomplete, or insufficient for a conclusive determination.
- **CONTRADICTED**: The claim conflicts directly with official government policy guidelines, eligibility rules, or numeric limits.
- **UNSUPPORTED**: No relevant official evidence was found in the indexed repository to support the claim.

---

## Architecture & Verification Pipeline

```text
User Question / Existing Answer
        ↓
Evidence Retrieval
        ↓
Grounded Answer / Claim Extraction
        ↓
Claim-Specific Retrieval
        ↓
BGE-M3 (Dense Semantic Embeddings)
        ↓
Numeric / Condition Analysis (Deterministic Validator)
        ↓
DeBERTa-v3 NLI (Natural Language Inference)
        ↓
Claim Verification (Entailment / Contradiction / Neutral)
        ↓
Evidence + Source + Page Attribution
        ↓
Comprehensive Verification Report
```

---

## Technology Stack

- **Frontend**:
  - React 18
  - TypeScript
  - Vite
  - Tailwind CSS & Lucide Icons
  - Recharts

- **Backend**:
  - FastAPI (Python 3.10+)
  - Uvicorn (ASGI server)
  - Pydantic v2

- **AI & Natural Language Inference**:
  - `BAAI/bge-m3` (Dense Semantic Retrieval & Embedding Model)
  - `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (Pretrained NLI Cross-Encoder)

- **Vector Store & Indexing**:
  - FAISS (Facebook AI Similarity Search)

- **Document Processing**:
  - PyMuPDF (fitz) for PDF text extraction and chunking

---

## Project Structure

```text
GovVerify-AI/
├── frontend/
│   ├── public/
│   │   ├── assets/
│   │   │   └── govverify-ai-logo.png
│   │   └── favicon.png
│   ├── src/
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── App.tsx
│   │   ├── context.tsx
│   │   ├── index.css
│   │   ├── main.tsx
│   │   └── types.ts
│   ├── package.json
│   ├── package-lock.json
│   ├── index.html
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tsconfig.app.json
│   ├── tsconfig.node.json
│   ├── eslint.config.js
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   └── .env.example
│
├── backend/
│   ├── app/
│   │   ├── services/
│   │   │   ├── embeddings.py
│   │   │   ├── llm.py
│   │   │   ├── pdf_processor.py
│   │   │   ├── retrieval.py
│   │   │   ├── vector_store.py
│   │   │   └── verifier.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── pipeline.py
│   │   ├── schemas.py
│   │   └── seed.py
│   ├── tests/
│   │   └── test_pipeline.py
│   ├── data/
│   ├── requirements.txt
│   ├── .env.example
│   ├── run.bat
│   ├── run.ps1
│   └── test.bat
│
├── .gitignore
├── README.md
└── LICENSE
```

---

## Getting Started

### Prerequisites

- **Node.js**: v18+ (v20+ recommended) and `npm`
- **Python**: v3.10, v3.11, or v3.12
- **Git**

---

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   - **Windows (Command Prompt / PowerShell)**:
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional):
   ```bash
   cp .env.example .env
   ```

5. Start the FastAPI backend server:
   - **Using startup scripts (Windows)**:
     ```cmd
     run.bat
     # or
     .\run.ps1
     ```
   - **Using Uvicorn directly**:
     ```bash
     uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
     ```

6. Run automated test suite:
   ```bash
   pytest tests/ -v
   # or on Windows:
   test.bat
   ```

---

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Configure environment variables (optional):
   ```bash
   cp .env.example .env
   ```
   *By default, `VITE_API_BASE_URL` points to `http://localhost:8000`.*

4. Start development server:
   ```bash
   npm run dev
   ```

5. Build for production:
   ```bash
   npm run build
   ```

---

## Environment Variables

### Backend (`backend/.env.example`)

| Variable | Description | Default / Example |
|---|---|---|
| `LLM_API_KEY` | Optional OpenAI-compatible API key | *(empty for local fallback)* |
| `LLM_BASE_URL` | Base URL for OpenAI-compatible endpoint | `https://api.openai.com/v1` |
| `LLM_MODEL` | LLM model identifier | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | Hugging Face embedding model | `BAAI/bge-m3` |
| `NLI_MODEL` | Hugging Face NLI model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` |
| `RETRIEVAL_THRESHOLD` | Cosine similarity threshold for evidence retrieval | `0.38` |
| `NLI_ENTAILMENT_THRESHOLD` | Confidence cutoff for ENTAILMENT | `0.50` |
| `NLI_CONTRADICTION_THRESHOLD` | Confidence cutoff for CONTRADICTION | `0.45` |
| `FRONTEND_ORIGIN` | Allowed CORS frontend origin | `http://localhost:5173` |
| `MAX_FILE_SIZE_MB` | Maximum PDF upload size limit (MB) | `25` |

### Frontend (`frontend/.env.example`)

| Variable | Description | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Base URL of the running FastAPI backend | `http://localhost:8000` |

---

## Security & Repository Guidelines

- **API Keys & Secrets**: All LLM and service API keys are kept strictly in backend environment variables and must never be committed to Git.
- **Environment Files**: `.env` is ignored in `.gitignore`. Only `.env.example` templates are tracked.
- **Runtime Data**: User-uploaded documents, runtime FAISS vector indexes, and generated reports are stored locally at runtime and are excluded from Git tracking.
- **Clean Repository**: The repository starts clean with automated seed data generation on first startup.

---

## License

This project is licensed under the [MIT License](LICENSE).
