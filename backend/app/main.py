import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import re

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    FRONTEND_ORIGIN,
    CORS_ORIGINS,
    DOCS_DIR,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    logger
)

from app.schemas import (
    VerifyQuestionRequest,
    VerifyAnswerRequest,
    VerificationReport,
    DocumentRecord,
    RetrievalDiagnosticResponse
)
from app.pipeline import (
    verify_question_flow,
    verify_answer_flow,
    reports,
    report_by_id
)
from app.services.pdf_processor import extract_chunks
from app.services.embeddings import encode
from app.services.vector_store import (
    add_or_replace_document_chunks,
    register_document,
    get_all_documents,
    get_document_by_id,
    get_document_by_hash
)
from app.services.retrieval import search_with_diagnostics
from app.seed import ensure_seed_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GovVerify AI FastAPI Server...")
    # Initialize seed data in background thread so HTTP server starts instantly
    asyncio.create_task(asyncio.to_thread(ensure_seed_data))
    yield
    logger.info("GovVerify AI Backend shutting down.")


app = FastAPI(
    title='GovVerify AI API',
    version='2.0.0',
    description='GOV-16 Unsupported Government Claim Detector & Evidence Verification Engine',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get('/')
def root():
    return {
        'project': 'GovVerify AI',
        'problem': 'GOV-16 Unsupported Government Claim Detector',
        'version': '2.0.0',
        'status': 'running'
    }


@app.get('/health')
def health():
    return {
        'status': 'healthy',
        'service': 'GovVerify AI',
        'pipeline': 'PyMuPDF -> BGE-M3 -> FAISS -> DeBERTa-v3 NLI'
    }


@app.post('/api/verify/question', response_model=VerificationReport)
async def verify_question(req: VerifyQuestionRequest):
    logger.info(f"Received question verification request: {req.question[:80]}")
    return await verify_question_flow(req.question)


@app.post('/api/verify/answer', response_model=VerificationReport)
async def verify_answer(req: VerifyAnswerRequest):
    logger.info(f"Received answer verification request: {req.answer[:80]}")
    return await verify_answer_flow(req.answer)


@app.post('/api/documents/upload', response_model=DocumentRecord)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, 'Filename is required.')

    # Sanitize filename to prevent path traversal
    clean_filename = Path(file.filename).name
    ext = Path(clean_filename).suffix.lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f'Invalid file format. Only {", ".join(ALLOWED_EXTENSIONS)} files are supported.')

    safe_name = re.sub(r'[^a-zA-Z0-9_\.-]', '_', clean_filename)
    doc_id = f"doc-{uuid4().hex[:8]}"
    dest_path = DOCS_DIR / f"{doc_id}_{safe_name}"

    # Copy with file size enforcement
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    bytes_written = 0

    with dest_path.open('wb') as out:
        while chunk := await file.read(65536):
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB.")
            out.write(chunk)

    try:
        chunks, pages_count, pages_text, file_hash, ocr_required = extract_chunks(
            dest_path,
            document_id=doc_id,
            document_name=clean_filename
        )

        # Check for duplicate document by SHA-256 hash
        existing_doc = get_document_by_hash(file_hash)
        if existing_doc:
            logger.info(f"Document with hash {file_hash[:8]} already indexed as {existing_doc['name']}.")
            dest_path.unlink(missing_ok=True)
            return existing_doc

        if ocr_required or not chunks:
            # Document is scanned or has no selectable text
            doc_record = {
                'id': doc_id,
                'name': clean_filename,
                'filename': safe_name,
                'file_hash': file_hash,
                'category': 'Government',
                'pages': pages_count,
                'claimsReferenced': 0,
                'status': 'OCR_Required',
                'updated': datetime.now().astimezone().strftime('%d %b %Y'),
                'description': 'Scanned/image-only PDF. Text cannot be extracted without OCR.',
                'pagesText': pages_text
            }
            register_document(doc_record)
            return doc_record

        # Generate embeddings and add to FAISS
        embeddings = encode([c['text'] for c in chunks])
        add_or_replace_document_chunks(doc_id, embeddings, chunks, encode)

        doc_record = {
            'id': doc_id,
            'name': clean_filename,
            'filename': safe_name,
            'file_hash': file_hash,
            'category': 'Government',
            'pages': pages_count,
            'claimsReferenced': len(chunks),
            'status': 'Indexed',
            'updated': datetime.now().astimezone().strftime('%d %b %Y'),
            'description': f'Official government document indexed ({pages_count} pages, {len(chunks)} evidence chunks).',
            'pagesText': pages_text
        }
        register_document(doc_record)
        return doc_record

    except Exception as e:
        dest_path.unlink(missing_ok=True)
        logger.error(f"Failed to process uploaded PDF: {e}")
        raise HTTPException(500, f'Document indexing failed: {e}')


@app.get('/api/documents', response_model=list[DocumentRecord])
def get_documents():
    return get_all_documents()


@app.get('/api/documents/{document_id}', response_model=DocumentRecord)
def get_document(document_id: str):
    doc = get_document_by_id(document_id)
    if not doc:
        raise HTTPException(404, f'Document {document_id} not found')
    return doc


@app.get('/api/diagnostics/retrieve', response_model=RetrievalDiagnosticResponse)
def get_retrieval_diagnostics(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=20)):
    """Diagnostic endpoint to inspect raw semantic retrieval scores for any query."""
    return search_with_diagnostics(q, k=k)


@app.get('/api/reports', response_model=list[VerificationReport])
def get_reports():
    return reports()


@app.get('/api/reports/{report_id}', response_model=VerificationReport)
def get_report(report_id: str):
    r = report_by_id(report_id)
    if not r:
        raise HTTPException(404, f'Report {report_id} not found')
    return r
