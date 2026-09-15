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
    UPLOADS_DIR,
    MAX_UPLOAD_SIZE_MB,
    MAX_CHUNKS_PER_BATCH,
    EMBEDDING_BATCH_SIZE,
    ALLOWED_EXTENSIONS,
    get_process_memory_mb,
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
from app.services.pdf_processor import extract_chunks, stream_pdf_chunks
from app.services.embeddings import encode, get_embedding_model, release_embedding_model
from app.services.vector_store import (
    add_chunks,
    add_or_replace_document_chunks,
    register_document,
    get_all_documents,
    get_document_by_id,
    get_document_by_hash
)
from app.services.retrieval import search_with_diagnostics


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GovVerify AI FastAPI Server (Lightweight CPU mode)...")
    # Lazy startup: Models and seed data are initialized on demand
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
    logger.info(f"[GovVerify] Upload received | RAM: {get_process_memory_mb()} MB")

    if not file.filename:
        raise HTTPException(400, 'Filename is required.')

    # Sanitize filename to prevent path traversal
    clean_filename = Path(file.filename).name
    ext = Path(clean_filename).suffix.lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f'Invalid file format. Only {", ".join(ALLOWED_EXTENSIONS)} files are supported.')

    safe_name = re.sub(r'[^a-zA-Z0-9_\.-]', '_', clean_filename)
    doc_id = f"doc-{uuid4().hex[:8]}"
    dest_path = UPLOADS_DIR / f"{doc_id}_{safe_name}"

    # Stream file to disk with strict file size enforcement
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    bytes_written = 0

    try:
        with dest_path.open('wb') as out:
            while chunk := await file.read(65536):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(413, f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_MB} MB.")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        logger.error(f"Error saving upload stream: {e}")
        raise HTTPException(500, f"Failed to save uploaded file: {e}")

    logger.info(f"[GovVerify] PDF saved | RAM: {get_process_memory_mb()} MB")
    logger.info(f"[GovVerify] PDF processing started | RAM: {get_process_memory_mb()} MB")

    try:
        total_chunks_indexed = 0
        total_pages = 1
        file_hash = ""
        is_ocr_required = False
        summary_pages_text = {}
        model_loaded = False

        # Stream chunks in bounded batches
        for chunk_batch, is_last, num_pages, f_hash, ocr_flag, p_text in stream_pdf_chunks(
            dest_path,
            document_id=doc_id,
            document_name=clean_filename,
            batch_size=MAX_CHUNKS_PER_BATCH
        ):
            total_pages = num_pages
            file_hash = f_hash
            is_ocr_required = ocr_flag
            summary_pages_text = p_text

            # Check for duplicate document on first iteration
            if not total_chunks_indexed and not model_loaded:
                existing_doc = get_document_by_hash(file_hash)
                if existing_doc:
                    logger.info(f"Document with hash {file_hash[:8]} already indexed as {existing_doc['name']}.")
                    dest_path.unlink(missing_ok=True)
                    logger.info(f"[GovVerify] Upload completed | RAM: {get_process_memory_mb()} MB")
                    return existing_doc

            # Scanned / empty PDF detection
            if is_ocr_required and not chunk_batch and not total_chunks_indexed:
                break

            if chunk_batch:
                if not model_loaded:
                    logger.info(f"[GovVerify] Before BGE-M3 load | RAM: {get_process_memory_mb()} MB")
                    get_embedding_model()
                    model_loaded = True
                    logger.info(f"[GovVerify] After BGE-M3 load | RAM: {get_process_memory_mb()} MB")

                logger.info(f"[GovVerify] Before embedding batch | RAM: {get_process_memory_mb()} MB")
                batch_embs = encode([c['text'] for c in chunk_batch], batch_size=EMBEDDING_BATCH_SIZE)
                logger.info(f"[GovVerify] After embedding batch | RAM: {get_process_memory_mb()} MB")

                add_chunks(batch_embs, chunk_batch)
                total_chunks_indexed += len(chunk_batch)
                logger.info(f"[GovVerify] After FAISS update | RAM: {get_process_memory_mb()} MB")

                del batch_embs
                del chunk_batch

        # Release embedding model immediately after indexing
        if model_loaded:
            release_embedding_model()

        import gc
        gc.collect()
        logger.info(f"[GovVerify] After cleanup | RAM: {get_process_memory_mb()} MB")

        if is_ocr_required and total_chunks_indexed == 0:
            doc_record = {
                'id': doc_id,
                'name': clean_filename,
                'filename': safe_name,
                'file_hash': file_hash,
                'category': 'Government',
                'pages': total_pages,
                'claimsReferenced': 0,
                'status': 'OCR_Required',
                'updated': datetime.now().astimezone().strftime('%d %b %Y'),
                'description': 'Scanned/image-only PDF. Text cannot be extracted without OCR.',
                'pagesText': summary_pages_text
            }
            register_document(doc_record)
            logger.info(f"[GovVerify] Upload completed | RAM: {get_process_memory_mb()} MB")
            return doc_record

        doc_record = {
            'id': doc_id,
            'name': clean_filename,
            'filename': safe_name,
            'file_hash': file_hash,
            'category': 'Government',
            'pages': total_pages,
            'claimsReferenced': total_chunks_indexed,
            'status': 'Indexed',
            'updated': datetime.now().astimezone().strftime('%d %b %Y'),
            'description': f'Official government document indexed ({total_pages} pages, {total_chunks_indexed} evidence chunks).',
            'pagesText': summary_pages_text
        }
        register_document(doc_record)
        logger.info(f"[GovVerify] Upload completed | RAM: {get_process_memory_mb()} MB")
        return doc_record

    except HTTPException:
        dest_path.unlink(missing_ok=True)
        release_embedding_model()
        raise
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        release_embedding_model()
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
