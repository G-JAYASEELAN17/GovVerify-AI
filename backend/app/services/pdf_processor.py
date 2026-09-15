import hashlib
from pathlib import Path
from typing import Tuple, List, Dict
import fitz


def compute_file_hash(path: Path) -> str:
    """Computes SHA-256 checksum of a file to prevent accidental duplicate indexing."""
    sha256 = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_chunks(
    path: Path,
    document_id: str | None = None,
    document_name: str | None = None,
    chunk_size: int = 350,
    overlap: int = 70
) -> Tuple[List[Dict], int, Dict[int, str], str, bool]:
    """
    Extract page-aware text chunks and full page texts from a PDF using PyMuPDF.
    Preserves:
      - chunk_id
      - document_id
      - document_name
      - filename
      - page_number
      - text
      - source_metadata
    
    Detects scanned/image-only PDFs and flags ocr_required.
    """
    file_hash = compute_file_hash(path)
    doc = fitz.open(path)
    
    chunks: List[Dict] = []
    pages_text: Dict[int, str] = {}
    doc_id = document_id or path.stem
    doc_name = document_name or path.name
    filename = path.name

    total_words_count = 0

    for page_idx, page in enumerate(doc, start=1):
        # Extract plain text preserving layout flow
        raw_text = page.get_text('text').strip()
        pages_text[page_idx] = raw_text
        if not raw_text:
            continue

        words = raw_text.split()
        total_words_count += len(words)
        if not words:
            continue

        start = 0
        c_idx = 1
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk_content = ' '.join(words[start:end]).strip()
            if chunk_content:
                chunk_id = f'{doc_id}-p{page_idx}-c{c_idx}'
                chunks.append({
                    'chunk_id': chunk_id,
                    'document_id': doc_id,
                    'document_name': doc_name,
                    'filename': filename,
                    'page_number': page_idx,
                    'text': chunk_content,
                    'source_metadata': {
                        'total_pages': len(doc),
                        'page': page_idx,
                        'word_count': len(words),
                        'file_hash': file_hash
                    }
                })
                c_idx += 1
            if end >= len(words):
                break
            start = max(0, end - overlap)

    doc.close()
    
    # Scanned or image-only PDF detection (less than 15 words across the entire document)
    ocr_required = (total_words_count < 15)
    
    return chunks, len(pages_text), pages_text, file_hash, ocr_required
