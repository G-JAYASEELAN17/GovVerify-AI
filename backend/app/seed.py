from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import fitz

from app.config import DOCS_DIR, logger
from app.services.pdf_processor import extract_chunks
from app.services.embeddings import encode, release_embedding_model
from app.services.vector_store import add_chunks, register_document, get_all_documents

# Multi-Domain Official Government Corpus for General Verification
OFFICIAL_SEED_DOCUMENTS = [
    {
        'doc_id': 'doc-seed-agri-01',
        'filename': 'National_Agriculture_Development_Scheme_Guidelines.pdf',
        'category': 'Agriculture',
        'description': 'Official operational guidelines for Comprehensive Agricultural Welfare & Income Support Scheme.',
        'pages': [
            (
                "GOVERNMENT OF INDIA\nMINISTRY OF AGRICULTURE & FARMERS WELFARE\n\n"
                "NATIONAL COMPREHENSIVE AGRICULTURAL WELFARE SCHEME\n\n"
                "1. SCHEME OBJECTIVES\n"
                "The scheme provides direct income support to small and marginal farmer families to supplement their financial needs "
                "in procuring agricultural inputs and domestic obligations. All payments are transferred directly via Aadhaar-linked DBT."
            ),
            (
                "2. ELIGIBILITY CRITERIA & CONDITIONS\n\n"
                "2.1 Age Limits:\n"
                "Applicants between 18 and 60 years of age at the time of application are eligible for assistance under the scheme.\n\n"
                "2.2 Landholding & Exclusions:\n"
                "Small and marginal farmer families possessing cultivable landholding up to 2 hectares are eligible. "
                "Institutional landholders and serving government employees paying income tax are strictly excluded."
            ),
            (
                "3. FINANCIAL CEILINGS & PAYMENT SCHEDULE\n\n"
                "3.1 Income Ceiling:\n"
                "The combined annual family income of the applicant from all sources must not exceed Rs 2,00,000 (Rupees Two Lakh only).\n\n"
                "3.2 Assistance Amount:\n"
                "Eligible farmer families receive financial benefit of Rs 6,000 per annum, payable in three equal four-monthly installments of Rs 2,000 each."
            )
        ]
    },
    {
        'doc_id': 'doc-seed-edu-02',
        'filename': 'Higher_Education_Merit_Scholarship_Rules.pdf',
        'category': 'Education',
        'description': 'National Merit Scholarship rules and eligibility for undergraduate and postgraduate university students.',
        'pages': [
            (
                "GOVERNMENT OF INDIA\nMINISTRY OF EDUCATION\n\n"
                "CENTRAL SECTOR MERIT SCHOLARSHIP SCHEME FOR HIGHER EDUCATION\n\n"
                "1. OBJECTIVE\n"
                "To provide financial assistance to meritorious students from low-income families to meet day-to-day expenses while pursuing higher studies."
            ),
            (
                "2. ELIGIBILITY REQUIREMENTS\n\n"
                "2.1 Academic Threshold:\n"
                "Students must have secured above the 80th percentile or a minimum of 75% marks in the Class 12 Board Examination.\n\n"
                "2.2 Age Limit:\n"
                "The applicant must be below 25 years of age on the date of application submission.\n\n"
                "2.3 Income Limit:\n"
                "Gross annual family income of parents must not exceed Rs 4,50,000 per annum from all sources."
            ),
            (
                "3. SCHOLARSHIP AMOUNT & DURATION\n\n"
                "3.1 Financial Rate:\n"
                "The scholarship amount is Rs 12,000 per annum for Undergraduate studies (first 3 years) and Rs 20,000 per annum at Postgraduate level.\n\n"
                "3.2 Renewal Condition:\n"
                "Scholarship is renewed annually subject to maintaining at least 60% marks and 75% attendance in each academic year."
            )
        ]
    },
    {
        'doc_id': 'doc-seed-health-03',
        'filename': 'Universal_Public_Healthcare_Assistance_Policy.pdf',
        'category': 'Healthcare',
        'description': 'Operational manual for cashless secondary and tertiary hospitalization coverage under Ayushman Bharat.',
        'pages': [
            (
                "GOVERNMENT OF INDIA\nNATIONAL HEALTH AUTHORITY\n\n"
                "UNIVERSAL HEALTHCARE ASSISTANCE & HOSPITALIZATION POLICY\n\n"
                "1. SCHEME COVERAGE\n"
                "The healthcare scheme provides a defined benefit cover of Rs 5,00,000 (Rupees Five Lakh) per family per year on a family floater basis."
            ),
            (
                "2. BENEFIT PACKAGE & ENTITLEMENT\n\n"
                "2.1 Cashless Inpatient Care:\n"
                "Beneficiaries are entitled to cashless hospitalization at all empanelled public and private hospitals across the country.\n\n"
                "2.2 Pre-existing Diseases:\n"
                "All pre-existing medical conditions are covered from day one of enrolment without any waiting period.\n\n"
                "2.3 Family Size & Age:\n"
                "There is no cap on family size or age of family members for healthcare benefit entitlement."
            )
        ]
    }
]


def ensure_seed_data():
    """Seeds multi-domain official government PDFs into FAISS index if missing."""
    index, metadata = load_index()
    existing_docs = get_all_documents()
    existing_ids = {d['id'] for d in existing_docs}

    # If index and documents catalog already exist on disk, ensure PDFs exist and return immediately
    for item in OFFICIAL_SEED_DOCUMENTS:
        dest_path = DOCS_DIR / item['filename']
        if not dest_path.exists():
            doc = fitz.open()
            for page_text in item['pages']:
                page = doc.new_page()
                rect = fitz.Rect(50, 50, 550, 750)
                page.insert_textbox(rect, page_text, fontsize=11, fontname="helv")
            doc.save(str(dest_path))
            doc.close()

    # If all seed documents are already registered in the catalog and FAISS index is loaded
    if index is not None and index.ntotal > 0 and len(metadata) > 0 and all(item['doc_id'] in existing_ids for item in OFFICIAL_SEED_DOCUMENTS):
        logger.info(f"Official seed corpus already indexed ({index.ntotal} vectors). Skipping re-embedding.")
        return

    # Otherwise index missing documents
    model_used = False
    for item in OFFICIAL_SEED_DOCUMENTS:
        if item['doc_id'] in existing_ids:
            continue

        dest_path = DOCS_DIR / item['filename']
        chunks, pages_count, pages_text, file_hash, _ = extract_chunks(
            dest_path,
            document_id=item['doc_id'],
            document_name=item['filename']
        )

        if chunks:
            logger.info(f"Seeding missing domain document: {item['filename']} ({item['category']})...")
            embs = encode([c['text'] for c in chunks], batch_size=2)
            add_chunks(embs, chunks)
            model_used = True

        doc_record = {
            'id': item['doc_id'],
            'name': item['filename'],
            'filename': item['filename'],
            'file_hash': file_hash,
            'category': item['category'],
            'pages': pages_count,
            'claimsReferenced': len(chunks),
            'status': 'Indexed',
            'updated': datetime.now().astimezone().strftime('%d %b %Y'),
            'description': item['description'],
            'pagesText': pages_text
        }
        register_document(doc_record)

    if model_used:
        import gc
        release_embedding_model()
        gc.collect()
    logger.info("Multi-domain official government corpus check complete.")

