import asyncio
from pathlib import Path
import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import DOCS_DIR, SIMILARITY_THRESHOLD
from app.services.pdf_processor import extract_chunks, compute_file_hash
from app.services.embeddings import encode, get_embedding_dimension
from app.services.vector_store import (
    add_chunks,
    load_index,
    add_or_replace_document_chunks,
    get_all_documents,
    get_document_by_id
)
from app.services.retrieval import search, search_with_diagnostics
from app.services.verifier import verify, check_numeric_conflict
from app.services.llm import extract_claims, deduplicate_claims, is_question_sentence
from app.pipeline import verify_question_flow, verify_answer_flow, build_evidence_for_claim
from app.seed import ensure_seed_data

client = TestClient(app)


# Test 1: Health and pipeline status
def test_health_and_model_status():
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert 'service' in data
    assert 'pipeline' in data


# Test 2: PDF chunking & metadata preservation
def test_pdf_chunking_and_metadata_preservation(tmp_path: Path):
    test_pdf_path = tmp_path / "test_scheme.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Government of India Ministry of Agriculture. Farmers are eligible for annual financial support.")
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Annual family income must not exceed 2 lakh rupees. Beneficiaries must possess Aadhaar.")
    doc.save(str(test_pdf_path))
    doc.close()

    chunks, total_pages, pages_text, file_hash, ocr_required = extract_chunks(
        test_pdf_path,
        document_id="test-doc-001",
        document_name="test_scheme.pdf",
        chunk_size=100,
        overlap=20
    )

    assert total_pages == 2
    assert len(pages_text) == 2
    assert len(chunks) >= 2
    assert not ocr_required
    assert len(file_hash) == 64
    for c in chunks:
        assert c['document_id'] == "test-doc-001"
        assert c['document_name'] == "test_scheme.pdf"
        assert 'page_number' in c
        assert 'chunk_id' in c
        assert 'text' in c
        assert len(c['text']) > 0


# Test 3: Empty / Scanned PDF detection
def test_scanned_pdf_detection(tmp_path: Path):
    empty_pdf_path = tmp_path / "empty_scanned.pdf"
    doc_empty = fitz.open()
    doc_empty.new_page()
    doc_empty.save(str(empty_pdf_path))
    doc_empty.close()

    _, _, _, _, ocr_flag = extract_chunks(empty_pdf_path)
    assert ocr_flag is True


# Test 4: Embedding dimension and unit normalization
def test_embedding_dimension_and_normalization():
    texts = ["Eligibility age is 18 to 60 years.", "Income must be under two lakhs."]
    embeddings = encode(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == get_embedding_dimension()
    
    # Test strict unit normalization for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-3)


# Test 5: FAISS persistence and clean document replacement
def test_faiss_persistence_and_clean_replacement():
    doc_id = "test-persist-doc"
    doc_name = "test_persistence.pdf"
    chunks_v1 = [
        {'chunk_id': f"{doc_id}-c0", 'document_id': doc_id, 'document_name': doc_name, 'page_number': 1, 'text': "Version 1 of agricultural policy guidelines."},
        {'chunk_id': f"{doc_id}-c1", 'document_id': doc_id, 'document_name': doc_name, 'page_number': 2, 'text': "Version 1 eligibility requires 2 hectares."}
    ]
    embs_v1 = encode([c['text'] for c in chunks_v1])
    add_or_replace_document_chunks(doc_id, embs_v1, chunks_v1, encode)
    
    # Register in catalog
    from app.services.vector_store import register_document
    register_document({'id': doc_id, 'name': doc_name, 'pages': 2, 'category': 'Test', 'claimsReferenced': 2, 'status': 'Indexed', 'updated': '15 Sep 2026', 'description': 'Test v1', 'pagesText': {}})
    
    docs = get_all_documents()
    assert any(d['id'] == doc_id for d in docs)
    
    # Replace with version 2 (3 chunks)
    chunks_v2 = [
        {'chunk_id': f"{doc_id}-c0", 'document_id': doc_id, 'document_name': doc_name, 'page_number': 1, 'text': "Version 2 of updated agricultural guidelines."},
        {'chunk_id': f"{doc_id}-c1", 'document_id': doc_id, 'document_name': doc_name, 'page_number': 2, 'text': "Version 2 revised criteria to 5 hectares."},
        {'chunk_id': f"{doc_id}-c2", 'document_id': doc_id, 'document_name': doc_name, 'page_number': 3, 'text': "Version 2 additional financial grant."}
    ]
    embs_v2 = encode([c['text'] for c in chunks_v2])
    add_or_replace_document_chunks(doc_id, embs_v2, chunks_v2, encode)
    register_document({'id': doc_id, 'name': doc_name, 'pages': 3, 'category': 'Test', 'claimsReferenced': 3, 'status': 'Indexed', 'updated': '15 Sep 2026', 'description': 'Test v2', 'pagesText': {}})
    
    doc_rec = get_document_by_id(doc_id)
    assert doc_rec is not None
    assert doc_rec['pages'] == 3


# Test 6: Multi-domain retrieval and isolation
def test_domain_isolated_retrieval():
    ensure_seed_data()
    
    # 1. Agriculture Query -> Should retrieve Agriculture document
    agri_hits = search("What are the landholding and income limits for the agricultural welfare scheme?", k=3)
    assert len(agri_hits) > 0
    assert "Agriculture" in agri_hits[0]['document_name']
    assert agri_hits[0]['score'] >= SIMILARITY_THRESHOLD

    # 2. Education Query -> Should retrieve Higher Education document
    edu_hits = search("What minimum percentage of marks is required for higher education scholarship?", k=3)
    assert len(edu_hits) > 0
    assert "Scholarship" in edu_hits[0]['document_name']
    assert edu_hits[0]['score'] >= SIMILARITY_THRESHOLD

    # 3. Healthcare Query -> Should retrieve Healthcare policy
    health_hits = search("What is the annual hospitalization benefit coverage amount?", k=3)
    assert len(health_hits) > 0
    assert "Healthcare" in health_hits[0]['document_name']
    assert health_hits[0]['score'] >= SIMILARITY_THRESHOLD


# Test 7: Out-of-domain retrieval rejection
def test_out_of_domain_retrieval_rejection():
    ensure_seed_data()
    
    # Fabricated claim with zero basis in public documents
    fabricated_claim = "All citizens will receive a free private helicopter on national holidays."
    hits = search(fabricated_claim, k=5)
    
    # Top score must be below relevance threshold or empty
    if hits:
        assert hits[0]['score'] < SIMILARITY_THRESHOLD


# Test 8: NLI Direct Entailment
def test_nli_direct_entailment():
    premise = "The combined annual family income of the applicant from all sources must not exceed Rs 2,00,000 (Rupees Two Lakh only)."
    claim = "The annual family income must not exceed 2 lakh rupees."
    
    status, nli_label, conf, probs, num_conflict = verify(claim, premise)
    assert status == 'SUPPORTED'
    assert nli_label == 'ENTAILMENT'
    assert conf >= 50
    assert probs['entailment'] > 0.50
    assert num_conflict is None


# Test 9: Numeric and condition contradiction detection
def test_nli_numeric_contradiction_detection():
    premise = "Under the welfare scheme, eligible beneficiaries will receive financial assistance of ₹6,000 per year in three installments."
    
    # Currency contradiction
    claim_wrong_amount = "Beneficiaries will receive financial assistance of ₹10,000 per year."
    status, nli_label, conf, probs, conflict = verify(claim_wrong_amount, premise)
    assert status == 'CONTRADICTED'
    assert nli_label == 'CONTRADICTION'
    assert conflict is not None
    assert "6,000" in conflict and "10,000" in conflict

    # Age limit contradiction
    premise_age = "Applicants must be between 18 and 60 years of age."
    claim_wrong_age = "Applicants must be between 25 and 75 years of age."
    status_age, nli_age, _, _, conflict_age = verify(claim_wrong_age, premise_age)
    assert status_age == 'CONTRADICTED'
    assert conflict_age is not None

    # Percentage contradiction
    premise_pct = "Students must secure at least 60% aggregate marks in the qualifying examination."
    claim_wrong_pct = "Students must secure at least 85% aggregate marks."
    status_pct, nli_pct, _, _, conflict_pct = verify(claim_wrong_pct, premise_pct)
    assert status_pct == 'CONTRADICTED'
    assert conflict_pct is not None


# Test 10: Claim deduplication and question filtering
def test_claim_deduplication_and_question_filtering():
    # 1. Question sentence filter
    assert is_question_sentence("What are the eligibility requirements?") is True
    assert is_question_sentence("How do applicants register online?") is True
    assert is_question_sentence("Applicants must be at least 18 years of age.") is False

    # 2. Claim deduplication
    raw_claims = [
        "Applicants must be 18 to 60 years old.",
        "Applicants must be 18 to 60 years old.",  # Exact duplicate
        "What are the eligibility criteria?",       # Question sentence
        "Annual income must not exceed 2 lakhs.",
        "Applicants must be 18 to 60 years old."   # Another duplicate
    ]
    deduped = deduplicate_claims(raw_claims)
    assert len(deduped) == 2
    assert "Applicants must be 18 to 60 years old." in deduped
    assert "Annual income must not exceed 2 lakhs." in deduped


# Test 11: Deterministic 5-status aggregation
def test_deterministic_5_status_aggregation():
    ensure_seed_data()
    
    # 1. All supported -> 'Verified'
    verified_answer = (
        "Under the agricultural scheme, applicants between 18 and 60 years of age are eligible. "
        "The combined annual family income must not exceed Rs 2,00,000."
    )
    r_ver = asyncio.run(verify_answer_flow(verified_answer, "Eligibility test"))
    assert r_ver['status'] == 'Verified'
    assert r_ver['stats']['supported'] == 2
    assert r_ver['stats']['unsupported'] == 0
    assert r_ver['stats']['contradicted'] == 0

    # 2. Mixed supported & unsupported/uncertain -> 'Partially Verified'
    part_answer = (
        "Under the agricultural scheme, applicants between 18 and 60 years of age are eligible. "
        "Beneficiaries will receive a free private helicopter."
    )
    r_part = asyncio.run(verify_answer_flow(part_answer, "Partial test"))
    assert r_part['status'] == 'Partially Verified'
    assert r_part['stats']['supported'] >= 1
    assert r_part['stats']['uncertain'] + r_part['stats']['unsupported'] >= 1

    # 3. Contradicted only -> 'Contradicted'
    contra_answer = "Under the agricultural scheme, annual assistance is ₹25,000 per year."
    r_contra = asyncio.run(verify_answer_flow(contra_answer, "Contradiction test"))
    assert r_contra['status'] == 'Contradicted'
    assert r_contra['stats']['contradicted'] >= 1

    # 4. Purely unsupported -> 'Unsupported'
    unsupp_answer = "All citizens are entitled to an all-expenses-paid trip to Mars."
    r_unsupp = asyncio.run(verify_answer_flow(unsupp_answer, "Unsupported test"))
    assert r_unsupp['status'] == 'Unsupported'
    assert r_unsupp['stats']['unsupported'] >= 1


# Test 12: Full End-to-End Pipeline on Multi-Domain Question Flow
def test_full_e2e_pipeline_multi_domain():
    ensure_seed_data()
    
    # 1. Agriculture question flow
    agri_report = asyncio.run(verify_question_flow("What are the eligibility requirements for the agricultural support scheme?"))
    assert 'id' in agri_report
    assert len(agri_report['claims']) >= 1
    assert agri_report['status'] in ['Verified', 'Partially Verified']
    assert agri_report['coverage'] > 0
    assert agri_report['stats']['total_claims'] == len(agri_report['claims'])
    
    # 2. Education scholarship question flow
    edu_report = asyncio.run(verify_question_flow("What are the scholarship benefits and qualification requirements for higher education?"))
    assert 'id' in edu_report
    assert len(edu_report['claims']) >= 1
    assert edu_report['status'] in ['Verified', 'Partially Verified']
    assert edu_report['coverage'] > 0

    # 3. Healthcare question flow
    health_report = asyncio.run(verify_question_flow("What is the hospital cover amount under the national health protection scheme?"))
    assert 'id' in health_report
    assert len(health_report['claims']) >= 1
    assert health_report['status'] in ['Verified', 'Partially Verified']
    assert health_report['coverage'] > 0


# Test 13: Legacy reports backward compatibility
def test_legacy_report_normalization():
    from app.pipeline import _normalize_report
    legacy_raw = {
        "id": "legacy-test-01",
        "question": "What is the scholarship?",
        "answer": "Scholarship is Rs 12,000 per year.",
        "claims": [
            {
                "id": "c-1",
                "number": 1,
                "text": "Scholarship is Rs 12,000 per year.",
                "status": "SUPPORTED",
                "confidence": 95,
                "reason": "Direct evidence."
            }
        ],
        "createdAt": "14 Sep 2026",
        "coverage": 100,
        "status": "Verified"
    }
    normalized = _normalize_report(legacy_raw)
    assert 'stats' in normalized
    assert normalized['stats']['evidence_coverage'] == 100
    assert normalized['stats']['verification_score'] == 100
    assert normalized['stats']['supported'] == 1
    assert normalized['stats']['total_claims'] == 1

    # Verify /api/reports endpoint succeeds with 200 OK
    res = client.get('/api/reports')
    assert res.status_code == 200
    assert isinstance(res.json(), list)


# Test 14: Existing answer verification mode with mixed statuses
def test_existing_answer_verification_mode():
    ensure_seed_data()
    
    mixed_answer = (
        "Under the agricultural scheme, applicants between 18 and 60 years of age are eligible. "
        "Eligible beneficiaries receive direct financial support of ₹20,000 per year. "
        "All beneficiaries will receive a private moon rover."
    )
    report = asyncio.run(verify_answer_flow(mixed_answer, "Verification of citizen answer"))
    assert report['id'].startswith('report-')
    assert len(report['claims']) >= 3
    
    statuses = {c['status'] for c in report['claims']}
    assert 'SUPPORTED' in statuses
    assert 'CONTRADICTED' in statuses
    assert report['status'] in ['Partially Verified', 'Contradicted']


# Test 15: Out-of-domain question rejection without NLI execution
def test_out_of_domain_question_rejection():
    ensure_seed_data()
    
    fabricated_question = "How many extraterrestrial spacecraft are provided under the intergalactic transit subsidy?"
    report = asyncio.run(verify_question_flow(fabricated_question))
    
    assert report['status'] == 'Unsupported'
    assert report['coverage'] == 0
    assert len(report['claims']) == 0
    assert "No matching official evidence" in report['answer']
