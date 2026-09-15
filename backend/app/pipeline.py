from datetime import datetime
from uuid import uuid4
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from app.config import DATA_DIR, SIMILARITY_THRESHOLD, logger
from app.services.retrieval import search
from app.services.llm import generate_answer, extract_claims, is_question_sentence
from app.services.verifier import verify

REPORTS_PATH = DATA_DIR / 'reports.json'


def _normalize_report(r: Dict) -> Dict:
    """
    Ensures backward compatibility for existing legacy reports by computing
    missing stats (evidence_coverage, verification_score, overall_score)
    and ensuring required fields exist.
    """
    claims = r.get('claims', []) or []
    supported = sum(1 for c in claims if c.get('status') == 'SUPPORTED')
    uncertain = sum(1 for c in claims if c.get('status') == 'UNCERTAIN')
    unsupported = sum(1 for c in claims if c.get('status') == 'UNSUPPORTED')
    contradicted = sum(1 for c in claims if c.get('status') == 'CONTRADICTED')
    total = len(claims)
    
    if total > 0:
        coverage = r.get('coverage', round((supported / total) * 100))
        ver_score = round(((supported * 1.0 + uncertain * 0.5) / total) * 100)
    else:
        coverage = r.get('coverage', 0)
        ver_score = 0

    stats = r.get('stats') or {}
    stats.setdefault('total_claims', total)
    stats.setdefault('supported', supported)
    stats.setdefault('uncertain', uncertain)
    stats.setdefault('unsupported', unsupported)
    stats.setdefault('contradicted', contradicted)
    stats.setdefault('evidence_coverage', coverage)
    stats.setdefault('verification_score', ver_score)
    stats.setdefault('overall_score', coverage)
    
    r['stats'] = stats
    r.setdefault('coverage', coverage)
    r.setdefault('status', 'Unsupported')
    r.setdefault('createdAt', datetime.now().astimezone().strftime('%d %b %Y, %I:%M %p'))

    # Normalize claim schema
    for i, c in enumerate(claims, 1):
        c.setdefault('number', i)
        c.setdefault('status', 'UNCERTAIN')
        c.setdefault('confidence', 50)
        c.setdefault('reason', 'Verification details')
        c.setdefault('nli_label', 'NEUTRAL')
        c.setdefault('nli_probability', None)
        c.setdefault('nli_distribution', None)
        c.setdefault('numeric_conflict', None)
        c.setdefault('evidence', None)

    return r


def _load_reports() -> List[Dict]:
    if REPORTS_PATH.exists():
        try:
            raw = json.loads(REPORTS_PATH.read_text(encoding='utf-8'))
            if isinstance(raw, list):
                return [_normalize_report(r) for r in raw]
        except Exception as e:
            logger.error(f"Error reading reports from {REPORTS_PATH}: {e}")
    return []


def _save_report(r: Dict):
    reports_list = _load_reports()
    normalized = _normalize_report(r)
    reports_list = [normalized] + [x for x in reports_list if x['id'] != normalized['id']]
    REPORTS_PATH.write_text(json.dumps(reports_list[:50], ensure_ascii=False, indent=2), encoding='utf-8')
    logger.info(f"[REPORT_GENERATION] Report {normalized['id']} saved. Coverage: {normalized.get('coverage')}%, Status: {normalized.get('status')}")


def reports() -> List[Dict]:
    return _load_reports()


def report_by_id(i: str) -> Optional[Dict]:
    return next((x for x in _load_reports() if x['id'] == i), None)


def build_evidence_for_claim(claim: str, hits: List[Dict]) -> Tuple[Optional[Dict], Tuple[str, str, int, str, float, Dict[str, float], Optional[str]]]:
    """
    Evaluates evidence against an atomic factual claim:
    1. If no hits or top similarity < SIMILARITY_THRESHOLD:
       Returns UNSUPPORTED with NO_RELEVANT_OFFICIAL_EVIDENCE.
    2. If hits are relevant (>= threshold), evaluates candidates with:
       - Deterministic Numeric and Condition Validator
       - Pretrained DeBERTa-v3 NLI
    """
    if not hits:
        return None, (
            'UNSUPPORTED',
            'NO_RELEVANT_OFFICIAL_EVIDENCE',
            85,
            'No relevant official evidence was found in the indexed documents.',
            0.0,
            {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 0.0},
            None
        )

    best_hit = hits[0]
    similarity = float(best_hit.get('score', 0.0))

    if similarity < SIMILARITY_THRESHOLD:
        return None, (
            'UNSUPPORTED',
            'NO_RELEVANT_OFFICIAL_EVIDENCE',
            max(60, int(round((1.0 - similarity) * 100))),
            'No sufficiently relevant official evidence was found in the indexed documents.',
            0.0,
            {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 0.0},
            None
        )

    evaluated_candidates = []
    # Only test candidates that meet the threshold
    candidates_to_check = [c for c in hits[:3] if float(c.get('score', 0)) >= SIMILARITY_THRESHOLD]
    if not candidates_to_check:
        candidates_to_check = [hits[0]]

    for cand in candidates_to_check:
        try:
            status, nli, conf, probs, num_conflict = verify(claim, cand['text'])
            evaluated_candidates.append((status, nli, conf, cand, probs, num_conflict))
            if status in ['SUPPORTED', 'CONTRADICTED']:
                break
        except Exception as e:
            logger.error(f"NLI verification error on candidate: {e}")
            evaluated_candidates.append(('UNCERTAIN', 'NEUTRAL', 50, cand, {'neutral': 0.5}, None))

    # Priority selection: SUPPORTED > CONTRADICTED > UNCERTAIN
    chosen = next((c for c in evaluated_candidates if c[0] == 'SUPPORTED'), None)
    if not chosen:
        chosen = next((c for c in evaluated_candidates if c[0] == 'CONTRADICTED'), None)
    if not chosen:
        chosen = evaluated_candidates[0]

    status, nli, conf, chosen_hit, probs, num_conflict = chosen
    
    # Calculate display relevance percentage (0-99%)
    relevance_pct = max(1, min(99, int(round(((chosen_hit.get('score', 0.5) + 1.0) / 2.0) * 100))))
    
    evidence = {
        'id': f"ev-{uuid4().hex[:8]}",
        'document': chosen_hit.get('document_name', 'Official Document'),
        'page': chosen_hit.get('page_number', 1),
        'section': 'Official document',
        'text': chosen_hit.get('text', ''),
        'relevance': relevance_pct
    }

    if status == 'SUPPORTED':
        reason = f"Official evidence from {evidence['document']} (p.{evidence['page']}) directly supports this claim."
    elif status == 'CONTRADICTED':
        if num_conflict:
            reason = num_conflict
        else:
            reason = f"Official evidence from {evidence['document']} (p.{evidence['page']}) contradicts this claim."
    else:
        reason = f"Retrieved evidence is on-topic but insufficient or ambiguous to verify this claim."

    dominant_prob = probs.get(nli.lower(), float(conf) / 100.0)
    return evidence, (status, nli, conf, reason, dominant_prob, probs, num_conflict)


async def verify_answer_flow(answer: str, question: str = '') -> Dict:
    """
    EXISTING ANSWER PIPELINE:
    answer → extract factual claims → verify each claim independently against indexed evidence
    """
    # If the user passed a question into verify_answer_flow, seamlessly route to question flow
    if is_question_sentence(answer.strip()):
        logger.info(f"[ROUTING] Input is a question sentence. Routing to verify_question_flow: '{answer[:60]}...'")
        return await verify_question_flow(answer.strip())

    claims_text = await extract_claims(answer)
    
    # If no factual claims can be extracted from text
    if not claims_text:
        report = {
            'id': f'report-{uuid4().hex[:10]}',
            'question': question if question else (answer[:80] + '...' if len(answer) > 80 else answer),
            'answer': answer,
            'claims': [],
            'createdAt': datetime.now().astimezone().strftime('%d %b %Y, %I:%M %p'),
            'coverage': 0,
            'status': 'Unsupported',
            'stats': {
                'total_claims': 0,
                'supported': 0,
                'uncertain': 0,
                'unsupported': 0,
                'contradicted': 0,
                'evidence_coverage': 0,
                'verification_score': 0,
                'overall_score': 0
            }
        }
        _save_report(report)
        return report

    claims: List[Dict] = []
    for i, claim in enumerate(claims_text, 1):
        # Independent retrieval per atomic claim
        hits = search(claim, k=5)
        evidence, result_tuple = build_evidence_for_claim(claim, hits)
        status, nli, conf, reason, nli_prob, probs, num_conflict = result_tuple

        claims.append({
            'id': f'claim-{uuid4().hex[:10]}',
            'number': i,
            'text': claim,
            'status': status,
            'confidence': conf,
            'reason': reason,
            'nli_label': nli,
            'nli_probability': round(nli_prob, 4),
            'nli_distribution': probs,
            'numeric_conflict': num_conflict,
            'evidence': evidence
        })

    supported_count = sum(c['status'] == 'SUPPORTED' for c in claims)
    uncertain_count = sum(c['status'] == 'UNCERTAIN' for c in claims)
    unsupported_count = sum(c['status'] == 'UNSUPPORTED' for c in claims)
    contradicted_count = sum(c['status'] == 'CONTRADICTED' for c in claims)
    total = max(1, len(claims))
    coverage = round((supported_count / total) * 100)
    verification_score = round(((supported_count * 1.0 + uncertain_count * 0.5) / total) * 100)

    # Deterministic 5-status aggregation:
    if supported_count == total:
        overall_status = 'Verified'
    elif supported_count > 0:
        overall_status = 'Partially Verified'
    elif contradicted_count > 0:
        overall_status = 'Contradicted'
    elif uncertain_count > 0:
        overall_status = 'Uncertain'
    else:
        overall_status = 'Unsupported'

    stats = {
        'total_claims': len(claims),
        'supported': supported_count,
        'uncertain': uncertain_count,
        'unsupported': unsupported_count,
        'contradicted': contradicted_count,
        'evidence_coverage': coverage,
        'verification_score': verification_score,
        'overall_score': coverage
    }

    report = {
        'id': f'report-{uuid4().hex[:10]}',
        'question': question if question else (answer[:80] + '...' if len(answer) > 80 else answer),
        'answer': answer,
        'claims': claims,
        'createdAt': datetime.now().astimezone().strftime('%d %b %Y, %I:%M %p'),
        'coverage': coverage,
        'status': overall_status,
        'stats': stats
    }
    
    _save_report(report)
    return report


async def verify_question_flow(question: str) -> Dict:
    """
    QUESTION PIPELINE:
    question → retrieve evidence passages → generate grounded answer → extract claims from answer → verify claims
    """
    # 1. Retrieve relevant official evidence
    hits = search(question, k=5)
    
    # 2. If no evidence passes the similarity threshold
    if not hits or float(hits[0].get('score', 0.0)) < SIMILARITY_THRESHOLD:
        logger.info(f"[QUESTION_PIPELINE] No relevant evidence found above threshold {SIMILARITY_THRESHOLD} for: '{question[:60]}'")
        no_ev_answer = "No matching official evidence was found in the indexed government repository for this query. Please upload the relevant official scheme guidelines."
        report = {
            'id': f'report-{uuid4().hex[:10]}',
            'question': question,
            'answer': no_ev_answer,
            'claims': [],
            'createdAt': datetime.now().astimezone().strftime('%d %b %Y, %I:%M %p'),
            'coverage': 0,
            'status': 'Unsupported',
            'stats': {
                'total_claims': 0,
                'supported': 0,
                'uncertain': 0,
                'unsupported': 0,
                'contradicted': 0,
                'evidence_coverage': 0,
                'verification_score': 0,
                'overall_score': 0
            }
        }
        _save_report(report)
        return report

    # 3. Generate grounded answer strictly from retrieved official evidence
    answer = await generate_answer(question, hits)
    
    # 4. Extract atomic factual claims FROM THE GENERATED ANSWER (never the question)
    claims_text = await extract_claims(answer)
    if not claims_text:
        # Extractive fallback on declarative sentences of generated answer
        sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 15 and not is_question_sentence(s.strip())]
        claims_text = sentences[:4] if sentences else []

    if not claims_text:
        report = {
            'id': f'report-{uuid4().hex[:10]}',
            'question': question,
            'answer': answer,
            'claims': [],
            'createdAt': datetime.now().astimezone().strftime('%d %b %Y, %I:%M %p'),
            'coverage': 0,
            'status': 'Unsupported',
            'stats': {
                'total_claims': 0,
                'supported': 0,
                'uncertain': 0,
                'unsupported': 0,
                'contradicted': 0,
                'evidence_coverage': 0,
                'verification_score': 0,
                'overall_score': 0
            }
        }
        _save_report(report)
        return report

    # 5. Verify each factual claim independently
    claims: List[Dict] = []
    for i, claim in enumerate(claims_text, 1):
        claim_hits = search(claim, k=5)
        evidence, result_tuple = build_evidence_for_claim(claim, claim_hits)
        status, nli, conf, reason, nli_prob, probs, num_conflict = result_tuple

        claims.append({
            'id': f'claim-{uuid4().hex[:10]}',
            'number': i,
            'text': claim,
            'status': status,
            'confidence': conf,
            'reason': reason,
            'nli_label': nli,
            'nli_probability': round(nli_prob, 4),
            'nli_distribution': probs,
            'numeric_conflict': num_conflict,
            'evidence': evidence
        })

    supported_count = sum(c['status'] == 'SUPPORTED' for c in claims)
    uncertain_count = sum(c['status'] == 'UNCERTAIN' for c in claims)
    unsupported_count = sum(c['status'] == 'UNSUPPORTED' for c in claims)
    contradicted_count = sum(c['status'] == 'CONTRADICTED' for c in claims)
    total = max(1, len(claims))
    coverage = round((supported_count / total) * 100)
    verification_score = round(((supported_count * 1.0 + uncertain_count * 0.5) / total) * 100)

    if supported_count == total:
        overall_status = 'Verified'
    elif supported_count > 0:
        overall_status = 'Partially Verified'
    elif contradicted_count > 0:
        overall_status = 'Contradicted'
    elif uncertain_count > 0:
        overall_status = 'Uncertain'
    else:
        overall_status = 'Unsupported'

    stats = {
        'total_claims': len(claims),
        'supported': supported_count,
        'uncertain': uncertain_count,
        'unsupported': unsupported_count,
        'contradicted': contradicted_count,
        'evidence_coverage': coverage,
        'verification_score': verification_score,
        'overall_score': coverage
    }

    report = {
        'id': f'report-{uuid4().hex[:10]}',
        'question': question,
        'answer': answer,
        'claims': claims,
        'createdAt': datetime.now().astimezone().strftime('%d %b %Y, %I:%M %p'),
        'coverage': coverage,
        'status': overall_status,
        'stats': stats
    }
    
    _save_report(report)
    return report

