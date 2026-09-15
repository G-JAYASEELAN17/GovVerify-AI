import re
from functools import lru_cache
from typing import Tuple, Dict, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import NLI_MODEL, NLI_ENTAILMENT_THRESHOLD, NLI_CONTRADICTION_THRESHOLD, logger


@lru_cache(maxsize=1)
def get_nli():
    logger.info(f"Loading NLI verification model: {NLI_MODEL}")
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    return tok, model


def extract_numeric_tokens(text: str) -> Dict[str, List[float]]:
    """
    Extracts structured numeric values by context (currency, age, percentage, general numbers).
    Normalizes terms like '2 lakh' -> 200000, '5 lakh' -> 500000, '₹6,000' -> 6000.
    """
    results: Dict[str, List[float]] = {
        'currency': [],
        'age': [],
        'percentage': [],
        'general': []
    }
    
    t = text.lower().replace(',', '')

    # 1. Lakh / Crore currency
    lakh_matches = re.findall(r'(?:rs\.?|₹|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lac)', t)
    for m in lakh_matches:
        try:
            results['currency'].append(float(m) * 100000)
        except ValueError:
            pass

    # 2. Direct currency symbols
    curr_matches = re.findall(r'(?:rs\.?|₹|inr)\s*(\d+)', t)
    for m in curr_matches:
        try:
            results['currency'].append(float(m))
        except ValueError:
            pass

    # 3. Percentages
    pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', t)
    for m in pct_matches:
        try:
            results['percentage'].append(float(m))
        except ValueError:
            pass

    # 4. Ages (e.g. "between 18 and 60 years", "below 25 years", "age 18")
    age_patterns = [
        r'(?:age|aged|years of age|years old)\s*(?:of|is|between|below|above)?\s*(\d+)',
        r'between\s+(\d+)\s+and\s+(\d+)\s*(?:years|years of age)?',
        r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years|years of age)',
        r'(?:below|under|above|over|exceeding)\s*(\d+)\s*(?:years|years of age)',
        r'(\d+)\s*(?:years of age|years old|years)',
    ]
    for pat in age_patterns:
        matches = re.findall(pat, t)
        for m in matches:
            if isinstance(m, tuple):
                for sub in m:
                    if sub:
                        results['age'].append(float(sub))
            elif m:
                results['age'].append(float(m))

    # 5. Generic integers
    gen_matches = re.findall(r'\b\d+\b', t)
    for m in gen_matches:
        try:
            results['general'].append(float(m))
        except ValueError:
            pass

    return results


def check_numeric_conflict(claim: str, evidence: str) -> Tuple[bool, Optional[str]]:
    """
    Heuristic validation to detect explicit numeric/threshold contradictions that semantic
    similarity or embeddings might otherwise mask.
    """
    claim_nums = extract_numeric_tokens(claim)
    ev_nums = extract_numeric_tokens(evidence)

    # Check Currency / Financial discrepancies
    if claim_nums['currency'] and ev_nums['currency']:
        # If claim mentions a specific amount that is not in evidence amounts
        c_set = set(claim_nums['currency'])
        e_set = set(ev_nums['currency'])
        if not c_set.intersection(e_set):
            c_val = next(iter(c_set))
            e_val = next(iter(e_set))
            return True, f"Official document specifies ₹{int(e_val):,} which conflicts with ₹{int(c_val):,} in the claim."

    # Check Percentage discrepancies
    if claim_nums['percentage'] and ev_nums['percentage']:
        c_pct = set(claim_nums['percentage'])
        e_pct = set(ev_nums['percentage'])
        if not c_pct.intersection(e_pct):
            return True, f"Official document specifies {next(iter(e_pct))}% which conflicts with {next(iter(c_pct))}% in the claim."

    # Check Age discrepancies (e.g. claim says 21 when evidence says 18 or 25)
    if claim_nums['age'] and ev_nums['age']:
        c_age = set(claim_nums['age'])
        e_age = set(ev_nums['age'])
        if not c_age.intersection(e_age):
            return True, f"Official document specifies age {int(next(iter(e_age)))} which conflicts with age {int(next(iter(c_age)))} in the claim."

    return False, None


def verify(claim: str, evidence: str) -> Tuple[str, str, int, Dict[str, float], Optional[str]]:
    """
    Evaluates Natural Language Inference and Numeric Validation between:
      Premise: Official retrieved government evidence
      Hypothesis: Extracted factual claim
    
    Returns:
      (status, nli_label, confidence_pct, probabilities_dict, numeric_conflict_reason)
      status: 'SUPPORTED' | 'CONTRADICTED' | 'UNCERTAIN'
    """
    # 1. First check explicit numeric and condition contradictions
    has_conflict, conflict_reason = check_numeric_conflict(claim, evidence)
    if has_conflict:
        logger.info(f"[NUMERIC_CONTRADICTION] claim='{claim[:50]}...' -> CONTRADICTED ({conflict_reason})")
        prob_dict = {'entailment': 0.05, 'neutral': 0.05, 'contradiction': 0.90}
        return 'CONTRADICTED', 'CONTRADICTION', 92, prob_dict, conflict_reason

    # 2. Run pretrained DeBERTa-v3 NLI
    tok, model = get_nli()
    inputs = tok(evidence, claim, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Dynamic label mapping from model config
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    
    entail_idx = next((k for k, v in id2label.items() if 'entail' in v), 0)
    neutral_idx = next((k for k, v in id2label.items() if 'neutral' in v), 1)
    contrad_idx = next((k for k, v in id2label.items() if 'contrad' in v), 2)

    p_entail = float(probs[entail_idx])
    p_neutral = float(probs[neutral_idx])
    p_contrad = float(probs[contrad_idx])

    prob_dict = {
        'entailment': round(p_entail, 4),
        'neutral': round(p_neutral, 4),
        'contradiction': round(p_contrad, 4)
    }

    # Classification logic using configurable thresholds
    if p_entail >= NLI_ENTAILMENT_THRESHOLD and p_entail > max(p_neutral, p_contrad):
        status = 'SUPPORTED'
        mapped = 'ENTAILMENT'
        confidence = int(round(p_entail * 100))
    elif p_contrad >= NLI_CONTRADICTION_THRESHOLD and p_contrad > p_entail:
        status = 'CONTRADICTED'
        mapped = 'CONTRADICTION'
        confidence = int(round(p_contrad * 100))
    else:
        status = 'UNCERTAIN'
        mapped = 'NEUTRAL'
        confidence = int(round(max(p_neutral, 0.50) * 100))


    logger.info(f"[NLI_VERIFICATION] claim='{claim[:50]}...' -> {status} (Entail: {p_entail:.2f}, Neut: {p_neutral:.2f}, Contrad: {p_contrad:.2f})")
    return status, mapped, confidence, prob_dict, None
