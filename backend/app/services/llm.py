import json
import re
from typing import List, Optional
import httpx

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, logger

QUESTION_WORDS = {'what', 'how', 'when', 'where', 'who', 'which', 'why', 'is', 'are', 'can', 'does', 'do', 'should', 'could', 'would'}


def is_question_sentence(sentence: str) -> bool:
    """Detects whether a sentence is a user question rather than a factual claim."""
    s = sentence.strip()
    if not s:
        return False
    if s.endswith('?'):
        return True
    first_word = s.split()[0].lower().rstrip(':,?')
    if first_word in QUESTION_WORDS:
        return True
    return False


async def chat(system: str, user: str) -> Optional[str]:
    if not LLM_API_KEY:
        return None
    headers = {'Authorization': f'Bearer {LLM_API_KEY}', 'Content-Type': 'application/json'}
    payload = {
        'model': LLM_MODEL,
        'temperature': 0.1,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user}
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(f'{LLM_BASE_URL}/chat/completions', headers=headers, json=payload)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.warning(f"Backend LLM call failed: {e}. Falling back to grounded excerpting.")
        return None


def synthesize_extractive_answer(evidence: List[dict]) -> str:
    """
    Constructs a deterministic extractive/summarized factual answer from official evidence passages.
    NEVER uses question text.
    """
    if not evidence:
        return "No matching official evidence was found in the indexed government repository."
    
    extracted_sentences = []
    seen = set()
    
    for hit in evidence[:4]:
        text = hit.get('text', '').strip()
        # Split on sentence boundaries
        raw_s = re.split(r'(?<=[.!?])\s+|\n+', text)
        for s in raw_s:
            clean = s.strip()
            clean = re.sub(r'^(?:[•\-\*]|\d+[\.\)])\s*', '', clean)
            if len(clean) >= 20 and not is_question_sentence(clean):
                norm = re.sub(r'[^a-z0-9]', '', clean.lower())
                if norm not in seen:
                    seen.add(norm)
                    extracted_sentences.append(clean)
                    if len(extracted_sentences) >= 4:
                        break
        if len(extracted_sentences) >= 4:
            break
            
    if not extracted_sentences:
        return "Official government evidence is indexed. Please review the specific scheme documentation."
        
    return " ".join(extracted_sentences)


async def generate_answer(question: str, evidence: List[dict]) -> str:
    """
    Generates an evidence-grounded answer using retrieved official passages.
    NEVER reflects the user question as an answer.
    """
    if not evidence:
        return "No relevant official government documents were found in the evidence repository for this query. Please upload the relevant official scheme guidelines and run verification again."

    context = '\n\n'.join(f"[{x.get('document_name', 'Document')} p.{x.get('page_number', 1)}] {x.get('text', '')}" for x in evidence)
    
    prompt_system = (
        "You are GovVerify AI, an official government information verification assistant.\n"
        "RULES:\n"
        "1. Answer the question using ONLY the provided official evidence passages.\n"
        "2. Do NOT invent rules, income limits, eligibility criteria, or benefits.\n"
        "3. Preserve exact numbers, dates, age limits, and condition thresholds.\n"
        "4. Do NOT repeat or echo the user's question. Output ONLY factual assertions grounded strictly in the evidence."
    )
    prompt_user = f"Question:\n{question}\n\nOfficial Evidence:\n{context}\n\nProvide a concise, factual answer grounded strictly in this evidence."

    text = await chat(prompt_system, prompt_user)
    if text and not is_question_sentence(text):
        return text

    # Deterministic extractive synthesis from official passages
    return synthesize_extractive_answer(evidence)


def deduplicate_claims(claims_list: List[str]) -> List[str]:
    """Normalizes and deduplicates claim strings."""
    deduped = []
    seen = set()
    for c in claims_list:
        c_clean = c.strip()
        if not c_clean or len(c_clean) < 10 or is_question_sentence(c_clean):
            continue
        norm = re.sub(r'[^a-z0-9]', '', c_clean.lower())
        if norm not in seen:
            seen.add(norm)
            deduped.append(c_clean)
    return deduped


async def extract_claims(answer: str) -> List[str]:
    """
    Extracts atomic, independently verifiable factual claims from an answer.
    - Strips questions and metadata.
    - Normalizes and deduplicates claims.
    - Preserves numbers, currencies, percentages, and conditions.
    """
    if not answer or len(answer.strip()) < 10 or is_question_sentence(answer):
        return []

    # 1. Try LLM Claim Extraction if available
    system_prompt = (
        "You are an expert factual claim extractor for government policy statements.\n"
        "RULES:\n"
        "1. Extract 1 to 6 standalone, independently verifiable factual assertions from the text.\n"
        "2. Do NOT include questions or user prompts.\n"
        "3. Preserve exact numeric thresholds (e.g. ₹6,000, 18 to 60 years, Rs 2,00,000, 75% marks).\n"
        "4. Deduplicate semantically identical statements.\n"
        "5. Return ONLY a valid JSON array of strings."
    )
    user_prompt = f"Text to extract claims from:\n\"{answer}\"\n\nJSON array:"

    llm_output = await chat(system_prompt, user_prompt)
    if llm_output:
        try:
            clean = llm_output.strip().replace('```json', '').replace('```', '').strip()
            data = json.loads(clean)
            if isinstance(data, list):
                extracted = []
                seen = set()
                for item in data:
                    claim_str = str(item).strip()
                    if len(claim_str) > 12 and not is_question_sentence(claim_str):
                        norm_key = re.sub(r'[^a-z0-9]', '', claim_str.lower())
                        if norm_key not in seen:
                            seen.add(norm_key)
                            extracted.append(claim_str)
                if extracted:
                    logger.info(f"[CLAIM_EXTRACTION] Extracted {len(extracted)} claims via LLM.")
                    return extracted[:6]
        except Exception as e:
            logger.warning(f"Failed to parse LLM claim extraction JSON: {e}")

    # 2. Heuristic extraction with deduplication and question filtering
    raw_candidates: List[str] = []
    
    # Check bullet points / lines
    lines = [l.strip() for l in answer.split('\n') if l.strip()]
    for line in lines:
        cleaned_line = re.sub(r'^(?:[•\-\*]|\d+[\.\)])\s*', '', line).strip()
        cleaned_line = re.sub(r'^(?:Based on|According to|Official evidence|Note that|Please note that)\s+[^:,]+[:,]\s*', '', cleaned_line, flags=re.IGNORECASE).strip()
        if len(cleaned_line) > 15:
            # Check if this line contains multiple sentences
            sub_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'•\-])', cleaned_line)
            for sub in sub_sentences:
                sub_clean = sub.strip()
                if len(sub_clean) > 15 and not is_question_sentence(sub_clean):
                    raw_candidates.append(sub_clean)

    claims = deduplicate_claims(raw_candidates)

    if not claims and not is_question_sentence(answer):
        clean_ans = answer.strip()
        if len(clean_ans) > 15:
            claims = [clean_ans]

    logger.info(f"[CLAIM_EXTRACTION] Extracted and deduplicated {len(claims)} factual claims.")
    return claims[:6]

