from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class VerifyQuestionRequest(BaseModel):
    question: str = Field(min_length=3, description="User's government-related question")
    collection: str = 'All Government Documents'


class VerifyAnswerRequest(BaseModel):
    answer: str = Field(min_length=5, description="Text containing claims to verify")


class Evidence(BaseModel):
    id: str
    document: str
    page: int
    section: str = 'Official document'
    text: str
    relevance: int = Field(ge=0, le=100)


class Claim(BaseModel):
    id: str
    number: int
    text: str
    status: str  # SUPPORTED | UNCERTAIN | UNSUPPORTED | CONTRADICTED
    confidence: int = Field(ge=0, le=100)
    reason: str
    nli_label: str  # ENTAILMENT | NEUTRAL | CONTRADICTION | NO_RELEVANT_OFFICIAL_EVIDENCE
    nli_probability: Optional[float] = None
    nli_distribution: Optional[Dict[str, float]] = None
    numeric_conflict: Optional[str] = None
    evidence: Optional[Evidence] = None


class ReportStats(BaseModel):
    total_claims: int = 0
    supported: int = 0
    uncertain: int = 0
    unsupported: int = 0
    contradicted: int = 0
    evidence_coverage: int = 0
    verification_score: int = 0
    overall_score: Optional[int] = None


class VerificationReport(BaseModel):
    id: str
    question: str
    answer: str
    claims: List[Claim] = []
    createdAt: str
    coverage: int = 0
    status: str = 'Unsupported'  # Verified | Partially Verified | Contradicted | Uncertain | Unsupported
    stats: Optional[ReportStats] = None
    legacy: Optional[bool] = False


class DocumentRecord(BaseModel):
    id: str
    name: str
    filename: Optional[str] = None
    file_hash: Optional[str] = None
    category: str = 'Government'
    pages: int
    claimsReferenced: int = 0
    status: str  # Indexed | Processing | OCR_Required | Failed
    updated: str
    description: str
    pagesText: Dict[int, str] = {}


class DiagnosticHit(BaseModel):
    rank: int
    document: str
    page: int
    score: float
    threshold: float = 0.38
    accepted: bool = True
    text: str


class RetrievalDiagnosticResponse(BaseModel):
    query: str
    threshold: float
    total_candidates: int
    relevant_hits: List[DiagnosticHit]

