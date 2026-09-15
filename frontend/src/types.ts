export type ClaimStatus = 'SUPPORTED' | 'UNCERTAIN' | 'UNSUPPORTED' | 'CONTRADICTED';

export type ReportOverallStatus = 'Verified' | 'Partially Verified' | 'Contradicted' | 'Uncertain' | 'Unsupported';

export interface Evidence {
  id: string;
  document: string;
  page: number;
  section: string;
  text: string;
  relevance: number;
}

export interface Claim {
  id: string;
  number: number;
  text: string;
  status: ClaimStatus;
  confidence: number;
  reason: string;
  nli_label?: string;
  nli_probability?: number | null;
  nli_distribution?: Record<string, number> | null;
  numeric_conflict?: string | null;
  evidence?: Evidence | null;
}

export interface ReportStats {
  total_claims: number;
  supported: number;
  uncertain: number;
  unsupported: number;
  contradicted: number;
  evidence_coverage: number;
  verification_score: number;
  overall_score?: number;
}

export interface VerificationResult {
  id: string;
  question: string;
  answer: string;
  claims: Claim[];
  createdAt: string;
  stats?: ReportStats;
}

export interface DocumentRecord {
  id: string;
  name: string;
  filename?: string;
  file_hash?: string;
  category: string;
  pages: number;
  claimsReferenced: number;
  status: 'Indexed' | 'Processing' | 'OCR_Required' | 'Failed';
  updated: string;
  description: string;
  pagesText: Record<number, string>;
}

export interface VerificationReport extends VerificationResult {
  coverage: number;
  status: ReportOverallStatus | string;
  stats?: ReportStats;
  legacy?: boolean;
}

