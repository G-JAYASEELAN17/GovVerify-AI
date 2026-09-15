import type { DocumentRecord, VerificationReport, Claim } from '@/types';

export type ProgressFn = (step: string, pct: number) => void;

export const API_URL = (
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000' : 'https://govverify-ai.onrender.com')
).replace(/\/$/, '');

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  try {
    const res = await fetch(`${API_URL}${path}`, options);
    if (!res.ok) {
      let message = `Backend request failed (${res.status})`;
      try {
        const data = await res.json();
        message = data.detail || message;
      } catch {
        /* ignore json parse */
      }
      throw new Error(message);
    }
    return (await res.json()) as T;
  } catch (err: unknown) {
    if (err instanceof Error && err.message.includes('Failed to fetch')) {
      throw new Error('Unable to connect to GovVerify AI backend. Please ensure the backend server is running on ' + API_URL);
    }
    throw err;
  }
}

const progress = async (onProgress: ProgressFn | undefined, steps: [string, number][]) => {
  for (const [step, pct] of steps) {
    onProgress?.(step, pct);
    await new Promise(r => setTimeout(r, 180));
  }
};

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(10000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function verifyQuestion(
  question: string,
  collection = 'All Government Documents',
  onProgress?: ProgressFn
): Promise<VerificationReport> {
  await progress(onProgress, [
    ['Searching official document index…', 15],
    ['Retrieving relevant evidence…', 30],
    ['Generating evidence-grounded answer…', 48],
    ['Extracting factual claims…', 65],
    ['Running NLI verification…', 82],
  ]);
  return request<VerificationReport>('/api/verify/question', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, collection }),
  }).then(r => {
    onProgress?.('Verification report ready.', 100);
    return r;
  });
}

export async function verifyAnswer(
  answer: string,
  onProgress?: ProgressFn
): Promise<VerificationReport> {
  await progress(onProgress, [
    ['Parsing AI-generated answer…', 20],
    ['Extracting factual claims…', 40],
    ['Retrieving official evidence…', 58],
    ['Running NLI verification…', 82],
  ]);
  return request<VerificationReport>('/api/verify/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer }),
  }).then(r => {
    onProgress?.('Verification report ready.', 100);
    return r;
  });
}

export async function uploadDocument(file: File): Promise<DocumentRecord> {
  const form = new FormData();
  form.append('file', file);
  return request<DocumentRecord>('/api/documents/upload', {
    method: 'POST',
    body: form,
  });
}

export async function getDocuments(): Promise<DocumentRecord[]> {
  return request<DocumentRecord[]>('/api/documents');
}

export async function getDocument(id: string): Promise<DocumentRecord> {
  return request<DocumentRecord>(`/api/documents/${encodeURIComponent(id)}`);
}

export async function getReports(): Promise<VerificationReport[]> {
  return request<VerificationReport[]>('/api/reports');
}

export async function getReport(id: string): Promise<VerificationReport> {
  return request<VerificationReport>(`/api/reports/${encodeURIComponent(id)}`);
}

export async function getClaim(id: string): Promise<Claim | undefined> {
  const reports = await getReports();
  for (const report of reports) {
    const claim = report.claims.find(c => c.id === id);
    if (claim) return claim;
  }
  return undefined;
}

export async function getEvidence(id: string) {
  const claim = await getClaim(id);
  return claim?.evidence;
}
