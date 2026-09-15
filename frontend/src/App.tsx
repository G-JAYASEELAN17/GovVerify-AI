import { useEffect, useRef, useState } from 'react';
import { BrowserRouter, Link, NavLink, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  ArrowRight, BarChart3, BookOpen, Check, CheckCircle2, ChevronDown, ChevronRight,
  CircleAlert, ClipboardCheck, Clock3, Copy, Download, ExternalLink, FileText, Filter,
  Info, Loader2, Menu, MessageSquareText, Moon, MoreHorizontal, Search, Settings2,
  ShieldCheck, Sparkles, Sun, Trash2, Upload, X, Zap, Cpu,
} from 'lucide-react';
import { getDocument, uploadDocument, verifyAnswer, verifyQuestion, API_URL } from '@/services/api';
import { useApp } from '@/context';
import type { Claim, ClaimStatus, DocumentRecord, VerificationReport } from '@/types';
import '@/index.css';

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic Page Title Manager
// ─────────────────────────────────────────────────────────────────────────────

function PageTitleManager() {
  const location = useLocation();
  useEffect(() => {
    const p = location.pathname;
    if (p === '/') {
      document.title = 'GovVerify AI — Government Claim Verification';
    } else if (p.startsWith('/verify')) {
      document.title = 'GovVerify AI — Verify';
    } else if (p.startsWith('/verification')) {
      document.title = 'GovVerify AI — Verify';
    } else if (p.startsWith('/results')) {
      document.title = 'GovVerify AI — Results';
    } else if (p.startsWith('/claims')) {
      document.title = 'GovVerify AI — Results';
    } else if (p.startsWith('/documents')) {
      document.title = 'GovVerify AI — Documents';
    } else if (p.startsWith('/reports')) {
      document.title = 'GovVerify AI — Reports';
    } else if (p.startsWith('/how-it-works')) {
      document.title = 'GovVerify AI — How It Works';
    } else if (p.startsWith('/about')) {
      document.title = 'GovVerify AI — About';
    } else {
      document.title = 'GovVerify AI — Government Claim Verification';
    }
  }, [location.pathname]);
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation & Header
// ─────────────────────────────────────────────────────────────────────────────

const navItems: [string, string][] = [
  ['Home', '/'], ['Verify', '/verify'], ['Reports', '/reports'],
  ['Documents', '/documents'], ['How It Works', '/how-it-works'], ['About', '/about'],
];

function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/" className="logo" aria-label="GovVerify AI Home">
      <img
        src="/assets/govverify-ai-logo.png"
        alt="GovVerify AI logo"
        className={`logo-img ${compact ? 'compact' : ''}`}
      />
      <span>GovVerify <b>AI</b></span>
      <small>GOV-16</small>
    </Link>
  );
}

function Navbar() {
  const [open, setOpen] = useState(false);
  const { darkMode, toggleDark, hasApiKey, backendConnected } = useApp();
  return (
    <header className="navbar">
      <div className="nav-inner">
        <Logo compact />
        <nav className={open ? 'nav-links open' : 'nav-links'}>
          {navItems.map(([label, path]) => (
            <NavLink key={path} to={path} onClick={() => setOpen(false)}
              className={({ isActive }) => isActive ? 'active' : ''}>
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="nav-actions">
          {hasApiKey && (
            <span className="ai-badge" title="Backend LLM active">
              <Cpu size={11} /> AI Active
            </span>
          )}
          {backendConnected === false ? (
            <span className="offline" title={`FastAPI backend is unreachable on ${API_URL}`}>
              <CircleAlert size={12} /> Backend Offline
            </span>
          ) : (
            <span className="online"><i /> System Online</span>
          )}
          <button className="dark-toggle" onClick={toggleDark} aria-label="Toggle dark mode">
            {darkMode ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <Link className="button button-primary button-sm" to="/verify">
            Start Verification <ArrowRight size={14} />
          </Link>
        </div>
        <button className="menu-toggle" onClick={() => setOpen(!open)} aria-label="Toggle navigation">
          <Menu size={22} />
        </button>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer>
      <div className="footer-inner">
        <div>
          <Logo compact />
          <p style={{ marginTop: '12px', fontWeight: 600, color: '#dbe7f4' }}>
            GOV-16 — Unsupported Government Claim Detector
          </p>
          <p style={{ fontStyle: 'italic', color: '#93adc9', marginTop: '6px', fontSize: '12px' }}>
            "Verify every government AI claim with official evidence."
          </p>
          <p style={{ color: '#7a93b2', fontSize: '11px', marginTop: '8px' }}>
            GovVerify AI verifies factual claims in AI-generated government information against retrieved official evidence and provides traceable sources for every verification decision.
          </p>
        </div>
        <div className="footer-links">
          <div>
            <strong>Product</strong>
            <Link to="/verify">Verify</Link>
            <Link to="/reports">Reports</Link>
            <Link to="/documents">Documents</Link>
          </div>
          <div>
            <strong>Learn</strong>
            <Link to="/how-it-works">How It Works</Link>
            <Link to="/about">About GovVerify AI</Link>
            <span>GOV-16 Problem</span>
          </div>
          <div>
            <strong>Trust & Verification</strong>
            <span>PyMuPDF Extraction</span>
            <span>BGE-M3 Embeddings</span>
            <span>DeBERTa-v3 NLI</span>
            <span>FAISS Retrieval</span>
          </div>
        </div>
      </div>
      <div className="footer-bottom">
        <span>© 2026 GovVerify AI · All rights reserved.</span>
        <span>GOV-16 — Unsupported Government Claim Detector</span>
      </div>
    </footer>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared UI primitives
// ─────────────────────────────────────────────────────────────────────────────

function Button({
  children, variant = 'primary', className = '', ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' }) {
  return (
    <button className={`button button-${variant} ${className}`} {...props}>
      {children}
    </button>
  );
}

function StatusBadge({ status }: { status: ClaimStatus | VerificationReport['status'] }) {
  const lower = status.toLowerCase().replace(/\s+/g, '-');
  return <span className={`status-badge ${lower}`}><i />{status}</span>;
}

function Metric({ label, value, tone = 'blue', icon }: {
  label: string; value: string | number; tone?: string; icon?: React.ReactNode;
}) {
  return (
    <div className={`metric-card ${tone}`}>
      <span className="metric-icon">{icon}</span>
      <div><strong>{value}</strong><span>{label}</span></div>
    </div>
  );
}

function PageHeader({ eyebrow, title, description, action }: {
  eyebrow?: string; title: string; description?: string; action?: React.ReactNode;
}) {
  return (
    <div className="page-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="section-label">{children}</div>;
}

function LoadingState({ text = 'Loading…' }: { text?: string }) {
  return (
    <div className="state-card">
      <Loader2 className="spin" size={28} />
      <strong>{text}</strong>
      <span>Preparing the latest verification data.</span>
    </div>
  );
}

function EmptyState({ icon = <ShieldCheck size={38} />, title, text, action }: {
  icon?: React.ReactNode; title: string; text: string; action?: React.ReactNode;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">{icon}</div>
      <h3>{title}</h3>
      <p>{text}</p>
      {action}
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="confidence">
      <div><span>Confidence</span><b>{value}%</b></div>
      <span className="bar"><i style={{ width: `${value}%` }} /></span>
    </div>
  );
}

function ClaimCard({ claim, expanded = false }: { claim: Claim; expanded?: boolean }) {
  const [show, setShow] = useState(expanded);
  const navigate = useNavigate();
  return (
    <article className={`claim-card ${claim.status.toLowerCase()}`}>
      <button className="claim-summary" onClick={() => setShow(!show)}>
        <span className="claim-number">{String(claim.number).padStart(2, '0')}</span>
        <span className="claim-text">{claim.text}</span>
        <StatusBadge status={claim.status} />
        <span className="claim-score">{claim.confidence}%</span>
        <ChevronDown size={16} className={show ? 'rotate' : ''} />
      </button>
      {show && (
        <div className="claim-detail">
          <div className="reason">
            <strong>Verification reasoning</strong>
            <p>{claim.reason}</p>
            {claim.numeric_conflict && (
              <div className="numeric-conflict-alert" style={{ marginTop: '8px', padding: '8px 12px', background: 'rgba(239, 68, 68, 0.1)', borderLeft: '3px solid #ef4444', borderRadius: '4px', fontSize: '13px', color: '#b91c1c' }}>
                <strong>Numeric / Condition Conflict:</strong> {claim.numeric_conflict}
              </div>
            )}
            {claim.nli_label && (
              <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--muted)', letterSpacing: '0.04em' }}>
                NLI EVALUATION: <span style={{ fontWeight: 600, color: 'var(--text)' }}>{claim.nli_label}</span>
                {claim.nli_probability !== undefined && claim.nli_probability !== null && ` (${Math.round(claim.nli_probability * 100)}% prob)`}
              </div>
            )}
          </div>
          <ConfidenceBar value={claim.confidence} />
          {claim.evidence ? (
            <div className="evidence-preview">
              <div className="evidence-heading">
                <FileText size={15} />
                <span>Retrieved official evidence</span>
                <span className="relevance">{claim.evidence.relevance}% relevant</span>
              </div>
              <p>"{claim.evidence.text}"</p>
              <div className="source-line">
                <span>{claim.evidence.document}</span>
                <span>Page {claim.evidence.page}</span>
                <span>{claim.evidence.section}</span>
              </div>
              <Button variant="secondary" onClick={() => navigate(`/claims/${claim.id}`)}>
                View Evidence <ArrowRight size={14} />
              </Button>
            </div>
          ) : (
            <div className="unsupported-note">
              <CircleAlert size={16} /> Unsupported by retrieved evidence in indexed documents
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function ScoreRing({ score }: { score: number }) {
  return (
    <div className="score-ring" style={{ '--score': `${score * 3.6}deg` } as React.CSSProperties}>
      <div>
        <strong>{score}%</strong>
        <span>Evidence<br />Coverage</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Home Page
// ─────────────────────────────────────────────────────────────────────────────

const heroPreviewClaims = [
  { id: 'h-1', text: 'Applicants must be between 18 and 60 years old.', status: 'SUPPORTED' as const },
  { id: 'h-2', text: 'Annual family income must be below ₹2 lakh.', status: 'SUPPORTED' as const },
  { id: 'h-3', text: 'Applicants must have an Aadhaar card.', status: 'UNCERTAIN' as const },
  { id: 'h-4', text: 'Applicants must own agricultural land.', status: 'UNSUPPORTED' as const },
];

function Home() {
  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <div className="hero-logo-badge">
            <img src="/assets/govverify-ai-logo.png" alt="GovVerify AI logo" className="logo-img" />
            <span style={{ fontWeight: 800, fontSize: '12px', letterSpacing: '0.6px', color: 'var(--blue)' }}>
              GOV-16 — UNSUPPORTED GOVERNMENT CLAIM DETECTOR
            </span>
          </div>
          <h1 style={{ marginBottom: '8px' }}>GovVerify AI</h1>
          <h2 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--navy)', margin: '0 0 12px', letterSpacing: '-0.5px' }}>
            AI-Powered Government Claim Verification
          </h2>
          <div className="hero-tagline">
            "Verify every government AI claim with official evidence."
          </div>
          <p>
            GovVerify AI verifies factual claims in AI-generated government information against retrieved official evidence and provides traceable sources for every verification decision.
          </p>
          <div className="hero-actions">
            <Link className="button button-primary" to="/verify">
              Start Verification <ArrowRight size={16} />
            </Link>
            <Link className="button button-secondary" to="/how-it-works">
              Explore How It Works <ChevronRight size={16} />
            </Link>
          </div>
          <div className="hero-proof">
            <span><CheckCircle2 size={16} /> Official sources</span>
            <span><ShieldCheck size={16} /> Claim-level checks</span>
            <span><Zap size={16} /> Fast results</span>
          </div>
        </div>
        <PreviewCard />
      </section>

      <section className="section">
        <SectionLabel>WHY THIS MATTERS</SectionLabel>
        <h2>Government information deserves<br /><span>more than a confident answer.</span></h2>
        <div className="three-grid">
          <InfoCard icon={<CircleAlert />} title="AI Hallucinations" text="AI systems can generate convincing information that is not supported by official sources." tone="red" />
          <InfoCard icon={<BookOpen />} title="Evidence Grounding" text="Government claims are checked against retrieved official documents in real time." tone="blue" />
          <InfoCard icon={<ShieldCheck />} title="Transparent Verification" text="Users can see the evidence and source behind every verification result." tone="green" />
        </div>
      </section>

      <section className="section process-section">
        <div>
          <SectionLabel>THE VERIFICATION LOOP</SectionLabel>
          <h2>From question to<br /><span>evidence-backed answer.</span></h2>
          <p className="section-intro">Every answer is broken into individual claims and checked against a trusted evidence base.</p>
        </div>
        <div className="process-grid">
          {[
            ['01', 'Ask', 'User enters a government-related question.'],
            ['02', 'Generate', 'The system generates an AI answer.'],
            ['03', 'Extract', 'Individual factual claims are extracted.'],
            ['04', 'Retrieve', 'Relevant official evidence is retrieved.'],
            ['05', 'Verify', 'Claims are compared against evidence with NLI.'],
            ['06', 'Report', 'A transparent verification report is generated.'],
          ].map(([n, t, d]) => (
            <div className="process-step" key={n}>
              <b>{n}</b><h3>{t}</h3><p>{d}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section feature-section">
        <SectionLabel>BUILT FOR TRUST</SectionLabel>
        <h2>Make every answer<br /><span>accountable.</span></h2>
        <div className="feature-grid">
          {['Evidence-based AI', 'Claim-level verification', 'Official source citations', 'BGE-M3 vector search', 'DeBERTa NLI inference', 'Transparent reasoning'].map((x, i) => (
            <div className="feature-pill" key={x}>
              <span>0{i + 1}</span>{x}<Check size={15} />
            </div>
          ))}
        </div>
      </section>
      <CTA />
    </main>
  );
}

function PreviewCard() {
  return (
    <div className="preview-wrap">
      <div className="preview-card">
        <div className="preview-top">
          <span><i /> Verification Preview</span>
          <MoreHorizontal size={16} />
        </div>
        <div className="preview-title">
          <span>AI Generated Answer</span>
          <span className="mini-chip">4 claims</span>
        </div>
        <p className="preview-answer">Applicants must be between 18 and 60 years old. Annual family income must be below ₹2 lakh…</p>
        <div className="preview-claims">
          {heroPreviewClaims.map(c => (
            <div key={c.id}>
              <span className={`mini-dot ${c.status.toLowerCase()}`} />
              <span>{c.text}</span>
              <StatusBadge status={c.status} />
            </div>
          ))}
        </div>
        <div className="preview-source">
          <FileText size={14} />
          <span>Agricultural_Support_Scheme_Guidelines.pdf</span>
          <span>Page 2</span>
        </div>
      </div>
      <div className="preview-orbit orbit-one" />
      <div className="preview-orbit orbit-two" />
    </div>
  );
}

function InfoCard({ icon, title, text, tone }: {
  icon: React.ReactNode; title: string; text: string; tone: string;
}) {
  return (
    <div className="info-card">
      <div className={`info-icon ${tone}`}>{icon}</div>
      <h3>{title}</h3>
      <p>{text}</p>
      <ArrowRight size={16} />
    </div>
  );
}

function CTA() {
  return (
    <section className="cta">
      <div className="cta-icon"><ShieldCheck /></div>
      <div>
        <SectionLabel>READY WHEN YOU ARE</SectionLabel>
        <h2>Ready to verify an AI answer?</h2>
        <p>See exactly what is supported, what needs context, and what has no official evidence.</p>
      </div>
      <Link className="button button-light" to="/verify">
        Start Verification <ArrowRight size={16} />
      </Link>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Verify Page
// ─────────────────────────────────────────────────────────────────────────────

function Verify() {
  const navigate = useNavigate();
  const { setVerificationRequest } = useApp();
  const [tab, setTab] = useState<'question' | 'answer'>('question');
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const [collection, setCollection] = useState('All Government Documents');
  const [file, setFile] = useState<File | null>(null);
  const [advanced, setAdvanced] = useState(false);

  const submit = () => {
    if (!text.trim()) {
      setError('Please enter a question or answer before continuing.');
      return;
    }
    setError('');
    setVerificationRequest({ question: text, answer: tab === 'answer' ? text : undefined, tab, collection });
    navigate('/verification');
  };

  return (
    <main className="app-page">
      <PageHeader
        eyebrow="GOV-16 · VERIFICATION WORKSPACE"
        title="Verify Government Information"
        description="Enter a question or AI-generated answer and verify its claims against official government evidence."
      />

      <div className="verify-layout">
        <div className="workspace-card">
          <div className="tabs">
            <button className={tab === 'question' ? 'selected' : ''} onClick={() => setTab('question')}>
              <MessageSquareText size={16} /> Ask a Question
            </button>
            <button className={tab === 'answer' ? 'selected' : ''} onClick={() => setTab('answer')}>
              <ClipboardCheck size={16} /> Verify an Existing Answer
            </button>
          </div>

          <div className="form-content">
            <label>
              {tab === 'question' ? 'Your question' : 'AI-generated answer'}
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                placeholder={
                  tab === 'question'
                    ? 'Ask a government-related question… (e.g. What are the eligibility rules for agricultural support?)'
                    : 'Paste an AI-generated answer here…'
                }
              />
            </label>

            {tab === 'question' && (
              <label>
                Document collection
                <select value={collection} onChange={e => setCollection(e.target.value)}>
                  {['All Government Documents', 'Agriculture', 'Education', 'Healthcare', 'Employment', 'Housing'].map(x => (
                    <option key={x}>{x}</option>
                  ))}
                </select>
              </label>
            )}

            {tab === 'answer' && <FileUpload file={file} setFile={setFile} />}

            {error && (
              <div className="form-error"><CircleAlert size={15} />{error}</div>
            )}

            <Button onClick={submit} className="full-button">
              {tab === 'question' ? 'Generate & Verify' : 'Verify Claims'} <ArrowRight size={16} />
            </Button>

            {tab === 'question' && (
              <div className="examples">
                <strong>Example questions</strong>
                {[
                  'What are the eligibility requirements for the agricultural support scheme?',
                  'What is the maximum income ceiling for farmer financial assistance?',
                  'What documents are required to apply for agricultural support?',
                ].map(x => (
                  <button key={x} onClick={() => setText(x)}>{x}<ChevronRight size={14} /></button>
                ))}
              </div>
            )}
          </div>
        </div>

        <aside className="settings-card">
          <button className="settings-heading" onClick={() => setAdvanced(!advanced)}>
            <span><Settings2 size={17} /> Verification Pipeline</span>
            <ChevronDown className={advanced ? 'rotate' : ''} size={16} />
          </button>
          {advanced ? (
            <div className="settings-fields">
              <label>Embedding Model: <b>BAAI/bge-m3</b></label>
              <label>NLI Model: <b>DeBERTa-v3-base</b></label>
              <label>Vector Index: <b>FAISS Inner Product</b></label>
            </div>
          ) : (
            <div className="settings-collapsed">
              <span>Embedding <b>BGE-M3</b></span>
              <span>NLI Verifier <b>DeBERTa-v3</b></span>
              <span>Vector Store <b>FAISS</b></span>
            </div>
          )}

          <div className="upload-sidebar">
            <div className="upload-icon"><Upload size={20} /></div>
            <strong>Upload supporting documents</strong>
            <p>Add official government PDFs to expand your evidence base.</p>
            <label className="button button-secondary button-sm">
              Browse files
              <input type="file" accept="application/pdf" hidden onChange={e => setFile(e.target.files?.[0] ?? null)} />
            </label>
            {file && <small>{file.name} <X size={13} /></small>}
          </div>
        </aside>
      </div>

      <div className="verify-ready">
        <EmptyState title="Ready to verify" text="Enter a government question or paste an AI-generated answer to execute real-time evidence retrieval and NLI verification." />
      </div>
    </main>
  );
}

function FileUpload({ file, setFile }: { file: File | null; setFile: (f: File | null) => void }) {
  return (
    <div className="file-upload">
      <Upload size={22} />
      <strong>Upload supporting documents</strong>
      <span>
        Drag and drop PDF files here or{' '}
        <label>browse<input type="file" accept="application/pdf" hidden onChange={e => setFile(e.target.files?.[0] ?? null)} /></label>
      </span>
      <small>PDF only · Real PyMuPDF + FAISS indexing</small>
      {file && (
        <div className="file-row">
          <FileText size={15} />{file.name}
          <button onClick={() => setFile(null)}><X size={14} /></button>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Verification (Live processing)
// ─────────────────────────────────────────────────────────────────────────────

function Verification() {
  const navigate = useNavigate();
  const { verificationRequest, addReport } = useApp();
  const [step, setStep] = useState('Initialising…');
  const [pct, setPct] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const ran = useRef(false);

  const progressSteps = [
    'Searching official document index…',
    'Retrieving relevant evidence passages…',
    'Generating evidence-grounded answer…',
    'Extracting factual claims…',
    'Running DeBERTa NLI verification…',
    'Compiling verification report…',
  ];
  const [animStep, setAnimStep] = useState(0);

  useEffect(() => {
    if (ran.current) return;
    ran.current = true;

    if (!verificationRequest) {
      navigate('/verify');
      return;
    }

    const run = async () => {
      setErrorMsg('');
      setStep('Initialising verification pipeline…');
      setPct(10);
      try {
        const onProgress = (s: string, p: number) => {
          setStep(s);
          setPct(Math.round(p));
        };

        let report: VerificationReport;
        if (verificationRequest.tab === 'question') {
          report = await verifyQuestion(verificationRequest.question, verificationRequest.collection, onProgress);
        } else {
          report = await verifyAnswer(verificationRequest.answer ?? verificationRequest.question, onProgress);
        }

        addReport(report);
        navigate('/results');
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Verification failed. Please ensure the backend is running.';
        setErrorMsg(msg);
      }
    };

    run();
  }, [verificationRequest, addReport, navigate]);

  const handleRetry = () => {
    setErrorMsg('');
    ran.current = false;
    if (!verificationRequest) {
      navigate('/verify');
      return;
    }
    const onProgress = (s: string, p: number) => {
      setStep(s);
      setPct(Math.round(p));
    };
    if (verificationRequest.tab === 'question') {
      verifyQuestion(verificationRequest.question, verificationRequest.collection, onProgress)
        .then(r => { addReport(r); navigate('/results'); })
        .catch(err => setErrorMsg(err?.message ?? 'Verification failed.'));
    } else {
      verifyAnswer(verificationRequest.answer ?? verificationRequest.question, onProgress)
        .then(r => { addReport(r); navigate('/results'); })
        .catch(err => setErrorMsg(err?.message ?? 'Verification failed.'));
    }
  };

  useEffect(() => {
    const t = setInterval(() => setAnimStep(s => (s + 1) % progressSteps.length), 900);
    return () => clearInterval(t);
  }, [progressSteps.length]);

  if (errorMsg) {
    return (
      <main className="processing-page">
        <div className="processing-head">
          <div className="eyebrow" style={{ color: 'var(--red)' }}><CircleAlert size={14} /> VERIFICATION ERROR</div>
          <h1>Unable to complete verification</h1>
          <p style={{ color: 'var(--red)', maxWidth: 600, margin: '12px auto' }}>{errorMsg}</p>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 32, gap: 12 }}>
          <Button onClick={handleRetry} variant="primary"><Sparkles size={15} /> Try Verification Now</Button>
          <Link className="button button-secondary" to="/verify"><ChevronRight style={{ transform: 'rotate(180deg)' }} size={15} /> Go Back to Workspace</Link>
        </div>
      </main>
    );
  }

  return (
    <main className="processing-page">
      <div className="processing-head">
        <div className="eyebrow"><span className="pulse" /> REAL-TIME AI PIPELINE</div>
        <h1>Analyzing Government Information</h1>
        <p>Executing PyMuPDF chunks retrieval, BGE-M3 vector search, and DeBERTa NLI inference.</p>
      </div>

      <div className="processing-layout">
        <div className="progress-card">
          {progressSteps.map((s, i) => {
            const isDone = pct > (i / progressSteps.length) * 100;
            const isCurrent = !isDone && i === animStep;
            return (
              <div className={`progress-step ${isDone ? 'done' : isCurrent ? 'current' : ''}`} key={s}>
                <span className="progress-dot">
                  {isDone ? <Check size={14} /> : isCurrent ? <Loader2 size={14} className="spin" /> : i + 1}
                </span>
                <div>
                  <strong>{s}</strong>
                  <span>{isDone ? 'Completed' : isCurrent ? 'In progress' : 'Pending'}</span>
                </div>
                <span className="progress-state">{isDone ? 'Completed' : isCurrent ? 'In progress' : 'Pending'}</span>
              </div>
            );
          })}

          <div style={{ margin: '16px 4px 4px', background: '#eef2f6', borderRadius: 6, height: 5, overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${pct}%`, background: 'var(--blue)', borderRadius: 6, transition: 'width .4s ease' }} />
          </div>
          <div style={{ textAlign: 'right', fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>{pct}% complete</div>
        </div>

        <div className="live-card">
          <div className="live-icon"><Sparkles size={26} /></div>
          <span className="live-label">BACKEND PIPELINE ACTIVE</span>
          <h3>{step}</h3>
          <p>GovVerify AI is verifying claims against indexed government evidence passages.</p>
          <div className="activity-bars"><i /><i /><i /><i /><i /></div>
        </div>
      </div>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Results Page
// ─────────────────────────────────────────────────────────────────────────────

function Results() {
  const { currentReport, reports } = useApp();
  const result = currentReport || (reports.length > 0 ? reports[0] : null);

  if (!result) {
    return (
      <main className="app-page">
        <PageHeader
          eyebrow="VERIFICATION REPORT"
          title="No Report Available"
          description="Run a verification to analyze claims against official government sources."
        />
        <EmptyState
          icon={<ShieldCheck size={38} />}
          title="No verification results yet"
          text="Enter a question or AI answer in the workspace to generate a live verification report."
          action={
            <Link className="button button-primary" to="/verify">
              Start Verification <ArrowRight size={14} />
            </Link>
          }
        />
      </main>
    );
  }

  const supported = result.claims.filter(c => c.status === 'SUPPORTED').length;
  const uncertain = result.claims.filter(c => c.status === 'UNCERTAIN').length;
  const unsupported = result.claims.filter(c => c.status === 'UNSUPPORTED').length;
  const contradicted = result.claims.filter(c => c.status === 'CONTRADICTED').length;
  const copy = () => navigator.clipboard?.writeText(result.answer);

  const uniqueDocs = [...new Set(result.claims.map(c => c.evidence?.document).filter(Boolean))] as string[];

  return (
    <main className="app-page">
      <PageHeader
        eyebrow={`VERIFICATION COMPLETE · ${result.createdAt ?? 'JUST NOW'}`}
        title="Verification Report"
        description={result.question}
        action={<StatusBadge status={result.status} />}
      />
      <div className="report-overview">
        <div className="report-metrics">
          <Metric label="Total claims" value={result.claims.length} tone="blue" icon={<ClipboardCheck />} />
          <Metric label="Supported" value={supported} tone="green" icon={<CheckCircle2 />} />
          <Metric label="Uncertain" value={uncertain} tone="amber" icon={<CircleAlert />} />
          <Metric label="Unsupported" value={unsupported} tone="red" icon={<CircleAlert />} />
          {contradicted > 0 && <Metric label="Contradicted" value={contradicted} tone="red" icon={<CircleAlert />} />}
        </div>
        <div className="coverage-card">
          <ScoreRing score={result.coverage} />
          <div>
            <span>Overall verification</span>
            <h3>{result.status}</h3>
            <p>
              {result.coverage >= 75
                ? 'Most claims are supported by retrieved official evidence.'
                : result.coverage >= 40
                ? 'Some claims lack sufficient supporting evidence in indexed documents.'
                : 'Most claims are not supported by retrieved official evidence.'}
            </p>
          </div>
        </div>
      </div>

      <div className="report-grid">
        <div>
          <section className="panel answer-panel">
            <div className="panel-heading">
              <div><SectionLabel>INPUT</SectionLabel><h2>AI Generated Answer</h2></div>
              <Button variant="secondary" onClick={copy}><Copy size={14} /> Copy Answer</Button>
            </div>
            <p>{result.answer}</p>
          </section>
          <section className="claims-section">
            <div className="panel-heading">
              <div><SectionLabel>CLAIM ANALYSIS</SectionLabel><h2>Claim-by-claim verification</h2></div>
              <span className="muted">{result.claims.length} claims analyzed</span>
            </div>
            {result.claims.map(c => <ClaimCard key={c.id} claim={c} />)}
          </section>
        </div>

        <aside>
          <div className="panel summary-panel">
            <SectionLabel>VERIFICATION SUMMARY</SectionLabel>
            <div className="summary-chart">
              <div className="donut">
                <span>{result.claims.length}<small>claims</small></span>
              </div>
              <div>
                <span><i className="dot supported" />{supported} Supported</span>
                <span><i className="dot uncertain" />{uncertain} Uncertain</span>
                <span><i className="dot unsupported" />{unsupported} Unsupported</span>
                {contradicted > 0 && <span><i className="dot contradicted" />{contradicted} Contradicted</span>}
              </div>
            </div>
            <div className="summary-list">
              <span>Evidence coverage <b>{result.coverage}%</b></span>
              <span>Generated <b>{result.createdAt ?? 'Just now'}</b></span>
              <span>Source documents <b>{uniqueDocs.length || 0}</b></span>
            </div>
          </div>

          <div className="panel sources-panel">
            <div className="panel-heading">
              <h3>Evidence sources</h3>
              <BookOpen size={17} />
            </div>
            {uniqueDocs.length > 0 ? (
              uniqueDocs.map(doc => (
                <div className="source-card" key={doc}>
                  <FileText size={18} />
                  <div>
                    <strong>{doc}</strong>
                    <span>{result.claims.filter(c => c.evidence?.document === doc).length} claims referenced</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="source-card">
                <FileText size={18} />
                <div>
                  <strong>No matching documents</strong>
                  <span>No direct evidence found in current collection</span>
                </div>
              </div>
            )}
          </div>

          <div className="report-actions">
            <Button onClick={() => window.print()}><Download size={15} /> Download Report</Button>
            <Button variant="secondary" onClick={copy}><Copy size={15} /> Copy Verification</Button>
            <Link className="button button-secondary" to="/verify">
              Verify Another Answer <ArrowRight size={15} />
            </Link>
          </div>
        </aside>
      </div>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Claim Detail Page
// ─────────────────────────────────────────────────────────────────────────────

function ClaimDetail() {
  const { claimId } = useParams();
  const { currentReport, reports } = useApp();
  const navigate = useNavigate();

  const result = currentReport || (reports.length > 0 ? reports[0] : null);
  const claim = result?.claims.find(c => c.id === claimId);

  if (!claim) {
    return (
      <main className="app-page">
        <button className="back-link" onClick={() => navigate('/results')}>
          <ChevronRight size={15} className="back-arrow" /> Back to report
        </button>
        <EmptyState
          icon={<CircleAlert size={38} />}
          title="Claim not found"
          text="The requested claim is not available in the active verification report."
          action={
            <Button variant="secondary" onClick={() => navigate('/results')}>
              View Report
            </Button>
          }
        />
      </main>
    );
  }

  return (
    <main className="app-page">
      <button className="back-link" onClick={() => navigate('/results')}>
        <ChevronRight size={15} className="back-arrow" /> Back to report
      </button>
      <PageHeader
        eyebrow={`CLAIM ${String(claim.number).padStart(2, '0')} · VERIFICATION DETAIL`}
        title="Claim Verification"
        action={<StatusBadge status={claim.status} />}
      />
      <div className="claim-detail-grid">
        <div className="panel claim-main">
          <SectionLabel>CLAIM</SectionLabel>
          <h2>{claim.text}</h2>
          <ConfidenceBar value={claim.confidence} />
          <div className="detail-block">
            <SectionLabel>VERIFICATION REASON</SectionLabel>
            <p>{claim.reason}</p>
            {claim.numeric_conflict && (
              <div style={{ marginTop: '10px', padding: '10px 14px', background: 'rgba(239, 68, 68, 0.1)', borderLeft: '4px solid #ef4444', borderRadius: '4px', fontSize: '14px', color: '#b91c1c' }}>
                <strong>Contradiction Detected:</strong> {claim.numeric_conflict}
              </div>
            )}
            {claim.nli_label && (
              <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--muted)' }}>
                NLI Decision: <strong>{claim.nli_label}</strong> {claim.nli_probability ? `(${Math.round(claim.nli_probability * 100)}% confidence)` : ''}
              </div>
            )}
          </div>
          <div className="detail-block">
            <SectionLabel>VERIFICATION METHOD</SectionLabel>
            <div className="method-list">
              <span><Check size={14} /> BGE-M3 Semantic Similarity</span>
              <span><Check size={14} /> FAISS Vector Retrieval</span>
              <span><Check size={14} /> DeBERTa Natural Language Inference</span>
              <span><Check size={14} /> Deterministic Numeric & Rule Checker</span>
            </div>
          </div>
          <div className="detail-actions">
            <Button variant="secondary" onClick={() => navigate('/results')}>
              <ChevronRight className="back-arrow" size={15} /> Back to Report
            </Button>
            {claim.evidence && (
              <Button onClick={() => navigate('/documents')}>
                <ExternalLink size={15} /> View Documents
              </Button>
            )}
          </div>
        </div>

        <div className="panel evidence-card">
          <div className="evidence-heading">
            <SectionLabel>OFFICIAL SUPPORTING EVIDENCE</SectionLabel>
            <span className="evidence-check"><Check size={14} /></span>
          </div>
          {claim.evidence ? (
            <>
              <dl>
                <div><dt>Document</dt><dd>{claim.evidence.document}</dd></div>
                <div><dt>Page</dt><dd>{claim.evidence.page}</dd></div>
                <div><dt>Section</dt><dd>{claim.evidence.section}</dd></div>
              </dl>
              <div className="highlighted-evidence">"{claim.evidence.text}"</div>
              <p className="evidence-footnote">Extracted verbatim from the official government document index.</p>
            </>
          ) : (
            <div className="no-evidence">
              <CircleAlert size={20} />
              <strong>No supporting evidence found</strong>
              <span>This claim is unsupported by the documents currently indexed in the repository.</span>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Documents Library Page
// ─────────────────────────────────────────────────────────────────────────────

function Documents() {
  const { documents, addDocument, refreshData, backendConnected } = useApp();
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('All');
  const [showUpload, setShowUpload] = useState(false);

  const filtered = documents.filter(d =>
    d.name.toLowerCase().includes(query.toLowerCase()) &&
    (filter === 'All' || d.category === filter)
  );

  const handleUpload = async (file: File) => {
    const doc = await uploadDocument(file);
    addDocument(doc);
    await refreshData();
    setShowUpload(false);
  };

  return (
    <main className="app-page">
      <PageHeader
        eyebrow="EVIDENCE LIBRARY"
        title="Government Documents"
        description="Official government PDF documents indexed in FAISS for claim verification."
        action={<Button onClick={() => setShowUpload(true)}><Upload size={15} /> Upload Document</Button>}
      />
      <div className="toolbar">
        <div className="search-box">
          <Search size={16} />
          <input placeholder="Search documents…" value={query} onChange={e => setQuery(e.target.value)} />
        </div>
        <select value={filter} onChange={e => setFilter(e.target.value)}>
          <option>All</option>
          {['Agriculture', 'Education', 'Healthcare', 'Housing', 'Government'].map(x => <option key={x}>{x}</option>)}
        </select>
        <button className="filter-button"><Filter size={15} /> Filters</button>
      </div>

      <div className="documents-table panel">
        <div className="table-head">
          <span>Document</span><span>Category</span><span>Pages</span>
          <span>Claims</span><span>Status</span><span>Updated</span><span />
        </div>
        {filtered.length ? filtered.map(d => (
          <div className="table-row" key={d.id}>
            <div className="doc-name">
              <span className="file-icon"><FileText size={17} /></span>
              <div><strong>{d.name}</strong><small>{d.description}</small></div>
            </div>
            <span>{d.category}</span>
            <span>{d.pages} pages</span>
            <span>{d.claimsReferenced}</span>
            <span className="indexed"><i />{d.status}</span>
            <span>{d.updated}</span>
            <div className="row-actions">
              <Link to={`/documents/${d.id}`}><ExternalLink size={15} /></Link>
            </div>
          </div>
        )) : (
          <EmptyState
            icon={<FileText size={34} />}
            title={backendConnected === false ? "Unable to connect to backend" : "No documents found"}
            text={backendConnected === false ? `Please ensure the FastAPI backend is running on ${API_URL}.` : "Upload an official government PDF to begin indexing evidence."}
            action={backendConnected !== false ? <Button onClick={() => setShowUpload(true)}><Upload size={14} /> Upload PDF</Button> : undefined}
          />
        )}
      </div>

      {showUpload && <UploadModal onClose={() => setShowUpload(false)} onUpload={handleUpload} />}
    </main>
  );
}

function UploadModal({ onClose, onUpload }: { onClose: () => void; onUpload: (f: File) => Promise<void> }) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploadError, setUploadError] = useState('');

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setUploadError('');
    try {
      await onUpload(file);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed.';
      setUploadError(msg);
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <button className="modal-close" onClick={onClose}><X size={18} /></button>
        <div className="modal-icon"><Upload size={20} /></div>
        <h2>Upload a government document</h2>
        <p>Extracts pages with PyMuPDF and indexes BGE-M3 embeddings in FAISS.</p>
        {file ? (
          <div className="selected-file">
            <FileText size={20} />
            <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · PDF</span></div>
            <button onClick={() => setFile(null)}><X size={15} /></button>
          </div>
        ) : (
          <label className="modal-drop">
            <Upload size={24} />
            <strong>Choose a PDF file</strong>
            <span>or drag and drop it here</span>
            <input type="file" accept="application/pdf" hidden onChange={e => setFile(e.target.files?.[0] ?? null)} />
          </label>
        )}
        {uploadError && <div className="form-error" style={{ margin: '12px 0' }}><CircleAlert size={15} />{uploadError}</div>}
        <div className="modal-actions">
          <Button variant="secondary" onClick={onClose}>Cancel</Button>
          <Button onClick={submit} disabled={!file || busy}>
            {busy ? <><Loader2 size={15} className="spin" /> Indexing PDF in FAISS…</> : <>Upload & Index <ArrowRight size={15} /></>}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Document Detail Viewer Page
// ─────────────────────────────────────────────────────────────────────────────

function DocumentDetail() {
  const { id } = useParams();
  const [doc, setDoc] = useState<DocumentRecord | null>(null);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) {
      setLoading(true);
      getDocument(id)
        .then(data => {
          setDoc(data);
          setLoading(false);
        })
        .catch(() => {
          setDoc(null);
          setLoading(false);
        });
    }
  }, [id]);

  if (loading) return <main className="app-page"><LoadingState text="Loading official document…" /></main>;

  if (!doc) {
    return (
      <main className="app-page">
        <button className="back-link" onClick={() => history.back()}>
          <ChevronRight size={15} className="back-arrow" /> Documents
        </button>
        <EmptyState title="Document not found" text="The requested government document is not available." />
      </main>
    );
  }

  const text = doc.pagesText?.[page] || `Page ${page}\n[No text extracted on this page]`;

  return (
    <main className="app-page">
      <button className="back-link" onClick={() => history.back()}>
        <ChevronRight size={15} className="back-arrow" /> Documents
      </button>
      <PageHeader
        eyebrow="DOCUMENT VIEWER"
        title={doc.name}
        description={doc.description}
        action={<span className="indexed large"><i /> Indexed</span>}
      />
      <div className="doc-meta">
        <Metric label="Category" value={doc.category} icon={<BookOpen />} />
        <Metric label="Pages" value={doc.pages} icon={<FileText />} />
        <Metric label="Claims referenced" value={doc.claimsReferenced} icon={<ClipboardCheck />} />
        <Metric label="Last updated" value={doc.updated} icon={<Clock3 />} />
      </div>
      <div className="viewer-layout">
        <aside className="page-list panel">
          <div className="search-box">
            <Search size={15} />
            <input placeholder="Search document…" value={search} onChange={e => setSearch(e.target.value)} />
          </div>
          <strong>Pages in document</strong>
          {Array.from({ length: Math.min(doc.pages, 20) }, (_, i) => i + 1).map(n => (
            <button key={n} className={page === n ? 'selected' : ''} onClick={() => setPage(n)}>
              <span>Page {n}</span>
            </button>
          ))}
        </aside>
        <section className="document-preview panel">
          <div className="document-toolbar">
            <span><FileText size={15} /> {doc.name}</span>
            <div>
              <button><Download size={15} /></button>
            </div>
          </div>
          <div className="paper">
            <span className="paper-page">PAGE {page} OF {doc.pages}</span>
            {text.split('\n').map((line, i) => (
              <p
                className={
                  line.includes('18 and 60') || line.includes('Income') || line.includes('Rs 2,00,000')
                    ? 'highlight-line'
                    : i === 0
                    ? 'paper-title'
                    : ''
                }
                key={`${line}-${i}`}
              >
                {line || ' '}
              </p>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Reports History Page
// ─────────────────────────────────────────────────────────────────────────────

function Reports() {
  const { reports, deleteReport, setCurrentReport, backendConnected } = useApp();
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState('All');
  const navigate = useNavigate();

  const filtered = reports.filter(r =>
    r.question.toLowerCase().includes(query.toLowerCase()) &&
    (filter === 'All' || r.status === filter)
  );

  const openReport = (r: VerificationReport) => {
    setCurrentReport(r);
    navigate('/results');
  };

  return (
    <main className="app-page">
      <PageHeader
        eyebrow="VERIFICATION HISTORY"
        title="Verification Reports"
        description="Review, export, and revisit your previous claim-level verification results."
      />
      <div className="toolbar">
        <div className="search-box">
          <Search size={16} />
          <input placeholder="Search reports…" value={query} onChange={e => setQuery(e.target.value)} />
        </div>
        <select value={filter} onChange={e => setFilter(e.target.value)}>
          <option>All</option>
          <option>Verified</option>
          <option>Partially Verified</option>
          <option>Unsupported</option>
        </select>
      </div>

      <div className="reports-table panel">
        <div className="table-head">
          <span>Date</span><span>Question</span><span>Claims</span>
          <span>Supported</span><span>Unsupported</span><span>Coverage</span>
          <span>Status</span><span />
        </div>
        {filtered.map(r => (
          <div className="table-row report-row" key={r.id}>
            <span className="date-cell">{r.createdAt}</span>
            <div className="question-cell">
              <strong>{r.question}</strong>
              <small>Government claim verification</small>
            </div>
            <span>{r.claims.length}</span>
            <span className="green-text">{r.claims.filter(c => c.status === 'SUPPORTED').length}</span>
            <span className="red-text">{r.claims.filter(c => c.status === 'UNSUPPORTED').length}</span>
            <strong>{r.coverage}%</strong>
            <StatusBadge status={r.status} />
            <div className="row-actions">
              <button onClick={() => openReport(r)} title="View report"><ExternalLink size={15} /></button>
              <button onClick={() => deleteReport(r.id)} title="Delete report"><Trash2 size={15} /></button>
            </div>
          </div>
        ))}
        {!filtered.length && (
          <EmptyState
            icon={<BarChart3 size={34} />}
            title={backendConnected === false ? "Unable to connect to backend" : "No reports found"}
            text={backendConnected === false ? `Please ensure the FastAPI backend is running on ${API_URL}.` : "Your completed verifications will appear here."}
            action={<Link className="button button-primary button-sm" to="/verify">Start Verifying <ArrowRight size={14} /></Link>}
          />
        )}
      </div>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// How It Works Page
// ─────────────────────────────────────────────────────────────────────────────

function HowItWorks() {
  const workflow = [
    ['01. USER INPUT', 'User Question or Existing AI Answer', 'User submits a government scheme question or pastes AI-generated output for verification.'],
    ['02. DOCUMENT RETRIEVAL', 'Retrieve Official Evidence', 'Performs dense vector retrieval against indexed official PDF documents.'],
    ['03. ANSWER FORMULATION', 'Generate / Analyze Answer', 'Synthesizes or parses an evidence-grounded answer with traceable context.'],
    ['04. CLAIM EXTRACTION', 'Extract Atomic Claims', 'Decomposes complex paragraphs into atomic, testable factual propositions.'],
    ['05. ISOLATED RETRIEVAL', 'Claim-Specific Retrieval', 'Independently matches evidence chunks for each atomic claim.'],
    ['06. DENSE EMBEDDING', 'BGE-M3 Semantic Retrieval', 'BAAI/bge-m3 cosine similarity scoring with similarity threshold filtering.'],
    ['07. DETERMINISTIC AUDIT', 'Numeric / Condition Analysis', 'Detects direct number, percentage, and condition contradictions.'],
    ['08. NLI INFERENCE', 'DeBERTa NLI Verification', 'DeBERTa-v3 cross-encoder natural language inference (entailment vs contradiction).'],
    ['09. CLAIM RESOLUTION', 'Claim Status (4 States)', 'Resolves state: SUPPORTED, UNCERTAIN, CONTRADICTED, or UNSUPPORTED.'],
    ['10. CITATION MAPPING', 'Evidence + Source + Page', 'Links verbatim excerpt, document name, page number, and section.'],
    ['11. REPORT COMPILATION', 'Verification Report', 'Generates an immutable report with aggregate coverage and confidence metrics.'],
  ];

  return (
    <main>
      <section className="inner-hero">
        <div className="eyebrow">GOV-16 · VERIFICATION PIPELINE</div>
        <h1>How GovVerify AI Works</h1>
        <p>From user question to evidence-backed claim verification — an end-to-end transparent architecture.</p>
      </section>
      <section className="section workflow-section">
        <div className="workflow-line" />
        {workflow.map(([stepNum, title, text], i) => (
          <div className="workflow-node" key={title}>
            <span>{stepNum}</span>
            <div>
              <strong style={{ fontSize: '12px', color: 'var(--navy)' }}>{title}</strong>
              <p style={{ marginTop: '4px', fontSize: '11px' }}>{text}</p>
            </div>
            {i < workflow.length - 1 && <ArrowRight size={16} />}
          </div>
        ))}
      </section>
      <section className="section tech-section">
        <SectionLabel>THE TECHNOLOGY STACK</SectionLabel>
        <h2>Built on a modern<br /><span>verification pipeline.</span></h2>
        <div className="tech-grid">
          {[
            ['PyMuPDF', 'Page-aware PDF text and structure extraction'],
            ['BGE-M3', 'State-of-the-art multilingual vector embeddings'],
            ['FAISS', 'High-speed vector similarity index'],
            ['DeBERTa-v3 NLI', 'Cross-encoder natural language inference model'],
            ['FastAPI', 'High-performance Python backend server'],
            ['React + TypeScript', 'Evidence-grounded web application'],
          ].map(([a, b]) => (
            <div className="tech-card" key={a}>
              <div>{a.slice(0, 1)}</div>
              <strong>{a}</strong>
              <span>{b}</span>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// About Page
// ─────────────────────────────────────────────────────────────────────────────

function About() {
  return (
    <main>
      <section className="inner-hero about-hero">
        <div className="eyebrow">ABOUT THE PROJECT · GOV-16</div>
        <h1>About GovVerify AI</h1>
        <p style={{ marginTop: '14px', fontSize: '15px', lineHeight: '1.7', color: 'var(--muted)' }}>
          GovVerify AI is an evidence-grounded AI verification platform designed to identify supported, uncertain, contradicted, and unsupported claims in government-related information.
        </p>
      </section>

      <section className="section about-grid">
        <div>
          <SectionLabel>PROJECT SPECIFICATION</SectionLabel>
          <h2>Confidence is not<br /><span>the same as truth.</span></h2>
          <p>
            Generative AI can produce clear, persuasive answers even when those answers are not supported by official sources. In government contexts, that gap can create confusion and erode public trust. GovVerify AI establishes rigorous evidence grounding for public information.
          </p>
          <div style={{ marginTop: '24px', padding: '16px 20px', background: '#f8fafc', border: '1px solid var(--line)', borderRadius: '8px' }}>
            <div style={{ fontSize: '11px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.6px', fontWeight: 700 }}>Official Product</div>
            <div style={{ fontSize: '17px', fontWeight: 800, color: 'var(--navy)', marginTop: '3px' }}>GovVerify AI</div>
            <div style={{ fontSize: '11px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.6px', fontWeight: 700, marginTop: '12px' }}>Problem Statement</div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--blue)', marginTop: '3px' }}>GOV-16 — Unsupported Government Claim Detector</div>
          </div>
        </div>
        <div className="about-cards">
          <div className="about-card">
            <CircleAlert />
            <h3>The Problem</h3>
            <p>Detect unsupported, ungrounded, or contradicted AI claims before they are treated as authoritative public information.</p>
          </div>
          <div className="about-card">
            <ShieldCheck />
            <h3>The Solution</h3>
            <p>Ground every claim in retrieved official government documents with transparent citations, page numbers, and DeBERTa NLI proof.</p>
          </div>
        </div>
      </section>

      <section className="section tech-section">
        <SectionLabel>CORE TECHNOLOGIES</SectionLabel>
        <h2>Implemented Architecture</h2>
        <div className="tech-grid">
          {[
            ['BGE-M3', 'Dense semantic vector embeddings for cross-lingual government document search'],
            ['DeBERTa-v3 NLI', 'Cross-encoder natural language inference verifier for entailment / contradiction'],
            ['FAISS', 'Facebook AI Similarity Search index for sub-millisecond evidence retrieval'],
            ['PyMuPDF', 'High-fidelity PDF parser preserving page numbers and document structure'],
            ['FastAPI', 'Asynchronous REST backend with full verification and document lifecycle endpoints'],
            ['React + TypeScript', 'Modern responsive interface with dynamic state verification and dark mode'],
          ].map(([a, b]) => (
            <div className="tech-card" key={a}>
              <div>{a.slice(0, 1)}</div>
              <strong>{a}</strong>
              <span>{b}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="section objectives">
        <SectionLabel>PROJECT OBJECTIVES</SectionLabel>
        <div className="objective-list">
          {[
            'Detect unsupported and contradicted AI claims',
            'Ground government information in official evidence passages',
            'Eliminate hallucination risk with strict similarity thresholding',
            'Deliver deterministic numeric and condition conflict analysis',
            'Provide traceable official source citations with exact page numbers',
          ].map((x, i) => (
            <div key={x}><b>0{i + 1}</b><span>{x}</span><Check size={17} /></div>
          ))}
        </div>
      </section>

      <section className="disclaimer">
        <Info size={20} />
        <p><strong>A note on interpretation</strong> Verification results indicate whether a claim is supported by the retrieved evidence in the repository. An unsupported result does not necessarily mean the claim is factually false in the real world, but rather that no authoritative evidence was found in the indexed repository.</p>
      </section>
    </main>
  );
}

function NotFound() {
  return (
    <main className="not-found">
      <div className="not-found-number">404</div>
      <h1>Page not found</h1>
      <p>The page you are looking for may have moved or no longer exists.</p>
      <Link className="button button-primary" to="/">Return Home <ArrowRight size={15} /></Link>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// App Root Component
// ─────────────────────────────────────────────────────────────────────────────

function App() {
  const { backendConnected } = useApp();

  return (
    <BrowserRouter>
      <PageTitleManager />
      {backendConnected === false && (
        <div className="backend-warning-banner">
          <CircleAlert size={16} /> Unable to connect to GovVerify AI backend. Please ensure the backend server is reachable on {API_URL}.
        </div>
      )}
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/verify" element={<Verify />} />
        <Route path="/verification" element={<Verification />} />
        <Route path="/results" element={<Results />} />
        <Route path="/claims/:claimId" element={<ClaimDetail />} />
        <Route path="/documents" element={<Documents />} />
        <Route path="/documents/:id" element={<DocumentDetail />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/how-it-works" element={<HowItWorks />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}

export default App;
