import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import type { VerificationReport, DocumentRecord } from '@/types';
import { getDocuments, getReports, checkBackendHealth } from '@/services/api';

export interface VerificationRequest {
  question: string;
  answer?: string;
  tab: 'question' | 'answer';
  collection: string;
}

interface AppContextType {
  // Verification flow
  verificationRequest: VerificationRequest | null;
  setVerificationRequest: (r: VerificationRequest | null) => void;
  currentReport: VerificationReport | null;
  setCurrentReport: (r: VerificationReport | null) => void;

  // History & Data
  reports: VerificationReport[];
  addReport: (r: VerificationReport) => void;
  deleteReport: (id: string) => void;
  documents: DocumentRecord[];
  addDocument: (d: DocumentRecord) => void;
  refreshData: () => Promise<void>;

  // System status
  backendConnected: boolean | null;
  darkMode: boolean;
  toggleDark: () => void;
  hasApiKey: boolean;
}

const AppContext = createContext<AppContextType>({} as AppContextType);

export function AppProvider({ children }: { children: ReactNode }) {
  const [verificationRequest, setVerificationRequest] = useState<VerificationRequest | null>(null);
  const [currentReport, setCurrentReport] = useState<VerificationReport | null>(null);
  const [reports, setReports] = useState<VerificationReport[]>([]);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);

  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('gv_dark') === 'true';
  });

  const hasApiKey = false;

  const refreshData = useCallback(async () => {
    try {
      const isHealthy = await checkBackendHealth();
      setBackendConnected(isHealthy);
      if (isHealthy) {
        const [docsData, reportsData] = await Promise.all([
          getDocuments().catch(() => []),
          getReports().catch(() => [])
        ]);
        setDocuments(docsData);
        setReports(reportsData);
        if (reportsData.length > 0 && !currentReport) {
          setCurrentReport(reportsData[0]);
        }
      }
    } catch {
      setBackendConnected(false);
    }
  }, [currentReport]);

  useEffect(() => {
    refreshData();
    const interval = setInterval(refreshData, 10000);
    return () => clearInterval(interval);
  }, [refreshData]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('gv_dark', String(darkMode));
  }, [darkMode]);

  const addReport = useCallback((r: VerificationReport) => {
    setReports(prev => [r, ...prev.filter(x => x.id !== r.id)]);
    setCurrentReport(r);
  }, []);

  const deleteReport = useCallback((id: string) => {
    setReports(prev => prev.filter(r => r.id !== id));
    setCurrentReport(prev => (prev?.id === id ? null : prev));
  }, []);

  const addDocument = useCallback((d: DocumentRecord) => {
    setDocuments(prev => [d, ...prev.filter(x => x.id !== d.id)]);
  }, []);

  const toggleDark = useCallback(() => setDarkMode(v => !v), []);

  return (
    <AppContext.Provider value={{
      verificationRequest, setVerificationRequest,
      currentReport, setCurrentReport,
      reports, addReport, deleteReport,
      documents, addDocument, refreshData,
      backendConnected,
      darkMode, toggleDark,
      hasApiKey,
    }}>
      {children}
    </AppContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export const useApp = () => useContext(AppContext);
