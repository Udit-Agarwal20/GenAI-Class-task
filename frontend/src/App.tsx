import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Menu,
  MessageSquare,
  BookOpen,
  Plus,
  FileText,
  Cpu,
} from 'lucide-react';
import type {
  Conversation,
  DocumentMeta,
  HealthResponse,
  SourceItem,
  PerfMetrics,
  DepthMode,
} from './lib/api';
import {
  fetchHealth,
  fetchDocuments,
  streamChatMessage,
  fetchAudit,
  uploadPdfDocument,
} from './lib/api';
import { Sidebar } from './components/Sidebar';
import { ChatPage } from './pages/ChatPage';
import { LibraryPage } from './pages/LibraryPage';
import { PdfViewer } from './components/PdfViewer';

const STORAGE_KEY = 'proofline_conversations_v1';
const ACTIVE_CONV_KEY = 'proofline_active_conv_id';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<'chat' | 'library'>('chat');
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [activeConvId, setActiveConvId] = useState<string | null>(() => {
    try {
      return localStorage.getItem(ACTIVE_CONV_KEY) || null;
    } catch {
      return null;
    }
  });

  const [inputPrompt, setInputPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [auditLoadingId, setAuditLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [depthMode, setDepthMode] = useState<DepthMode>('standard');
  const [selectedModel, setSelectedModel] = useState<string>('gemma4:e2b');

  const abortControllerRef = useRef<AbortController | null>(null);

  const [documents, setDocuments] = useState<DocumentMeta[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [activePdf, setActivePdf] = useState<{ filename: string; page: number | string } | null>(null);
  const [mobileOpen, setMobileOpen] = useState(false);

  // Save conversations to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
      console.warn('Failed to save conversations to localStorage', e);
    }
  }, [conversations]);

  useEffect(() => {
    try {
      if (activeConvId) {
        localStorage.setItem(ACTIVE_CONV_KEY, activeConvId);
      } else {
        localStorage.removeItem(ACTIVE_CONV_KEY);
      }
    } catch (e) {
      console.warn('Failed to save active conversation ID', e);
    }
  }, [activeConvId]);

  // Load health & documents on mount
  const loadInitialData = useCallback(async () => {
    try {
      const healthData = await fetchHealth();
      setHealth(healthData);
    } catch (err: any) {
      console.warn('Health check warning:', err.message);
    }

    try {
      setLoadingDocs(true);
      const docsData = await fetchDocuments();
      setDocuments(docsData.documents);
    } catch (err: any) {
      setError(`Failed to load document catalog: ${err.message}`);
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Current active conversation
  const currentConversation = conversations.find((c) => c.id === activeConvId) || null;

  const handleNewChat = () => {
    if (loading && abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setActiveConvId(null);
    setInputPrompt('');
    setError(null);
    setStreamingMessageId(null);
    setCurrentTab('chat');
  };

  const handleSelectConversation = (id: string) => {
    if (loading && abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setActiveConvId(id);
    setError(null);
    setStreamingMessageId(null);
  };

  const handleDeleteConversation = (id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeConvId === id) {
      setActiveConvId(null);
    }
  };

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setLoading(false);
    setStreamingMessageId(null);
  };

  const handleSendMessage = async (overridePrompt?: string) => {
    const question = (overridePrompt ?? inputPrompt).trim();
    if (!question || loading) return;

    setError(null);
    setLoading(true);
    setInputPrompt('');

    // Determine target conversation or create a new one
    let convId = activeConvId;
    const now = Date.now();

    if (!convId || !currentConversation) {
      const newTitle = question.length > 36 ? `${question.slice(0, 36)}...` : question;
      const newConv: Conversation = {
        id: `conv_${now}_${Math.random().toString(36).substring(2, 7)}`,
        title: newTitle,
        createdAt: now,
        messages: [],
      };
      convId = newConv.id;
      setConversations((prev) => [newConv, ...prev]);
      setActiveConvId(convId);
    }

    const assistantMsgId = `msg_a_${now + 1}`;
    setStreamingMessageId(assistantMsgId);

    // Optimistically add user message and placeholder assistant message
    const userMessage = {
      id: `msg_u_${now}`,
      role: 'user' as const,
      content: question,
      timestamp: now,
    };

    const initialAssistantMsg = {
      id: assistantMsgId,
      role: 'assistant' as const,
      content: '',
      sources: [],
      timestamp: now + 1,
    };

    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? { ...c, messages: [...c.messages, userMessage, initialAssistantMsg] }
          : c
      )
    );

    // Prepare history (trimmed to last 2 turns)
    const historyPayload = (
      (currentConversation?.messages || []).slice(-4)
    ).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const controller = new AbortController();
    abortControllerRef.current = controller;

    await streamChatMessage(
      question,
      historyPayload,
      {
        onSources: (sources: SourceItem[]) => {
          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMsgId ? { ...m, sources } : m
                    ),
                  }
                : c
            )
          );
        },
        onToken: (token: string) => {
          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMsgId
                        ? { ...m, content: m.content + token }
                        : m
                    ),
                  }
                : c
            )
          );
        },
        onDone: (metrics?: PerfMetrics) => {
          setLoading(false);
          setStreamingMessageId(null);
          abortControllerRef.current = null;

          if (metrics) {
            setConversations((prev) =>
              prev.map((c) =>
                c.id === convId
                  ? {
                      ...c,
                      messages: c.messages.map((m) =>
                        m.id === assistantMsgId ? { ...m, metrics } : m
                      ),
                    }
                  : c
              )
            );
          }
        },
        onError: (errMessage: string) => {
          setLoading(false);
          setStreamingMessageId(null);
          abortControllerRef.current = null;
          setError(errMessage);
        },
      },
      depthMode,
      selectedModel,
      controller.signal
    );
  };

  const handleRunAudit = async (
    messageId: string,
    question: string,
    answer: string,
    sources: SourceItem[]
  ) => {
    if (!question || !answer || sources.length === 0) return;

    setAuditLoadingId(messageId);
    try {
      const auditResult = await fetchAudit(question, answer, sources);
      setConversations((prev) =>
        prev.map((c) => ({
          ...c,
          messages: c.messages.map((m) =>
            m.id === messageId ? { ...m, audit: auditResult } : m
          ),
        }))
      );
    } catch (err: any) {
      setError(err.message || 'Failed to complete evidence audit.');
    } finally {
      setAuditLoadingId(null);
    }
  };

  const handleUploadFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please choose a valid PDF (.pdf) file.');
      return;
    }

    setUploading(true);
    setError(null);
    setUploadStatus(null);

    try {
      const res = await uploadPdfDocument(file);
      setUploadStatus(
        `Indexed "${res.filename}" (${res.pages_processed} pages, ${res.chunks_indexed} chunks added to Qdrant).`
      );

      const docsData = await fetchDocuments();
      setDocuments(docsData.documents);

      const healthData = await fetchHealth();
      setHealth(healthData);

      setTimeout(() => {
        setUploadStatus(null);
      }, 6000);
    } catch (err: any) {
      setError(err.message || 'Failed to upload and index PDF.');
    } finally {
      setUploading(false);
    }
  };

  const handleViewSource = (filename: string, page: number | string) => {
    setActivePdf({ filename, page });
  };

  return (
    <div className="app-container">
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        onNewChat={handleNewChat}
        conversations={conversations}
        activeConvId={activeConvId}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        health={health}
        docCount={documents.length}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <main className="main-content">
        <header className="top-bar">
          <div className="top-bar-left">
            <button
              className="mobile-menu-btn"
              onClick={() => setMobileOpen(true)}
              aria-label="Open menu"
            >
              <Menu size={20} />
            </button>

            <div className="page-title-badge">
              {currentTab === 'chat' ? (
                <>
                  <MessageSquare size={16} color="var(--forest-700)" />
                  <span>
                    {currentConversation ? currentConversation.title : 'Ask Documents'}
                  </span>
                </>
              ) : (
                <>
                  <BookOpen size={16} color="var(--forest-700)" />
                  <span>Document Library</span>
                </>
              )}
            </div>
          </div>

          <div className="top-bar-actions">
            <div className="ghost-btn" style={{ padding: '4px 10px', gap: 6 }} title="Select active Ollama model">
              <Cpu size={14} color="var(--forest-600)" />
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  color: 'var(--text-main)',
                  cursor: 'pointer',
                  outline: 'none',
                  fontFamily: 'inherit',
                }}
              >
                <option value="gemma4:e2b">gemma4:e2b (Default)</option>
                <option value="gemma4:e4b-it-q4_K_M">gemma4:e4b-it-q4_K_M (Candidate)</option>
              </select>
            </div>

            {currentTab === 'chat' && currentConversation && (
              <button className="ghost-btn" onClick={handleNewChat} title="Start new conversation">
                <Plus size={14} />
                <span>New Chat</span>
              </button>
            )}

            <button
              className="ghost-btn"
              onClick={() => setCurrentTab(currentTab === 'chat' ? 'library' : 'chat')}
            >
              {currentTab === 'chat' ? (
                <>
                  <FileText size={14} />
                  <span>Library ({documents.length})</span>
                </>
              ) : (
                <>
                  <MessageSquare size={14} />
                  <span>Chat Interface</span>
                </>
              )}
            </button>
          </div>
        </header>

        {currentTab === 'chat' ? (
          <ChatPage
            conversation={currentConversation}
            inputPrompt={inputPrompt}
            setInputPrompt={setInputPrompt}
            onSubmit={handleSendMessage}
            onStopStreaming={handleStopStreaming}
            loading={loading}
            streamingMessageId={streamingMessageId}
            docCount={documents.length}
            onViewSource={handleViewSource}
            onUploadFile={handleUploadFile}
            onRunAudit={handleRunAudit}
            auditLoadingId={auditLoadingId}
            error={error}
            onDismissError={() => setError(null)}
            depthMode={depthMode}
            onDepthModeChange={setDepthMode}
          />
        ) : (
          <LibraryPage
            documents={documents}
            loadingDocs={loadingDocs}
            onUploadFile={handleUploadFile}
            onViewSource={handleViewSource}
            uploading={uploading}
            uploadStatus={uploadStatus}
            error={error}
            onDismissError={() => setError(null)}
          />
        )}
      </main>

      {activePdf && (
        <PdfViewer
          filename={activePdf.filename}
          initialPage={activePdf.page}
          onClose={() => setActivePdf(null)}
        />
      )}
    </div>
  );
};

export default App;
