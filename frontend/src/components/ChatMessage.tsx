import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Sparkles, ChevronDown, ChevronUp, FileText, Zap } from 'lucide-react';
import type { SourceItem, AuditDetails, PerfMetrics } from '../lib/api';
import { SourceCard } from './SourceCard';
import { EvidenceAudit } from './EvidenceAudit';

interface MessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceItem[];
  audit?: AuditDetails;
  metrics?: PerfMetrics;
  onViewSource: (filename: string, page: number | string) => void;
  onRunAudit?: () => void;
  auditLoading?: boolean;
  isStreaming?: boolean;
}

export const ChatMessage: React.FC<MessageProps> = ({
  role,
  content,
  sources = [],
  audit,
  metrics,
  onViewSource,
  onRunAudit,
  auditLoading = false,
  isStreaming = false,
}) => {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [highlightedSource, setHighlightedSource] = useState<string | null>(null);

  if (role === 'user') {
    return (
      <div className="message-row">
        <div className="message-user">
          <div className="message-user-bubble">{content}</div>
        </div>
      </div>
    );
  }

  // Pre-process content to transform [S1], [S2] into clickable markdown links
  const processedContent = content.replace(
    /\[S(\d+)\]/g,
    '[$&](#cite-S$1)'
  );

  const handleCitationClick = (label: string) => {
    setHighlightedSource(label);
    setSourcesOpen(true);

    setTimeout(() => {
      const el = document.getElementById(`source-${label}`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }, 100);

    setTimeout(() => {
      setHighlightedSource((curr) => (curr === label ? null : curr));
    }, 3000);
  };

  return (
    <div className="message-row">
      <div className="message-assistant">
        <div className="message-assistant-header" style={{ justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="assistant-avatar">
              <Sparkles size={13} />
            </div>
            <span>Proofline AI</span>
            {isStreaming && (
              <span
                style={{
                  fontSize: '0.72rem',
                  color: 'var(--forest-600)',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                <span className="status-dot" style={{ width: 6, height: 6 }} />
                streaming...
              </span>
            )}
          </div>

          {metrics && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
                backgroundColor: 'var(--bg-subtle)',
                padding: '2px 8px',
                borderRadius: 'var(--radius-pill)',
                border: '1px solid var(--border-subtle)',
              }}
              title={`Retrieval: ${metrics.retrieval_ms}ms | TTFT: ${metrics.ttft_ms}ms | Gen: ${metrics.generation_ms}ms`}
            >
              <Zap size={11} color="var(--forest-600)" />
              <span>TTFT: {(metrics.ttft_ms / 1000).toFixed(2)}s</span>
              <span>•</span>
              <span>{(metrics.total_ms / 1000).toFixed(1)}s total</span>
              {metrics.tokens_per_sec > 0 && (
                <>
                  <span>•</span>
                  <span>{metrics.tokens_per_sec} tok/s</span>
                </>
              )}
            </div>
          )}
        </div>

        <div className="message-assistant-body">
          <div className="markdown-content">
            {content ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ href, children }) => {
                    if (href && href.startsWith('#cite-S')) {
                      const label = href.replace('#cite-', '');
                      return (
                        <button
                          type="button"
                          className="citation-pill"
                          onClick={(e) => {
                            e.preventDefault();
                            handleCitationClick(label);
                          }}
                          title={`Jump to source ${label}`}
                        >
                          {children}
                        </button>
                      );
                    }
                    return (
                      <a href={href} target="_blank" rel="noopener noreferrer">
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {processedContent}
              </ReactMarkdown>
            ) : (
              <div style={{ color: 'var(--text-muted)', fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: 6 }}>
                <div className="loading-spinner" style={{ width: 14, height: 14 }} />
                <span>Generating response...</span>
              </div>
            )}
          </div>

          {sources && sources.length > 0 && (
            <div className="sources-section">
              <button
                type="button"
                className="sources-toggle-btn"
                onClick={() => setSourcesOpen(!sourcesOpen)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <FileText size={14} />
                  <span>Retrieved Evidence ({sources.length} sources)</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.75rem' }}>
                  <span>{sourcesOpen ? 'Collapse' : 'Expand'}</span>
                  {sourcesOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </div>
              </button>

              {sourcesOpen && (
                <div className="sources-cards-grid">
                  {sources.map((src) => (
                    <SourceCard
                      key={src.label}
                      source={src}
                      isHighlighted={highlightedSource === src.label}
                      onViewSource={onViewSource}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* User-triggered or completed Evidence Audit */}
          <EvidenceAudit
            audit={audit}
            onRunAudit={onRunAudit}
            auditLoading={auditLoading}
          />
        </div>
      </div>
    </div>
  );
};
