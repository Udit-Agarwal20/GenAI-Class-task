import React, { useRef, useEffect } from 'react';
import {
  HelpCircle,
  FileText,
  GitCompare,
  Search,
  AlertCircle
} from 'lucide-react';
import { ChatComposer } from '../components/ChatComposer';
import { ChatMessage } from '../components/ChatMessage';
import type { Conversation, SourceItem, DepthMode } from '../lib/api';

interface ChatPageProps {
  conversation: Conversation | null;
  inputPrompt: string;
  setInputPrompt: (val: string) => void;
  onSubmit: (prompt?: string) => void;
  onStopStreaming: () => void;
  loading: boolean;
  streamingMessageId: string | null;
  docCount: number;
  onViewSource: (filename: string, page: number | string) => void;
  onUploadFile: (file: File) => void;
  onRunAudit: (messageId: string, question: string, answer: string, sources: SourceItem[]) => void;
  auditLoadingId: string | null;
  error: string | null;
  onDismissError: () => void;
  depthMode: DepthMode;
  onDepthModeChange: (mode: DepthMode) => void;
}

const SUGGESTIONS = [
  {
    icon: HelpCircle,
    title: 'Explain a concept',
    desc: 'Bottom-up thinking in binary trees and why it is useful.',
    prompt: 'Explain bottom-up thinking in binary trees and why it is useful.',
  },
  {
    icon: FileText,
    title: 'Summarize a document',
    desc: 'Key fundamentals, properties, and tree structures.',
    prompt: 'Summarize the core algorithms and rules in Tree Fundamentals.',
  },
  {
    icon: GitCompare,
    title: 'Compare information',
    desc: 'DFS recursive traversals vs BFS level-order techniques.',
    prompt: 'Compare DFS recursive traversals with BFS level-order techniques.',
  },
  {
    icon: Search,
    title: 'Find conflicting evidence',
    desc: 'When is Morris traversal preferred over iterative BFS?',
    prompt: 'Are there scenarios where Morris traversal is preferred over iterative BFS?',
  },
];

export const ChatPage: React.FC<ChatPageProps> = ({
  conversation,
  inputPrompt,
  setInputPrompt,
  onSubmit,
  onStopStreaming,
  loading,
  streamingMessageId,
  docCount,
  onViewSource,
  onUploadFile,
  onRunAudit,
  auditLoadingId,
  error,
  onDismissError,
  depthMode,
  onDepthModeChange,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const messages = conversation?.messages || [];
  const isWelcome = messages.length === 0;

  useEffect(() => {
    if (!isWelcome) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages.length, loading, isWelcome]);

  const handleSuggestionClick = (prompt: string) => {
    setInputPrompt(prompt);
    onSubmit(prompt);
  };

  if (isWelcome) {
    return (
      <div className="welcome-container">
        {error && (
          <div className="alert-banner alert-banner-error" style={{ width: '100%', marginBottom: 20 }}>
            <AlertCircle size={16} />
            <span style={{ flex: 1 }}>{error}</span>
            <button onClick={onDismissError} style={{ fontWeight: 600, color: 'inherit' }}>
              ✕
            </button>
          </div>
        )}

        <div className="welcome-header">
          <h1 className="welcome-heading">What do your documents say?</h1>
          <p className="welcome-subtitle">
            Ask questions, connect ideas, and verify answers against your own documents.
          </p>
        </div>

        <ChatComposer
          variant="hero"
          value={inputPrompt}
          onChange={setInputPrompt}
          onSubmit={() => onSubmit()}
          onStop={onStopStreaming}
          loading={loading}
          docCount={docCount}
          onUploadFile={onUploadFile}
          placeholder="Ask me anything about your documents..."
          depthMode={depthMode}
          onDepthModeChange={onDepthModeChange}
        />

        <div className="suggestions-wrap">
          <div className="suggestions-label">Get started with a use case:</div>
          <div className="suggestions-grid">
            {SUGGESTIONS.map((item, idx) => {
              const Icon = item.icon;
              return (
                <div
                  key={idx}
                  className="suggestion-card"
                  onClick={() => handleSuggestionClick(item.prompt)}
                >
                  <div className="suggestion-icon">
                    <Icon size={16} />
                  </div>
                  <div className="suggestion-content">
                    <div className="suggestion-title">{item.title}</div>
                    <div className="suggestion-desc">{item.desc}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <div className="messages-list">
        {error && (
          <div className="alert-banner alert-banner-error">
            <AlertCircle size={16} />
            <span style={{ flex: 1 }}>{error}</span>
            <button onClick={onDismissError} style={{ fontWeight: 600, color: 'inherit' }}>
              ✕
            </button>
          </div>
        )}

        {messages.map((msg, idx) => {
          // Find the corresponding user question if this is an assistant message
          let userQuestion = '';
          if (msg.role === 'assistant') {
            for (let i = idx - 1; i >= 0; i--) {
              if (messages[i].role === 'user') {
                userQuestion = messages[i].content;
                break;
              }
            }
          }

          return (
            <ChatMessage
              key={msg.id}
              role={msg.role}
              content={msg.content}
              sources={msg.sources}
              audit={msg.audit}
              metrics={msg.metrics}
              onViewSource={onViewSource}
              onRunAudit={
                msg.role === 'assistant'
                  ? () => onRunAudit(msg.id, userQuestion, msg.content, msg.sources || [])
                  : undefined
              }
              auditLoading={auditLoadingId === msg.id}
              isStreaming={streamingMessageId === msg.id}
            />
          );
        })}

        <div ref={messagesEndRef} />
      </div>

      <div className="bottom-composer-wrap">
        <ChatComposer
          variant="bottom"
          value={inputPrompt}
          onChange={setInputPrompt}
          onSubmit={() => onSubmit()}
          onStop={onStopStreaming}
          loading={loading}
          docCount={docCount}
          onUploadFile={onUploadFile}
          placeholder="Ask a follow-up question..."
          depthMode={depthMode}
          onDepthModeChange={onDepthModeChange}
        />
      </div>
    </div>
  );
};
