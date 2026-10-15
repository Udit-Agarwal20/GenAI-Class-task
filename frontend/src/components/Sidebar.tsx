import React from 'react';
import {
  FileText,
  MessageSquare,
  Plus,
  Trash2,
  X,
  Database,
  Cpu,
  Sparkles,
  BookOpen
} from 'lucide-react';
import type { Conversation, HealthResponse } from '../lib/api';

interface SidebarProps {
  currentTab: 'chat' | 'library';
  onSelectTab: (tab: 'chat' | 'library') => void;
  onNewChat: () => void;
  conversations: Conversation[];
  activeConvId: string | null;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  health: HealthResponse | null;
  docCount: number;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  onNewChat,
  conversations,
  activeConvId,
  onSelectConversation,
  onDeleteConversation,
  health,
  docCount,
  mobileOpen,
  onCloseMobile,
}) => {
  return (
    <>
      {mobileOpen && (
        <div
          className="modal-overlay"
          style={{ zIndex: 25, backgroundColor: 'rgba(0,0,0,0.3)' }}
          onClick={onCloseMobile}
        />
      )}

      <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-header">
          <div className="brand">
            <div className="brand-icon">
              <FileText size={18} strokeWidth={2.5} />
            </div>
            <div>
              <span className="brand-title">Proofline AI</span>
            </div>
          </div>
          <span className="brand-badge">Local</span>
          {mobileOpen && (
            <button
              onClick={onCloseMobile}
              style={{ padding: 4, color: 'var(--text-muted)' }}
              aria-label="Close menu"
            >
              <X size={18} />
            </button>
          )}
        </div>

        <div className="sidebar-action-wrap">
          <button className="new-chat-btn" onClick={onNewChat}>
            <Plus size={16} strokeWidth={2.5} />
            <span>New Chat</span>
          </button>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${currentTab === 'chat' ? 'active' : ''}`}
            onClick={() => {
              onSelectTab('chat');
              onCloseMobile();
            }}
          >
            <div className="nav-item-left">
              <MessageSquare size={16} />
              <span>Ask Documents</span>
            </div>
          </button>

          <button
            className={`nav-item ${currentTab === 'library' ? 'active' : ''}`}
            onClick={() => {
              onSelectTab('library');
              onCloseMobile();
            }}
          >
            <div className="nav-item-left">
              <BookOpen size={16} />
              <span>Document Library</span>
            </div>
            <span className="count-pill">{docCount}</span>
          </button>
        </nav>

        <div className="sidebar-section-title">Recent Conversations</div>

        <div className="sidebar-conversations">
          {conversations.length === 0 ? (
            <div
              style={{
                padding: '16px 12px',
                fontSize: '0.78rem',
                color: 'var(--text-muted)',
                fontStyle: 'italic',
                textAlign: 'center',
              }}
            >
              No past conversations
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={`conv-item ${
                  currentTab === 'chat' && activeConvId === conv.id ? 'active' : ''
                }`}
                onClick={() => {
                  onSelectConversation(conv.id);
                  onSelectTab('chat');
                  onCloseMobile();
                }}
              >
                <div className="conv-title">{conv.title}</div>
                <button
                  className="conv-delete-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                  title="Delete chat"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <div className="system-status-card">
            <div className="status-badge-row">
              <div className="status-indicator">
                <span className="status-dot"></span>
                <span>Local AI Ready</span>
              </div>
              <Sparkles size={13} color="var(--forest-600)" />
            </div>

            <div className="status-info-row" style={{ marginTop: 4 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Cpu size={11} /> {health?.model || 'gemma4:e2b'}
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <Database size={11} /> {health?.indexed_chunks ?? 226} chunks
              </span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
