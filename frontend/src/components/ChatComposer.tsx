import React, { useRef, useEffect } from 'react';
import { ArrowUp, Paperclip, FileText, Square } from 'lucide-react';
import type { DepthMode } from '../lib/api';

interface ChatComposerProps {
  variant?: 'hero' | 'bottom';
  value: string;
  onChange: (val: string) => void;
  onSubmit: () => void;
  onStop?: () => void;
  loading: boolean;
  docCount: number;
  onUploadFile?: (file: File) => void;
  placeholder?: string;
  depthMode?: DepthMode;
  onDepthModeChange?: (mode: DepthMode) => void;
}

export const ChatComposer: React.FC<ChatComposerProps> = ({
  variant = 'hero',
  value,
  onChange,
  onSubmit,
  onStop,
  loading,
  docCount,
  onUploadFile,
  placeholder = 'Ask anything about your indexed documents...',
  depthMode = 'standard',
  onDepthModeChange,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = variant === 'hero' ? 180 : 130;
      textareaRef.current.style.height = `${Math.min(scrollHeight, maxHeight)}px`;
    }
  }, [value, variant]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!loading && value.trim()) {
        onSubmit();
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0 && onUploadFile) {
      const file = e.target.files[0];
      onUploadFile(file);
      e.target.value = '';
    }
  };

  const isHero = variant === 'hero';

  return (
    <div className={isHero ? 'composer-box-hero' : 'composer-box-bottom'}>
      <textarea
        ref={textareaRef}
        className={isHero ? 'composer-textarea' : 'composer-textarea-bottom'}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
        rows={isHero ? 2 : 1}
      />

      <div className="composer-bottom-bar">
        <div className="composer-tools-left">
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
          />
          {onUploadFile && (
            <button
              type="button"
              className="composer-tool-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              title="Upload new PDF into local library"
            >
              <Paperclip size={14} />
              <span>Attach PDF</span>
            </button>
          )}

          <div className="composer-doc-indicator">
            <FileText size={13} />
            <span>{docCount} PDFs Indexed</span>
          </div>

          {onDepthModeChange && (
            <div className="depth-selector" title="Answer Depth Mode (Brief: 512, Standard: 1024, Detailed: 1536 tokens)">
              <button
                type="button"
                className={`depth-selector-btn ${depthMode === 'brief' ? 'active' : ''}`}
                onClick={() => onDepthModeChange('brief')}
                disabled={loading}
              >
                Brief
              </button>
              <button
                type="button"
                className={`depth-selector-btn ${depthMode === 'standard' ? 'active' : ''}`}
                onClick={() => onDepthModeChange('standard')}
                disabled={loading}
              >
                Standard
              </button>
              <button
                type="button"
                className={`depth-selector-btn ${depthMode === 'detailed' ? 'active' : ''}`}
                onClick={() => onDepthModeChange('detailed')}
                disabled={loading}
              >
                Detailed
              </button>
            </div>
          )}
        </div>

        {loading ? (
          <button
            type="button"
            className="send-btn"
            style={{ backgroundColor: '#be123c' }}
            onClick={onStop}
            title="Stop generation"
          >
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            className="send-btn"
            onClick={onSubmit}
            disabled={!value.trim()}
            title="Send question (Enter)"
          >
            <ArrowUp size={18} strokeWidth={2.4} />
          </button>
        )}
      </div>
    </div>
  );
};
