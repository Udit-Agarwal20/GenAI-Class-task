import React, { useState, useRef } from 'react';
import {
  Upload,
  Search,
  FileText,
  ExternalLink,
  CheckCircle2,
  AlertCircle,
  Loader2,
  BookOpen
} from 'lucide-react';
import type { DocumentMeta } from '../lib/api';

interface LibraryPageProps {
  documents: DocumentMeta[];
  loadingDocs: boolean;
  onUploadFile: (file: File) => Promise<void>;
  onViewSource: (filename: string, page: number | string) => void;
  uploading: boolean;
  uploadStatus: string | null;
  error: string | null;
  onDismissError: () => void;
}

export const LibraryPage: React.FC<LibraryPageProps> = ({
  documents,
  loadingDocs,
  onUploadFile,
  onViewSource,
  uploading,
  uploadStatus,
  error,
  onDismissError,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const filteredDocs = documents.filter((doc) =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase().trim())
  );

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await onUploadFile(file);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await onUploadFile(file);
      e.target.value = '';
    }
  };

  return (
    <div className="library-container">
      {error && (
        <div className="alert-banner alert-banner-error">
          <AlertCircle size={16} />
          <span style={{ flex: 1 }}>{error}</span>
          <button onClick={onDismissError} style={{ fontWeight: 600, color: 'inherit' }}>
            ✕
          </button>
        </div>
      )}

      {uploadStatus && (
        <div className="alert-banner alert-banner-success">
          <CheckCircle2 size={16} />
          <span style={{ flex: 1 }}>{uploadStatus}</span>
        </div>
      )}

      <div className="library-header">
        <div className="library-title-group">
          <h1>Document Library</h1>
          <p>
            {documents.length} local PDFs indexed and searchable with Qdrant and Nomic embeddings.
          </p>
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept=".pdf,application/pdf"
        onChange={handleFileChange}
      />

      <div
        className={`library-upload-box ${isDragOver ? 'drag-over' : ''}`}
        onClick={() => !uploading && fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        style={{
          borderColor: isDragOver ? 'var(--forest-600)' : undefined,
          backgroundColor: isDragOver ? 'var(--mint-surface)' : undefined,
        }}
      >
        <div className="upload-icon-circle">
          {uploading ? (
            <Loader2 size={24} className="loading-spinner" />
          ) : (
            <Upload size={24} />
          )}
        </div>
        <div className="upload-box-title">
          {uploading
            ? 'Processing & Indexing PDF...'
            : 'Click to upload or drag & drop PDF'}
        </div>
        <div className="upload-box-subtitle">
          {uploading
            ? 'Extracting pages, splitting chunks, and generating local Nomic embeddings'
            : 'Files are processed locally and appended to the proofline_docs collection'}
        </div>
      </div>

      <div className="library-controls-bar">
        <div className="search-input-wrap">
          <Search size={16} />
          <input
            type="text"
            className="search-input"
            placeholder="Search documents by filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
          Showing {filteredDocs.length} of {documents.length} documents
        </div>
      </div>

      {loadingDocs ? (
        <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
          <Loader2 size={24} className="loading-spinner" style={{ margin: '0 auto 12px auto' }} />
          <div>Loading local document catalog...</div>
        </div>
      ) : filteredDocs.length === 0 ? (
        <div
          style={{
            textAlign: 'center',
            padding: '60px 0',
            backgroundColor: '#ffffff',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <BookOpen size={36} color="var(--text-muted)" style={{ margin: '0 auto 12px auto' }} />
          <div style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: 4 }}>
            No documents matched your search
          </div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Try searching with a different keyword or upload a new PDF document.
          </div>
        </div>
      ) : (
        <div className="documents-grid">
          {filteredDocs.map((doc) => (
            <div key={doc.filename} className="doc-card">
              <div className="doc-card-top">
                <div className="doc-icon-wrap">
                  <FileText size={20} />
                </div>
                <div className="doc-info">
                  <div className="doc-name" title={doc.filename}>
                    {doc.filename}
                  </div>
                  <div className="doc-meta-row">
                    <span>{doc.size_human}</span>
                    <span>•</span>
                    <span className="doc-badge-indexed">
                      <CheckCircle2 size={10} />
                      Indexed
                    </span>
                  </div>
                </div>
              </div>

              <div className="doc-card-bottom">
                <span style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                  Local Storage
                </span>
                <button
                  className="doc-read-btn"
                  onClick={() => onViewSource(doc.filename, 1)}
                  title={`Open ${doc.filename}`}
                >
                  <ExternalLink size={13} />
                  <span>Read PDF</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
