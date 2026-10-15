import React, { useEffect, useState } from 'react';
import { X, ExternalLink, Download, FileText, ChevronLeft, ChevronRight } from 'lucide-react';
import { getPdfUrl } from '../lib/api';

interface PdfViewerProps {
  filename: string;
  initialPage?: number | string;
  onClose: () => void;
}

export const PdfViewer: React.FC<PdfViewerProps> = ({
  filename,
  initialPage = 1,
  onClose,
}) => {
  const [currentPage, setCurrentPage] = useState<number>(Number(initialPage) || 1);

  // Close on ESC
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const pdfUrl = getPdfUrl(filename, currentPage);
  const downloadUrl = getPdfUrl(filename);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="pdf-viewer-modal" onClick={(e) => e.stopPropagation()}>
        <div className="pdf-viewer-toolbar">
          <div className="pdf-viewer-title">
            <FileText size={18} color="#4ade80" />
            <span style={{ maxWidth: 360, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {filename}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 14 }}>
              <button
                className="viewer-tool-btn"
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                title="Previous page"
              >
                <ChevronLeft size={16} />
              </button>
              <span style={{ fontSize: '0.8rem', color: '#c3ded1' }}>
                Page {currentPage}
              </span>
              <button
                className="viewer-tool-btn"
                onClick={() => setCurrentPage((p) => p + 1)}
                title="Next page"
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>

          <div className="pdf-viewer-controls">
            <a
              href={downloadUrl}
              download={filename}
              className="viewer-tool-btn"
              title="Download PDF"
            >
              <Download size={16} />
            </a>

            <a
              href={pdfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="viewer-tool-btn"
              title="Open in new browser tab"
            >
              <ExternalLink size={16} />
            </a>

            <button
              className="viewer-tool-btn"
              onClick={onClose}
              title="Close viewer (Esc)"
              style={{ marginLeft: 6 }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="pdf-frame-wrap">
          <iframe
            key={`${filename}-${currentPage}`}
            src={pdfUrl}
            title={filename}
            className="pdf-iframe"
          />
        </div>
      </div>
    </div>
  );
};
