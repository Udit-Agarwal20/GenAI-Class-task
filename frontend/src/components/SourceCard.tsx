import React from 'react';
import { ExternalLink } from 'lucide-react';
import type { SourceItem } from '../lib/api';

interface SourceCardProps {
  source: SourceItem;
  isHighlighted?: boolean;
  onViewSource: (filename: string, page: number | string) => void;
}

export const SourceCard: React.FC<SourceCardProps> = ({
  source,
  isHighlighted = false,
  onViewSource,
}) => {
  return (
    <div
      id={`source-${source.label}`}
      className={`source-card ${isHighlighted ? 'highlighted' : ''}`}
    >
      <div className="source-header">
        <div className="source-meta">
          <span className="source-tag">[{source.label}]</span>
          <span className="source-filename" title={source.filename}>
            {source.filename}
          </span>
          <span className="source-page">Page {source.page}</span>
        </div>

        <button
          className="view-source-btn"
          onClick={() => onViewSource(source.filename, source.page)}
          title={`Open ${source.filename} at page ${source.page}`}
        >
          <ExternalLink size={12} />
          <span>View Source</span>
        </button>
      </div>

      <div className="source-excerpt">
        "{source.excerpt}"
      </div>
    </div>
  );
};
