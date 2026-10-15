export interface SourceItem {
  label: string;
  filename: string;
  page: number | string;
  excerpt: string;
}

export interface AuditDetails {
  text: string;
  status: string;
  used_labels: string[];
  invalid_labels: string[];
}

export interface PerfMetrics {
  retrieval_ms: number;
  ttft_ms: number;
  generation_ms: number;
  total_ms: number;
  token_count: number;
  tokens_per_sec: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceItem[];
  audit?: AuditDetails;
  timings?: Record<string, number>;
}

export interface DocumentMeta {
  filename: string;
  size_bytes: number;
  size_human: string;
  modified_timestamp: number;
  is_indexed: boolean;
}

export interface DocumentListResponse {
  documents: DocumentMeta[];
  total: number;
}

export interface HealthResponse {
  status: string;
  service: string;
  model: string;
  embedding_model: string;
  collection: string;
  database_path: string;
  indexed_chunks: number;
  documents_count: number;
  qdrant_status: string;
  is_warm?: boolean;
  top_k?: number;
  num_ctx?: number;
  num_predict?: number;
  local_only: boolean;
}

export interface UploadResponse {
  status: string;
  message: string;
  filename: string;
  size_bytes: number;
  size_human: string;
  pages_processed: number;
  chunks_indexed: number;
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  messages: Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: SourceItem[];
    audit?: AuditDetails;
    metrics?: PerfMetrics;
    timestamp: number;
  }>;
}

const API_BASE = '';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error(`Failed to check health: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchDocuments(): Promise<DocumentListResponse> {
  const res = await fetch(`${API_BASE}/api/documents`);
  if (!res.ok) {
    throw new Error(`Failed to fetch documents: ${res.statusText}`);
  }
  return res.json();
}

export async function sendChatMessage(
  question: string,
  history: ChatHistoryMessage[] = []
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      history,
    }),
  });

  if (!res.ok) {
    let errMessage = `Chat request failed: ${res.statusText}`;
    try {
      const errData = await res.json();
      if (errData.detail) errMessage = errData.detail;
    } catch {
      // fallback
    }
    throw new Error(errMessage);
  }

  return res.json();
}

export async function fetchAudit(
  question: string,
  answer: string,
  sources: SourceItem[]
): Promise<AuditDetails> {
  const res = await fetch(`${API_BASE}/api/chat/audit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      answer,
      sources,
    }),
  });

  if (!res.ok) {
    let errMessage = `Audit failed: ${res.statusText}`;
    try {
      const errData = await res.json();
      if (errData.detail) errMessage = errData.detail;
    } catch {
      // fallback
    }
    throw new Error(errMessage);
  }

  return res.json();
}

export interface StreamCallbacks {
  onSources: (sources: SourceItem[]) => void;
  onToken: (token: string) => void;
  onDone: (metrics?: PerfMetrics, used_labels?: string[], invalid_labels?: string[]) => void;
  onError: (err: string) => void;
}

export type DepthMode = 'brief' | 'standard' | 'detailed';

export async function streamChatMessage(
  question: string,
  history: ChatHistoryMessage[] = [],
  callbacks: StreamCallbacks,
  depthMode: DepthMode = 'standard',
  model?: string,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      history,
      depth_mode: depthMode,
      model,
    }),
    signal,
  });

  if (!res.ok) {
    let errMessage = `Streaming request failed: ${res.statusText}`;
    try {
      const errData = await res.json();
      if (errData.detail) errMessage = errData.detail;
    } catch {
      // fallback
    }
    callbacks.onError(errMessage);
    return;
  }

  if (!res.body) {
    callbacks.onError('ReadableStream not supported by browser.');
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        break;
      }

      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse complete SSE messages separated by \n\n
      const parts = buffer.split('\n\n');
      buffer = parts.pop() || '';

      for (const part of parts) {
        if (!part.trim()) continue;

        let eventType = 'message';
        let dataStr = '';

        const lines = part.split('\n');
        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.replace('event:', '').trim();
          } else if (line.startsWith('data:')) {
            dataStr = line.replace('data:', '').trim();
          }
        }

        if (!dataStr) continue;

        try {
          const parsed = JSON.parse(dataStr);
          if (eventType === 'sources' && parsed.sources) {
            callbacks.onSources(parsed.sources);
          } else if (eventType === 'token' && parsed.delta) {
            callbacks.onToken(parsed.delta);
          } else if (eventType === 'done') {
            callbacks.onDone(parsed.metrics, parsed.used_labels, parsed.invalid_labels);
          } else if (eventType === 'error') {
            callbacks.onError(parsed.error || 'Server error during generation');
          }
        } catch (e) {
          console.warn('Failed to parse SSE payload', dataStr, e);
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      console.log('Stream aborted by client.');
    } else {
      callbacks.onError(err.message || 'Stream connection error');
    }
  } finally {
    reader.releaseLock();
  }
}

export async function uploadPdfDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/api/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let errMessage = `Upload failed: ${res.statusText}`;
    try {
      const errData = await res.json();
      if (errData.detail) errMessage = errData.detail;
    } catch {
      // fallback
    }
    throw new Error(errMessage);
  }

  return res.json();
}

export function getPdfUrl(filename: string, page?: number | string): string {
  const base = `${API_BASE}/api/pdfs/${encodeURIComponent(filename)}`;
  return page ? `${base}#page=${page}` : base;
}
