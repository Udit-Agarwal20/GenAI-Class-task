# Proofline AI — Local PDF Intelligence & Evidence-First RAG

Proofline AI is a local, privacy-first research assistant that allows you to chat with your PDF documents, verify factual claims with clickable citations, inspect exact source excerpts and page numbers, and run automated evidence audits—all running 100% locally on your machine via Ollama and Qdrant.

---

## 🎯 Architecture & Local AI Stack

- **Local LLM**: `gemma4:e2b` via Ollama (`ChatOllama`)
- **Local Embedding Model**: `nomic-embed-text` with `NomicPrefixedEmbeddings` (`search_document:` & `search_query:` prefixes)
- **Vector Store**: Qdrant Local (`proofline_docs` collection in `qdrant_langchain_storage/`)
- **Backend**: FastAPI (`server.py`)
  - `GET /api/health` — live vector index status, chunk count, model status
  - `GET /api/documents` — PDF catalog, file sizes, indexed badges
  - `POST /api/documents/upload` — real-time upload, `PyPDFLoader` parsing, `RecursiveCharacterTextSplitter`, Nomic embedding, and incremental Qdrant insertion
  - `GET /api/pdfs/{filename}` — secure PDF serving with strict path traversal protection
  - `POST /api/chat` — semantic retrieval, evidence formatting (`[S1]`, `[S2]`), Gemma answer generation, and automated evidence auditing
- **Frontend**: React + TypeScript + Vite (`frontend/`)
  - Clean, research-oriented design system (warm off-white, mint surfaces, deep forest green)
  - Responsive collapsible sidebar with active local status indicator
  - Hero welcome screen with 4 interactive suggestion cards
  - Markdown message rendering with clickable citation badges
  - Expandable **Sources** section with excerpts and 1-click **View Source**
  - Built-in **PDF Viewer** modal with page navigation and download controls
  - Dedicated **Document Library** with drag-and-drop upload and real-time search

---

## 🚀 Quickstart & Launch Commands

### 1. Prerequisites
Ensure local models are installed in Ollama:
```bash
ollama pull nomic-embed-text
ollama pull gemma4:e2b
```

### 2. Start the Backend Server (Terminal 1)
From the project root:
```bash
# Using the existing virtual environment
source .venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000
```
Backend will start on `http://127.0.0.1:8000`.

### 3. Start the Frontend Dev Server (Terminal 2)
In a second terminal:
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser.

*(Note: The FastAPI backend also serves the built production frontend directly from `http://127.0.0.1:8000` when accessed directly!)*

---

## 📁 Project Structure

```
proofline-ai/
├── data/                         # Local PDF documents (16 indexed PDFs)
├── qdrant_langchain_storage/     # Persistent local Qdrant database
├── langchain_chunks.json         # Extracted and indexed document chunks
├── ingest_langchain.py           # Existing LangChain PDF ingestion script
├── index_langchain.py            # Existing LangChain Qdrant indexing script
├── answer_langchain.py           # Existing CLI RAG interface
├── server.py                     # FastAPI REST API backend
├── frontend/                     # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.tsx       # Collapsible navigation & local AI status
│   │   │   ├── ChatComposer.tsx  # Hero & bottom composer with PDF attachment
│   │   │   ├── ChatMessage.tsx   # Markdown chat bubbles & citation links
│   │   │   ├── SourceCard.tsx    # Evidence cards with excerpt & View Source
│   │   │   ├── EvidenceAudit.tsx # Automated Gemma preliminary audit panel
│   │   │   └── PdfViewer.tsx     # Built-in modal PDF reader with page jumping
│   │   ├── pages/
│   │   │   ├── ChatPage.tsx      # Welcome hero screen & conversation view
│   │   │   └── LibraryPage.tsx   # Document library, upload zone & search
│   │   ├── lib/
│   │   │   └── api.ts            # Type-safe API client
│   │   ├── App.tsx               # State coordination & localStorage history
│   │   ├── index.css             # Research-tool custom CSS design system
│   │   └── main.tsx
│   ├── vite.config.ts            # Vite config with /api proxy to FastAPI
│   └── package.json
└── README.md
```

---

## 🔍 Verification & Features

1. **Evidence-First Answers**:
   - Citations like `[S1]`, `[S2]` are displayed in the text as interactive badges.
   - Clicking a badge or expanding the "Retrieved Evidence" section highlights the source card, displaying the filename, page number, and relevant text passage.
2. **Built-in PDF Viewer**:
   - Click "View Source" on any card or document in the library to open the embedded PDF reader directly at that page.
3. **Automated Evidence Audit**:
   - Expand the "Automated Evidence Audit" accordion under any answer to review Gemma's independent audit classifying claims as SUPPORTED, PARTIALLY SUPPORTED, or UNSUPPORTED against the retrieved evidence.
4. **Document Library**:
   - Search by filename in real-time.
   - Upload new PDFs via file picker or drag-and-drop to extract, split, embed with Nomic, and append to Qdrant without recreating the database.
