# Document Loaders — Ingesting Unstructured Data into RAG Pipelines

> **来源**: https://docs.unstructured.io/open-source/core-functionality/partitioning + https://docs.unstructured.io/open-source/introduction/quick-start
> **抓取日期**: 2026-07-18
> **作者/机构**: Unstructured.io (open-source library) + LangChain official documentation patterns
> **备注**: 本文档综合 Unstructured.io 官方分区(partitioning)文档与 LangChain document loaders 的标准接口模式

## 1. What is a Document Loader?

In a Retrieval-Augmented Generation (RAG) pipeline, the very first step is to **load source documents** and convert them into a structured representation that downstream components (chunkers, embedders, vector databases) can consume.

A **document loader** is a component that:

1. Reads files from various sources (local disk, URLs, S3, Notion, Confluence, etc.).
2. Extracts the textual and structural content.
3. Returns a list of standardized `Document` objects — each with a `page_content` (text) and `metadata` (source, page number, author, etc.).

In RAG, the document loader is the **bridge between the messy real world** (PDFs, slides, HTML pages, spreadsheets, images, emails) **and the clean structured world** that an LLM + vector database can reason over.

## 2. Document Object (Standard Interface)

Regardless of source, loaders across frameworks (LangChain, LlamaIndex, Haystack, Unstructured) converge on a common **Document** schema:

```python
class Document:
    page_content: str   # The extracted text
    metadata: dict      # Source info: file path, page, author, etc.
```

Example:

```python
Document(
    page_content="Retrieval-Augmented Generation combines...",
    metadata={
        "source": "/docs/rag_paper.pdf",
        "page": 3,
        "author": "Lewis et al.",
        "created_at": "2020-05-22"
    }
)
```

## 3. Unstructured.io Partitioning Functions

[Unstructured](https://docs.unstructured.io/) is one of the most popular open-source libraries for parsing heterogeneous document formats. It exposes a `partition` API that automatically detects the file type and extracts structured elements.

### 3.1 Core API

```python
from unstructured.partition.auto import partition

# Auto-detect from a local file
elements = partition(filename="example-docs/pdf/layout-parser-paper-fast.pdf")

# From a URL
elements = partition(url="https://example.com/document.pdf")

# From a file-like object (e.g., uploaded file, S3 stream)
with open("doc.docx", "rb") as f:
    elements = partition(file=f)
```

Each returned `Element` has a `category` (type), `text`, and `metadata`:

| Element Type | Description |
|--------------|-------------|
| `NarrativeText` | Running prose (paragraphs) |
| `Title` | Section / document title |
| `ListItem` | Item in a bulleted or numbered list |
| `Table` | Tabular data (preserved as HTML or text) |
| `Image` | Embedded image (with optional OCR text) |
| `Header` / `Footer` | Page header/footer |
| `Address` | Postal address block |
| `EmailAddress` | Email address |
| `CodeSnippet` | Code block |
| `PageNumber` | Page number |
| `FigureCaption` | Caption of a figure |

### 3.2 File-Type-Specific Partition Functions

For more control, you can call the type-specific partitioner directly:

| File Type | Partition Function |
|-----------|--------------------|
| CSV | `partition_csv` |
| E-mail (.eml, .msg) | `partition_email`, `partition_msg` |
| Excel (.xlsx, .xls) | `partition_xlsx` |
| HTML | `partition_html` |
| Images (.png, .jpg, .jpeg, .tiff, .bmp, .heic) | `partition_image` |
| Markdown, Org Mode, RST | `partition_md`, `partition_org`, `partition_rst` |
| PDFs | `partition_pdf` |
| Plain Text | `partition_text` |
| PowerPoint (.ppt, .pptx) | `partition_ppt`, `partition_pptx` |
| Word (.doc, .docx) | `partition_doc`, `partition_docx` |
| XML | `partition_xml` |

### 3.3 PDF Partitioning — The Hard Case

PDFs are the most common document type in enterprise RAG and the trickiest to parse. Unstructured provides several strategies:

```python
from unstructured.partition.pdf import partition_pdf

# "fast" strategy: extract text quickly without OCR
elements = partition_pdf(
    "paper.pdf",
    strategy="fast",
    include_page_breaks=True
)

# "hi_res" strategy: use layout detection models for high-quality extraction
elements = partition_pdf(
    "paper.pdf",
    strategy="hi_res",
    infer_table_structure=True,        # extract tables as structured HTML
    extract_images_in_pdf=True,        # also extract embedded images
    extract_image_block_types=["Image", "Table"],
    extract_image_block_to_payload=False
)

# "ocr_only" strategy: pure OCR for scanned documents
elements = partition_pdf(
    "scanned.pdf",
    strategy="ocr_only",
    languages=["eng", "chi_sim"]
)
```

Strategies:

| Strategy | Speed | Quality | Use Case |
|----------|-------|---------|----------|
| `fast` | Fastest | Text-only PDFs | Digital text PDFs |
| `fast` + `ocr_fallback` | Medium | Handles scans | Mixed corpora |
| `hi_res` | Slow | Best layout/tables | Complex PDFs with tables, figures |
| `ocr_only` | Slow | Scanned docs | Image-only / scanned PDFs |

### 3.4 Element → LangChain Document Conversion

Once you have `Element` objects from Unstructured, convert them to LangChain `Document` objects:

```python
from unstructured.langchain import convert_to_documents

# elements from any partition_* call
docs = convert_to_documents(elements)

# Now ready for splitting, embedding, etc.
```

## 4. LangChain Document Loaders

LangChain ships with 100+ loaders, all conforming to the same simple interface:

```python
class BaseLoader:
    def load(self) -> List[Document]:
        """Load all documents at once."""

    def lazy_load(self) -> Iterator[Document]:
        """Stream documents one at a time — for large corpora."""
```

### 4.1 Common Loaders

| Loader | Source | Example |
|--------|--------|---------|
| `TextLoader` | Plain `.txt` files | `TextLoader("file.txt")` |
| `PyPDFLoader` | PDF files (text-based) | `PyPDFLoader("file.pdf")` |
| `PDFPlumberLoader` | PDFs with tables | `PDFPlumberLoader("file.pdf")` |
| `UnstructuredPDFLoader` | PDFs via Unstructured | `UnstructuredPDFLoader("file.pdf")` |
| `CSVLoader` | CSV files | `CSVLoader("data.csv")` |
| `JSONLoader` | JSON files (with jq schema) | `JSONLoader("data.json", jq_schema=".messages[].content")` |
| `HTMLLoader` / `BSHTMLLoader` | HTML files | `BSHTMLLoader("page.html")` |
| `WebBaseLoader` | Web pages (URL) | `WebBaseLoader("https://example.com")` |
| `NotionDBLoader` | Notion databases | `NotionDBLoader(integration_token=..., database_id=...)` |
| `GitHubLoader` | GitHub repo files | `GitHubLoader(repo="owner/repo", branch="main")` |
| `S3FileLoader` / `S3DirectoryLoader` | AWS S3 buckets | `S3FileLoader(bucket="...", key="...")` |
| `GoogleDriveLoader` | Google Drive files | `GoogleDriveLoader(folder_id="...")` |
| `ConfluenceLoader` | Confluence pages | `ConfluenceLoader(url=..., username=..., api_key=...)` |
| `SlackLoader` | Slack messages | `SlackLoader(channel_ids=["C123"])` |
| `YouTubeAudioLoader` + `OpenAIWhisperParser` | YouTube videos | Two-stage: audio + transcription |
| `UnstructuredFileLoader` / `UnstructuredAPIFileLoader` | Any file via Unstructured.io | `UnstructuredFileLoader("any_file")` |

### 4.2 Basic Usage Examples

```python
from langchain_community.document_loaders import PyPDFLoader

# Load a PDF
loader = PyPDFLoader("paper.pdf")
docs = loader.load()                  # all at once
print(f"Loaded {len(docs)} pages")
print(docs[0].page_content[:200])     # first page preview
print(docs[0].metadata)               # {'source': 'paper.pdf', 'page': 0}
```

```python
from langchain_community.document_loaders import WebBaseLoader

# Scrape a web page
loader = WebBaseLoader("https://blog.example.com/post")
docs = loader.load()
```

```python
from langchain_community.document_loaders import CSVLoader

# Load a CSV
loader = CSVLoader("data.csv")
docs = loader.load()
```

### 4.3 Lazy Loading — For Large Corpora

`lazy_load` returns an iterator so you can process millions of documents without blowing up memory:

```python
loader = S3DirectoryLoader("my-bucket", prefix="docs/")
for doc in loader.lazy_load():
    process(doc)   # handle one at a time
```

### 4.4 Load + Split + Embed — The Full RAG Ingestion Pipeline

A typical ingestion pipeline chains a loader → text splitter → embedding → vector store:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load
loader = PyPDFLoader("handbook.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed + Index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Save for later retrieval
vectorstore.save_local("handbook_faiss_index")
```

## 5. Multi-Source Loaders

For RAG systems that ingest from many sources, you can compose loaders:

```python
from langchain_community.document_loaders import (
    GitHubLoader,
    NotionDBLoader,
    S3DirectoryLoader,
    WebBaseLoader,
)

loaders = [
    GitHubLoader(repo="myorg/docs", glob="**/*.md"),
    NotionDBLoader(database_id="abc123"),
    S3DirectoryLoader("my-bucket", prefix="kb/"),
    WebBaseLoader("https://example.com/faq"),
]

all_docs = []
for loader in loaders:
    all_docs.extend(loader.load())
```

## 6. Loader Selection Guide

Choose your loader based on these criteria:

| If your source is... | Use... |
|----------------------|--------|
| Plain text / markdown | `TextLoader`, `UnstructuredMarkdownLoader` |
| Text-based PDF | `PyPDFLoader` (fast, simple) |
| PDF with tables / complex layout | `UnstructuredPDFLoader(strategy="hi_res")` |
| Scanned PDF (image-only) | `UnstructuredPDFLoader(strategy="ocr_only")` or `PDFPlumberLoader` |
| Word / PowerPoint / Excel | `UnstructuredWordDocumentLoader`, `UnstructuredPowerPointLoader`, `UnstructuredExcelLoader` |
| HTML / Web pages | `WebBaseLoader`, `BSHTMLLoader`, `AsyncHtmlLoader` |
| JSON with nested structure | `JSONLoader` (with `jq_schema`) |
| CSV | `CSVLoader` |
| GitHub / GitLab | `GitHubLoader` |
| Notion / Confluence / Slack | Respective dedicated loaders |
| Cloud storage (S3, GCS, Azure) | `S3FileLoader`, `GCSFileLoader`, `AzureBlobStorageFileLoader` |
| Databases | `SQLDatabaseLoader`, custom loaders |
| Audio / video | `OpenAIWhisperParser` + audio loader |

## 7. Best Practices

1. **Preserve metadata**: Always keep `source`, `page_number`, and other provenance fields — they enable citations and debugging.
2. **Handle encoding issues**: Specify `encoding="utf-8"` or use `autodetect_encoding=True` for text files.
3. **Stream large files**: Use `lazy_load` for corpora with thousands of files.
4. **Use OCR strategically**: Only enable `ocr_only` or `hi_res` strategies when needed; they are 10–100× slower than `fast`.
5. **Deduplicate**: Multiple sources may yield duplicate chunks — apply a hash-based dedup step after loading.
6. **Validate**: Spot-check a sample of loaded docs to ensure parsing quality before scaling.
7. **Chunk after loading**: Don't chunk during parsing — keep loaders pure so they can be swapped independently.

## 8. Role in the RAG Pipeline

The document loader sits at the **very beginning** of the RAG ingestion pipeline:

```
Sources → [LOADER] → Documents → [SPLITTER] → Chunks
   → [EMBEDDER] → Vectors → [VECTOR DB] → Index
                                              ↓
                              (Query time: RETRIEVER → GENERATOR → Answer)
```

A bad loader silently destroys downstream quality: if text is extracted incorrectly (wrong page boundaries, garbled OCR, missing tables), every retrieval and answer will be flawed. Investing in high-quality document loading is one of the highest-ROI engineering choices in any production RAG system.

## 9. References

- Unstructured.io documentation — https://docs.unstructured.io/
- Unstructured.io partitioning — https://docs.unstructured.io/open-source/core-functionality/partitioning
- LangChain Document Loaders (concept page) — https://docs.langchain.com/oss/python/langchain/overview
- LangChain v0.1 Document Loaders — https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
- LangChain integrations hub — https://python.langchain.com/docs/integrations/document_loaders/
