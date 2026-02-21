# 1. Imports
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

import hashlib
import os

load_dotenv()

print("All imports done")

# 2. Agent Configurations
ROOT_DIR = Path(__file__).parent.parent  # DataIngestion/ -> FinRAG/

@dataclass
class AgentConfig:
    model_name: str = os.getenv("MODEL_NAME", "ministral-3:14b-instruct-2512-q4_K_M")
    embedding_model: str = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
    base_url: str = os.getenv("BASE_URL", "http://localhost:11434")
    data_dir: Path = ROOT_DIR / os.getenv("DATA_DIR", "data")
    chroma_dir: Path = ROOT_DIR / os.getenv("CHROMA_DIR", "chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "fin_rag")

    def __post__init__(self):
        """Create necessary directories in the root if they do not exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


print("Configurations complete")

# 3. Embeddings and Vector Store
config = AgentConfig()
embeddings = OllamaEmbeddings(model=config.embedding_model, base_url=config.base_url)

vector_store = Chroma(
    collection_name=config.collection_name,
    embedding_function=embeddings,
    persist_directory=str(config.chroma_dir)
)

print("Instance Created and vector store initialized")


# 4. Extracting Metadata from the filename
def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract metadata from filename.

    Expected Format: {company} {doc_type} {quarter} {year}.pdf
    Example:
        amazon 10-k q2 2023
    
    Returns:
        dict with company_name, document_type, fiscal_quarter, fiscal_year
    """
    name = filename.replace('.pdf', '')
    parts = name.split(" ")
    metadata = {
        "company": None,
        "doc_type": None,
        "fiscal_quarter": None,
        "fiscal_year": None
    }

    if len(parts) == 4:
        metadata['fiscal_quarter'] = parts[2]
        metadata['fiscal_year'] = int(parts[3])
    
    else:
        metadata['fiscal_quarter'] = None
        metadata['fiscal_year'] = int(parts[2])
    
    metadata['company'] = parts[0]
    metadata['doc_type'] = parts[1]

    return metadata

print(extract_metadata_from_filename('amazon 10-k q3 2023'))

# 5. Extracting PDF pages using docling
def extract_pdf_pages(pdf_path):
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    page_break = "<!--- page break --->"
    markdown_text = result.document.export_to_markdown(page_break_placeholder=page_break)
    pages = markdown_text.split(page_break)


    return pages

# 6. Computing hash files for avoiding data/document duplication
def compute_hash_file(filepath: str) -> str:
    """Compute SHA-256 for of file content."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        
    return sha256_hash.hexdigest()

print(compute_hash_file('data/amazon/amazon 10-k 2023.pdf'))

# 7. Document Ingestion into the vector DB
def document_ingestion(pdf_path):
    existing_docs = vector_store.get(where={"file_hash": {"$ne": ""}}, include=['metadatas'])
    processed_hash = [m.get("file_hash") for m in existing_docs['metadatas'] if m.get('file_hash')]
    processed_hash = set(processed_hash)

    print(f"Processing: {pdf_path.name}")
    file_hash = compute_hash_file(pdf_path)

    if file_hash in processed_hash:
        print(f"Already Processed: {pdf_path}")
        return
    
    pages = extract_pdf_pages(pdf_path)
    file_metadata = extract_metadata_from_filename(pdf_path.name)

    processed_pages = []
    MAX_CHARS = 15000

    for page_num, page_text in enumerate(pages, start=1):
        print("Original Page Length:", len(page_text))

        if len(page_text) > MAX_CHARS:
            print(f"Truncating page {page_num} from {len(page_text)} chars")
            page_text = page_text[:MAX_CHARS]
        
        metadata_dict = file_metadata.copy()
        metadata_dict['page_num'] = page_num
        metadata_dict['file_hash'] = file_hash
        metadata_dict['source_file'] = pdf_path.name

        doc = Document(page_content=page_text, metadata=metadata_dict)
        processed_pages.append(doc)
    
    vector_store.add_documents(processed_pages)


# 8. Main Ingestion Loop
def ingest_all_documents():
    pdf_files = list(config.data_dir.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data directory")
        return
    print(f"Found {len(pdf_files)} in data directory")

    for pdf_path in pdf_files:
        document_ingestion(pdf_path)
    
    print("All documents ingested sucessfully!")

ingest_all_documents()