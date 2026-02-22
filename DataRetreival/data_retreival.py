# 1. Imports
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from Schemas.rag_schema import ChunkMetadata, RankingKeywords
from dataclasses import dataclass

from pathlib import Path
from rank_bm25 import BM25Plus

import hashlib
import re
import os

load_dotenv()

print("All Imports Successful!")

# 2. Configurations and LLMFactory

ROOT_DIR = Path(__file__).parent.parent

@dataclass
class AgentConfig:
    llm_model: str = os.getenv("MODEL_NAME")
    base_url: str = os.getenv("BASE_URL")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL")
    collection_name: str = os.getenv("COLLECTION_NAME", "financial_docs")
    chroma_dir: Path = ROOT_DIR / os.getenv("CHROMA_DIR")
    data_dir: Path = ROOT_DIR / os.getenv("DATA_DIR")

print("All Configs done")

# 3. Setting up embeddings, llm and vector store
config = AgentConfig()
embeddings = OllamaEmbeddings(model=config.embeddings_model, base_url=config.base_url)

vector_store = Chroma(
    collection_name=config.collection_name,
    embedding_function=embeddings,
    persist_directory=config.chroma_dir
)

llm = ChatOllama(model=config.llm_model, base_url=config.base_url)

print("LLM, Embeddings defined and vector store intialised")

# 4. Extract Filters
def extract_filters(user_query: str):
    """
    This function extracts metadata like company name, document type, fiscal year and fiscal quarter from the document name.
    Args:
        takes a string as input, user query
        e.g. (What is the annual financial report of amazon for the year 2023?)
    
    Returns:
        a serializable, a python dictionary, returns None for the fields not mentioned
        e.g. ({'company_name': 'amazon', 'doc_type': '10-k', 'fiscal_year': 2023, 'fiscal_quarter': None})
    """

    llm_structured = llm.with_structured_output(ChunkMetadata)
    prompt = f"""Extract metadata filters from the query. Return None for fields not mentioned.

                USER QUERY: {user_query}

                COMPANY MAPPINGS:
                - Amazon/AMZN -> amazon
                - Google/Alphabet/GOOGL/GOOG -> google
                - Apple/AAPL -> apple
                - Microsoft/MSFT -> microsoft
                - Tesla/TSLA -> tesla
                - Nvidia/NVDA -> nvidia
                - Meta/Facebook/FB -> meta

                DOC TYPE:
                - Annual report -> 10-k
                - Quarterly report -> 10-q
                - Current report -> 8-k

                EXAMPLES:
                "Amazon Q3 2024 revenue" -> {{"company_name": "amazon", "doc_type": "10-q", "fiscal_year": 2024, "fiscal_quarter": "q3"}}
                "Apple 2023 annual report" -> {{"company_name": "apple", "doc_type": "10-k", "fiscal_year": 2023}}
                "Tesla profitability" -> {{"company_name": "tesla"}}

                Extract metadata:
                """
    
    metadata = llm_structured.invoke(prompt)
    filters = metadata.model_dump(exclude_none=True)

    return filters

print(extract_filters("What is amazon's annual financial report for 2024?"))

# 5. Generate Ranking Keywords
def generate_ranking_keywords(user_query: str):
    """
    This function uses LLMs to extract/generate five keywords from the user query in accordance with the SEC filing terminology.
    Args:
        user_query: string
        e.g. ('What is apple's quarterly financial report for q3 2024?')
    
    Returns:
        exactly five keywords: List
        e.g. ("revenue analysis" -> ["revenue", "net revenue", "total revenue", "consolidated statements of operations", "net sales"])
    """
    llm_structured = llm.with_structured_output(RankingKeywords)

    prompt = f"""Generate EXACTLY 5 financial keywords from SEC filings terminology.

                USER QUERY: {user_query}

                USE EXACT TERMS FROM 10-K/10-Q FILINGS:

                STATEMENT HEADINGS:
                "consolidated statements of operations", "consolidated balance sheets", "consolidated statements of cash flows", "consolidated statements of stockholders equity"

                INCOME STATEMENT:
                "revenue", "net revenue", "cost of revenue", "gross profit", "operating income", "net income", "earnings per share"

                BALANCE SHEET:
                "total assets", "cash and cash equivalents", "total liabilities", "stockholders equity", "working capital", "long-term debt"

                CASH FLOWS:
                "cash flows from operating activities", "net cash provided by operating activities", "cash flows from investing activities", "free cash flow", "capital expenditures"

                RULES:
                - Return EXACTLY 5 keywords
                - Use exact phrases from SEC filings
                - Match query topic (revenue -> revenue terms, cash -> cash flow terms)
                - Use "cash flows" (plural), "stockholders equity"

                EXAMPLES:
                "revenue analysis" -> ["revenue", "net revenue", "total revenue", "consolidated statements of operations", "net sales"]
                "cash flow performance" -> ["consolidated statements of cash flows", "cash flows from operating activities", "net cash provided by operating activities", "free cash flow", "operating activities"]
                "balance sheet strength" -> ["consolidated balance sheets", "total assets", "stockholders equity", "cash and cash equivalents", "long-term debt"]

                Generate EXACTLY 5 keywords:
                """
    
    result = llm_structured.invoke(prompt)
    return result.keywords

print(generate_ranking_keywords("What is amazon's annual financial report for 2024?"))

# 6. Build Search Keywords
def search_kwargs(filters, ranking_keywords, k=3):
    pass

