# retriever.py
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging
from datetime import datetime

import config # Import config variables

# Initialize embedding function globally (or pass it around)
try:
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        google_api_key=config.GOOGLE_API_KEY
    )
except Exception as e:
    logging.error(f"Failed to initialize embedding function: {e}")
    embedding_function = None

def get_vector_store(path: str = config.VECTOR_DB_DIRECTORY,
                     collection_name: str = config.VECTOR_DB_COLLECTION) -> Chroma:
    """Initializes or loads a ChromaDB vector store."""
    if not embedding_function:
        raise ValueError("Embedding function not initialized.")

    # Using PersistentClient for saving to disk
    client = chromadb.PersistentClient(path=path)
    # Get or create the collection with the specified embedding function's metadata requirements if needed
    # Chroma automatically handles embedding function details for newer LangChain versions
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    logging.info(f"Initialized Chroma vector store at '{path}' with collection '{collection_name}'")
    return vector_store

def process_and_store_documents(docs: List[Document], vector_store: Chroma):
    """Chunks documents and adds them to the vector store."""
    # ... (No changes needed here for Phase 2, it accepts List[Document]) ...
    if not docs:
        logging.warning("No documents received for processing and storage.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    # Ensure metadata is preserved during splitting
    split_docs = text_splitter.split_documents(docs)

    if not split_docs:
        logging.warning(f"Splitting {len(docs)} documents resulted in zero chunks.")
        return

    logging.info(f"Adding {len(split_docs)} chunks to the vector store...")
    try:
        # Add IDs to potentially avoid duplicates if content is identical? Or rely on Chroma's handling.
        # ids = [f"{doc.metadata.get('source', 'unknown')}_{hash(doc.page_content)}" for doc in split_docs] # Example ID generation
        vector_store.add_documents(split_docs) # Chroma handles ID generation by default
        logging.info("Successfully added documents to vector store.")
    except Exception as e:
        logging.error(f"Error adding documents to vector store: {e}")

def create_basic_retriever(vector_store: Chroma, search_k: int = 5):
    """Creates a basic vector store retriever."""
    return vector_store.as_retriever(search_kwargs={"k": search_k})

def filter_documents_by_date(docs: List[Document], start_date_str: str) -> List[Document]:
    """
    Filters documents based on their 'publish_date' metadata *if* the source is 'newsapi'.
    Other sources (like 'website') are passed through without date filtering by this function.
    """
    if not docs: return []

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    filtered_docs = []
    news_count = 0
    news_kept_count = 0

    for doc in docs:
        source = doc.metadata.get('source')
        if source == 'newsapi':
            news_count += 1
            publish_date_str = doc.metadata.get('publish_date')
            if publish_date_str:
                try:
                    # Handle potential 'T'/'Z' in ISO format dates
                    if 'T' in publish_date_str:
                         publish_date = datetime.fromisoformat(publish_date_str.replace('Z', '+00:00')).replace(tzinfo=None) # Naive datetime
                    else:
                         publish_date = datetime.strptime(publish_date_str, '%Y-%m-%d') # Assume YYYY-MM-DD

                    if publish_date >= start_date:
                        filtered_docs.append(doc)
                        news_kept_count += 1
                except ValueError:
                    logging.warning(f"Could not parse news date '{publish_date_str}' for filtering.")
            else:
                 logging.warning("News document missing 'publish_date' metadata for filtering.")
        else:
            # Pass through documents from other sources (like 'website')
            filtered_docs.append(doc)

    logging.info(f"Date filtering applied to {news_count} news docs >= {start_date_str}. Kept {news_kept_count} news docs. Total docs after filter (incl. non-news): {len(filtered_docs)}.")
    return filtered_docs