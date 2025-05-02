# utils.py
from datetime import datetime, timedelta
import logging
import re 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_date_n_days_ago(days: int) -> str:
    """Returns the date N days ago in YYYY-MM-DD format."""
    date_n_days_ago = datetime.now() - timedelta(days=days)
    return date_n_days_ago.strftime('%Y-%m-%d')

def format_docs_for_llm(docs: list) -> str:
    """Formats a list of LangChain Document objects into a single string for LLM context."""
    # ... (keep existing code for this function) ...
    formatted_list = []
    for i, doc in enumerate(docs):
        # Limiting metadata displayed to avoid clutter, but keeping essential source/date
        metadata_str = f"Source: {doc.metadata.get('source', 'N/A')}, Date: {doc.metadata.get('publish_date', 'N/A')}"
        formatted_list.append(f"--- Document {i+1} ({metadata_str}) ---\n{doc.page_content}\n---")
    return "\n".join(formatted_list)


# --- ADD THIS FUNCTION ---
def clean_text(text: str) -> str:
    """
    Performs basic cleaning of text content fetched from NewsAPI or web scraping.
    - Removes common unwanted characters or sequences (like '[+ N chars]').
    - Replaces multiple whitespace characters with a single space.
    - Strips leading/trailing whitespace.
    """
    if not text:
        return ""

    # Remove common NewsAPI truncation indicator like "[+1234 chars]"
    text = re.sub(r'\[\+\d+\s*chars\]', '', text)

    # Replace multiple whitespace characters (space, tab, newline) with a single space
    cleaned = re.sub(r'\s+', ' ', text).strip()

    return cleaned