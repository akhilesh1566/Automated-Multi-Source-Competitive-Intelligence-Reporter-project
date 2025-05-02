# agent.py
from newsapi import NewsApiClient
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import logging
from langchain_community.document_loaders import WebBaseLoader
import config
import utils
import retriever as db_retriever # Use alias to avoid confusion

# Initialize NewsAPI client
try:
    newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)
except Exception as e:
    logging.error(f"Failed to initialize NewsAPI client: {e}")
    newsapi = None

# Initialize LLM globally (or pass it around)
try:
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL_NAME,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3, # Lower temperature for factual summary
        convert_system_message_to_human=True # Gemini API prefers this
    )
except Exception as e:
    logging.error(f"Failed to initialize Gemini LLM: {e}")
    llm = None


def collect_news_data(competitor_name: str, days_back: int) -> List[Document]:
    """Fetches news articles about the competitor from the last N days."""
    if not newsapi:
        logging.error("NewsAPI client not initialized.")
        return []
    if not competitor_name:
        logging.warning("No competitor name provided for news search.")
        return []

    start_date_str = utils.get_date_n_days_ago(days_back)
    logging.info(f"Fetching news for '{competitor_name}' from {start_date_str}...")

    try:
        all_articles = newsapi.get_everything(
            q=competitor_name,
            language='en',
            from_param=start_date_str,
            sort_by='relevancy', # Options: relevancy, popularity, publishedAt
            page_size=config.NEWS_API_MAX_RESULTS
        )
    except Exception as e:
        logging.error(f"Error fetching news from NewsAPI: {e}")
        return []

    docs = []
    if all_articles and all_articles['status'] == 'ok':
        logging.info(f"Received {len(all_articles['articles'])} articles from NewsAPI.")
        for article in all_articles['articles']:
            # Basic cleaning - can be expanded in utils
            content = utils.clean_text(article.get('content') or article.get('description') or "")
            if not content: # Skip articles with no usable content
                continue

            metadata = {
                "source": "newsapi",
                "competitor": competitor_name,
                "title": article.get('title', 'N/A'),
                "url": article.get('url', 'N/A'),
                "publish_date": article.get('publishedAt', 'N/A') # Keep original format for now
            }
            docs.append(Document(page_content=content, metadata=metadata))
    else:
        logging.warning(f"NewsAPI request failed or returned no articles. Status: {all_articles.get('status', 'N/A')}")

    logging.info(f"Created {len(docs)} Document objects from fetched news.")
    return docs

# --- ADD WEB SCRAPING FUNCTION ---
def scrape_website_data(url: str) -> List[Document]:
    """Scrapes text content from the main page of the given URL."""
    if not url:
        return []

    logging.info(f"Attempting to scrape website: {url}")
    try:
        # Using WebBaseLoader for simplicity. Handles fetching and basic parsing.
        # It might fetch multiple documents if the page has distinct sections.
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        loader = WebBaseLoader(
            web_path=url,
            header_template=headers,
            # You might need requests_per_second for rate limiting on some sites later
            # requests_per_second=1,
            # requests_kwargs={'verify': False} # Use cautiously if facing SSL issues
         )
        # Load the data
        web_docs = loader.load() # Returns a list of Documents

        # Add/update metadata for clarity
        fetch_date_str = datetime.now().strftime('%Y-%m-%d')
        for doc in web_docs:
            doc.metadata.setdefault('source', 'website') # Ensure source is set
            doc.metadata['fetch_date'] = fetch_date_str
            doc.metadata['url'] = url # Ensure original URL is stored

        logging.info(f"Successfully scraped {len(web_docs)} document sections from {url}")
        return web_docs
    except Exception as e:
        logging.error(f"Error scraping website {url}: {e}")
        return []

def retrieve_and_filter_context(vector_store, competitor_name: str, days_back: int) -> List[Document]:
    """Retrieves relevant documents from NEWS and WEBSITE sources and filters them."""
    start_date_str = utils.get_date_n_days_ago(days_back)
    retriever = db_retriever.create_basic_retriever(vector_store, search_k=15) # Increase K slightly

    # Modify query slightly to encompass both sources
    query = f"Recent news and website information about {competitor_name}"
    logging.info(f"Retrieving context for query: '{query}'")

    retrieved_docs = retriever.invoke(query)
    logging.info(f"Retrieved {len(retrieved_docs)} documents initially (News + Web).")

    # --- Modified Filtering Step ---
    # Filter only NEWS documents by publish date. Keep relevant web docs regardless of date.
    filtered_docs = db_retriever.filter_documents_by_date(
        [doc for doc in retrieved_docs if doc.metadata.get('source') == 'newsapi'],
        start_date_str
    )
    # Add back the web documents retrieved
    web_docs = [doc for doc in retrieved_docs if doc.metadata.get('source') == 'website']
    filtered_docs.extend(web_docs)
    # -------------------------

    # Optional: Limit number of docs sent to LLM after filtering/combining
    # Simple approach: just take the top N overall (similarity might mix sources)
    # More refined: ensure a mix if desired. Let's stick to simple top N for now.
    final_docs = filtered_docs[:7] # Take top 7 most relevant after filtering/combining
    logging.info(f"Providing {len(final_docs)} documents as context after filtering/combining.")
    return final_docs


def summarize_information(context_docs: List[Document], competitor_name: str) -> str:
    """Generates a summary using Gemini based on context from news and website data."""
    if not llm:
        return "Error: LLM not initialized."
    if not context_docs:
        return f"No relevant news or website content found for {competitor_name} in the specified timeframe to summarize."

    context_str = utils.format_docs_for_llm(context_docs)

    # --- Modify Prompt Slightly ---
    template = """
You are an AI assistant specialized in competitive intelligence analysis.
Your task is to provide a concise summary of the key news and developments about a competitor based *only* on the provided context from news articles and the competitor's website.
Focus on factual information like product launches, financial performance mentions, partnerships, leadership changes, website announcements, or significant market activities mentioned in the text.
Do not add any information not present in the context. Do not make assumptions or predictions. Clearly distinguish if information comes from news or the website if possible, otherwise synthesize concisely.

Competitor: {competitor}

Context Articles/Web Content:
{context}

Concise Summary of Key Developments:
"""
    # --- End of Modification ---

    prompt = PromptTemplate(template=template, input_variables=["competitor", "context"])
    summarization_chain = prompt | llm | StrOutputParser()

    logging.info(f"Generating summary for '{competitor_name}'...")
    try:
        summary = summarization_chain.invoke({
            "competitor": competitor_name,
            "context": context_str
        })
        logging.info("Summary generation complete.")
        return summary
    except Exception as e:
        logging.error(f"Error during summary generation: {e}")
        return f"Error generating summary: {e}"