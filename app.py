# app.py
import streamlit as st
import logging
import time
import config # Import config to ensure it's loaded
import ui
import agent
import retriever as db_retriever
import utils # Ensure utils logging is configured

# --- Page Configuration ---
st.set_page_config(
    page_title="Competitive Intelligence Reporter",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Automated Competitive Intelligence Reporter")
st.caption(f"Using {config.GEMINI_MODEL_NAME} and NewsAPI")

# --- Initialization ---
# Initialize components (only once per session ideally using Streamlit state)
# For simplicity in Phase 1, we initialize on each run, but know this isn't optimal
try:
    vector_store = db_retriever.get_vector_store()
except Exception as e:
    ui.display_error(f"Failed to initialize Vector Store: {e}")
    st.stop() # Stop execution if DB fails

if not agent.llm:
    ui.display_error("LLM Initialization Failed. Check API Key and configuration.")
    st.stop()
if not agent.newsapi:
    ui.display_error("NewsAPI Client Initialization Failed. Check API Key.")
    st.stop()


# --- UI Interaction ---
competitor_name, competitor_url, days_back, generate_button = ui.create_input_form()

if generate_button:
    # --- VALIDATE INPUTS ---
    if not competitor_name:
        ui.display_error("Please enter a competitor name.")
    # Optional: Add basic URL validation if needed
    # elif competitor_url and not competitor_url.startswith(('http://', 'https://')):
    #     ui.display_error("Please enter a valid URL (starting with http:// or https://).")
    # --- END OF VALIDATION ---
    else:
        with st.spinner(f"Generating report for {competitor_name}... Fetching news & scraping website..."):
            try:
                # --- 1. Collect Data (News + Web) ---
                logging.info(f"Button clicked. Starting analysis for '{competitor_name}', URL: '{competitor_url}', {days_back} days back.")

                # Collect News Data
                news_docs = agent.collect_news_data(competitor_name, days_back)
                if not news_docs:
                    ui.display_info(f"No recent news articles found for '{competitor_name}' via NewsAPI.")

                # Collect Web Data (if URL provided)
                web_docs = []
                if competitor_url:
                    # Optional: Add a small delay to avoid overwhelming sites/getting blocked
                    # time.sleep(1)
                    web_docs = agent.scrape_website_data(competitor_url)
                    if not web_docs:
                        ui.display_warning(f"Could not scrape or extract content from URL: {competitor_url}")
                else:
                    logging.info("No competitor URL provided, skipping web scraping.")

                all_docs = news_docs + web_docs

                if not all_docs:
                     ui.display_error(f"No information found for '{competitor_name}' from any source.")
                     st.stop() # Stop if absolutely nothing was found

                # --- 2. Process & Store Data ---
                logging.info(f"Processing {len(all_docs)} documents ({len(news_docs)} news, {len(web_docs)} web).")
                db_retriever.process_and_store_documents(all_docs, vector_store)
                # Add a small delay after storage if needed, ChromaDB writing might take a moment
                time.sleep(0.5)

                # --- 3. Retrieve Relevant Context ---
                logging.info("Retrieving combined context from vector store.")
                context_docs = agent.retrieve_and_filter_context(vector_store, competitor_name, days_back)

                # --- 4. Summarize Information ---
                logging.info("Generating final summary.")
                report = agent.summarize_information(context_docs, competitor_name)

                # --- 5. Display Report ---
                ui.display_report(report)
                logging.info("Report generation complete and displayed.")

            except Exception as e:
                logging.exception("An error occurred during report generation:")
                ui.display_error(f"An unexpected error occurred: {e}")
else:
    st.markdown("Enter competitor details in the sidebar to generate a report.")

