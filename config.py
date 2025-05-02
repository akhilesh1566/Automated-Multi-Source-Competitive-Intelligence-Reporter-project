# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Model Names
# Using Flash for speed and cost-effectiveness in summarization tasks
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
# Google's latest embedding model as of mid-2024
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# Vector Store Configuration
VECTOR_DB_DIRECTORY = "./data"
VECTOR_DB_COLLECTION = "competitor_news"

# NewsAPI Configuration
NEWS_API_MAX_RESULTS = 20 # Max results to fetch per request