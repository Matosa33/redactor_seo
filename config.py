import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Searches for .env file in the current directory or parent directories
dotenv_path = Path('.') / '.env'
load_dotenv(dotenv_path=dotenv_path)

# --- Project Settings ---
# Base directory for storing project data
PROJECTS_BASE_DIR = Path(os.getenv("PROJECTS_BASE_DIR", "projects"))

# --- Cache Settings ---
# Base directory for caching SERP results
SERP_CACHE_DIR = Path(os.getenv("SERP_CACHE_DIR", "cache/serp"))
# TODO: Add cache expiry settings if needed (e.g., SERP_CACHE_EXPIRY_DAYS = 7)

# --- LLM Service Settings ---
# Load provider-specific API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Default model name (can be overridden by .env) - Keep a generic default if needed
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-pro") # Example default

# --- SERP Scraper Settings (Bright Data) ---
BRIGHTDATA_USERNAME = os.getenv("BRIGHTDATA_USERNAME")
BRIGHTDATA_PASSWORD = os.getenv("BRIGHTDATA_PASSWORD")
BRIGHTDATA_HOST = os.getenv("BRIGHTDATA_HOST") # Still needed if proxy is used elsewhere? Keep for now.
BRIGHTDATA_PORT = os.getenv("BRIGHTDATA_PORT") # Still needed if proxy is used elsewhere? Keep for now.
BRIGHTDATA_COUNTRY = os.getenv("BRIGHTDATA_COUNTRY", "fr") # Default country
SERP_CACHE_EXPIRY_SECONDS = int(os.getenv("SERP_CACHE_EXPIRY_SECONDS", 86400)) # Cache expiry in seconds (default 1 day)
# --- Bright Data Direct SERP API Settings ---
BRIGHTDATA_API_TOKEN = os.getenv("BRIGHTDATA_API_TOKEN")
BRIGHTDATA_SERP_ZONE_NAME = os.getenv("BRIGHTDATA_SERP_ZONE_NAME", "serp") # Default to 'serp' based on user info


# --- Application Settings ---
# Example: Set application-wide logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# --- Validation (Optional but recommended) ---
# Check if at least one LLM key is loaded
llm_keys_loaded = any([OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY])
if not llm_keys_loaded:
    print("Warning: No LLM API keys (OPENAI, ANTHROPIC, GOOGLE, DEEPSEEK) found in .env or environment.")

# Check Bright Data config completeness (Proxy vs Direct API)
brightdata_proxy_complete = all([BRIGHTDATA_USERNAME, BRIGHTDATA_PASSWORD, BRIGHTDATA_HOST, BRIGHTDATA_PORT])
brightdata_direct_api_complete = all([BRIGHTDATA_API_TOKEN, BRIGHTDATA_SERP_ZONE_NAME])

if not brightdata_proxy_complete and not brightdata_direct_api_complete:
     print("Warning: Incomplete BrightData configuration. Provide either Proxy credentials (USERNAME, PASSWORD, HOST, PORT) or Direct API credentials (API_TOKEN, SERP_ZONE_NAME).")
elif not brightdata_direct_api_complete:
     print("Warning: BrightData Direct API configuration incomplete (API_TOKEN or SERP_ZONE_NAME missing). SERP API calls might fail.")


# --- Helper Functions (Optional) ---
def get_projects_base_dir() -> Path:
    """Returns the configured base directory for projects."""
    PROJECTS_BASE_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists
    return PROJECTS_BASE_DIR

def get_serp_cache_dir() -> Path:
    """Returns the configured base directory for SERP cache."""
    SERP_CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists
    return SERP_CACHE_DIR

# You can add more functions or validation as needed

if __name__ == "__main__":
    # Print loaded configuration for verification when run directly
    print("--- Configuration Loaded ---")
    print(f"Projects Base Directory: {PROJECTS_BASE_DIR}")
    print(f"SERP Cache Directory: {SERP_CACHE_DIR}")
    print(f"OpenAI Key Loaded: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"Anthropic Key Loaded: {'Yes' if ANTHROPIC_API_KEY else 'No'}")
    print(f"Google Key Loaded: {'Yes' if GOOGLE_API_KEY else 'No'}")
    print(f"DeepSeek Key Loaded: {'Yes' if DEEPSEEK_API_KEY else 'No'}")
    print(f"Perplexity Key Loaded: {'Yes' if PERPLEXITY_API_KEY else 'No'}")
    print(f"Default LLM Model: {DEFAULT_LLM_MODEL}")
    print(f"BrightData Proxy Config Complete: {'Yes' if brightdata_proxy_complete else 'No'}")
    print(f"BrightData Direct API Config Complete: {'Yes' if brightdata_direct_api_complete else 'No'}")
    print(f"Log Level: {LOG_LEVEL}")
    print("--------------------------")
