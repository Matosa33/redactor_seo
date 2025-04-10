import streamlit as st
import os
import json
import time
import asyncio
from datetime import datetime
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Coroutine # Added typing imports
import re # Import re for query generation parsing

# Explicitly load .env at the very beginning
from dotenv import load_dotenv
dotenv_path = Path('.') / '.env'
load_dotenv(dotenv_path=dotenv_path)
print(f"Attempted to load .env from: {dotenv_path.resolve()}") # Add print statement for verification

# Import configuration and new modules/core objects
import config
from core.project import Project
from core.keyword import Keyword
from core.article_plan import ArticlePlan
from core.generated_article import GeneratedArticle
from modules.llm_service import LLMService, QuotaError # Import exception
from modules.content_engine import ContentEngine
from modules.data_manager import DataManager
from modules.batch_processor import BatchProcessor, BatchStatus, ProcessItemFunc, ProgressCallback # Import Enum and types
from modules.serp_scraper import SERPScraper
from modules.research_engine import ResearchEngine, ResearchResult # Import new engine
from modules.web_scraper import scrape_urls_content # Import the new web scraper function
from modules.markdown_utils import plan_to_markdown, markdown_to_html, markdown_to_docx, prepare_for_wordpress # Import utils

# Configure logging using level from config
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(
    page_title="SEO Content Generator Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Service Initialization ---
@st.cache_resource # Re-enable caching
def initialize_services():
    """Initializes and caches application services."""
    logger.info("Initializing services...") # Revert log message

    # Load API keys from config module
    api_keys = {
        "openai": config.OPENAI_API_KEY,
        "anthropic": config.ANTHROPIC_API_KEY,
        "google": config.GOOGLE_API_KEY,
        "deepseek": config.DEEPSEEK_API_KEY,
    }
    # Filter out None or empty keys before passing to LLMService
    api_keys_filtered = {k: v for k, v in api_keys.items() if v}
    logger.info(f"Loaded API keys for: {list(api_keys_filtered.keys())}")

    # Store loaded keys (even if None) in session state for UI display
    st.session_state.api_keys = api_keys # Store original dict with potential None values

    # Initialize LLM service with filtered keys
    llm_service = LLMService(api_keys=api_keys_filtered)

    # SERP Scraper - Initialize for Proxy Access Method
    proxy_config = None
    # Load Bright Data proxy credentials directly from config module
    brightdata_user = config.BRIGHTDATA_USERNAME
    brightdata_pass = config.BRIGHTDATA_PASSWORD
    brightdata_host = config.BRIGHTDATA_HOST
    brightdata_port = config.BRIGHTDATA_PORT

    # ADDED MORE DETAILED LOGGING: Check individual values and types
    logger.debug(f"BRIGHTDATA_USERNAME: Type={type(brightdata_user)}, Value='{brightdata_user}'")
    logger.debug(f"BRIGHTDATA_PASSWORD: Type={type(brightdata_pass)}, Exists='{bool(brightdata_pass)}'") # Don't log password value
    logger.debug(f"BRIGHTDATA_HOST: Type={type(brightdata_host)}, Value='{brightdata_host}'")
    logger.debug(f"BRIGHTDATA_PORT: Type={type(brightdata_port)}, Value='{brightdata_port}'")

    if all([brightdata_user, brightdata_pass, brightdata_host, brightdata_port]):
         proxy_config = {
             'username': brightdata_user,
             'password': brightdata_pass,
             'host': brightdata_host,
             'port': brightdata_port
         }
         logger.info("Bright Data Proxy credentials loaded successfully.")
    else:
        # Log specific missing variables if possible
        missing_vars = [var for var, val in [('USERNAME', brightdata_user), ('PASSWORD', brightdata_pass), ('HOST', brightdata_host), ('PORT', brightdata_port)] if not val]
        logger.warning(f"Incomplete BrightData proxy config. Missing: {', '.join(missing_vars)}. SERP API via proxy will fail.")

    # Pass the potentially populated proxy_config to SERPScraper
    serp_scraper = SERPScraper(
        cache_dir=config.SERP_CACHE_DIR,
        proxy_config=proxy_config, # Pass the config dict
        country=config.BRIGHTDATA_COUNTRY # Pass country explicitly or let scraper handle default
    )

    # Research Engine needs LLM service and optional Perplexity key from config
    research_engine = ResearchEngine(
        llm_service=llm_service,
        perplexity_api_key=config.PERPLEXITY_API_KEY # Load from config
    )

    # Other services
    data_manager = DataManager(base_directory=config.PROJECTS_BASE_DIR)
    content_engine = ContentEngine(llm_service=llm_service)
    batch_processor = BatchProcessor(max_concurrent_tasks=3) # Example concurrency limit

    logger.info("Services initialized successfully.")
    return {
        'data_manager': data_manager,
        'llm_service': llm_service,
        'content_engine': content_engine,
        'batch_processor': batch_processor,
        'serp_scraper': serp_scraper,
        'research_engine': research_engine
    }

# Get services (cached)
services = initialize_services()

# --- Initialize Streamlit Session State ---
# Use direct st.session_state access

# Core application state
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'current_keyword_obj' not in st.session_state:
    st.session_state.current_keyword_obj = None
if 'current_plan_obj' not in st.session_state:
    st.session_state.current_plan_obj = None
if 'current_article_obj' not in st.session_state:
    st.session_state.current_article_obj = None
# Removed current_tab initialization as st.tabs handles it

# UI related state
if 'plan_generated' not in st.session_state:
    st.session_state.plan_generated = False
if 'article_generated' not in st.session_state:
    st.session_state.article_generated = False
if 'competitors_info_input' not in st.session_state:
     st.session_state.competitors_info_input = ""
if 'current_keyword_input' not in st.session_state:
     st.session_state.current_keyword_input = ""
if 'api_keys' not in st.session_state: # Ensure api_keys is initialized
     st.session_state.api_keys = {}
if 'last_serp_results' not in st.session_state:
     st.session_state.last_serp_results = None
if 'last_research_results' not in st.session_state:
     st.session_state.last_research_results = None


# Model settings persistence
# Initialize with defaults, trying to load from config/.env first
# Use getattr to safely access config attributes with fallbacks
default_plan_settings = {
    "provider": getattr(config, 'PLAN_PROVIDER', "Google"), # Default to Google if not set
    "model": getattr(config, 'PLAN_MODEL', "models/gemini-1.5-flash-latest"), # Default to flash
    "temperature": float(getattr(config, 'DEFAULT_TEMPERATURE', 0.7)),
    "max_tokens": int(getattr(config, 'DEFAULT_MAX_TOKENS', 2048))
}
default_article_settings = {
    "provider": getattr(config, 'WRITING_PROVIDER', "Anthropic"), # Default to Anthropic
    "model": getattr(config, 'WRITING_MODEL', "claude-3-haiku-20240307"), # Default to Haiku
    "temperature": float(getattr(config, 'DEFAULT_TEMPERATURE', 0.7)),
    "max_tokens": int(getattr(config, 'DEFAULT_MAX_TOKENS', 4000)) # Allow more tokens for articles
}
default_refine_settings = {
    "provider": getattr(config, 'REFINE_PROVIDER', "OpenAI"), # Default to OpenAI
    "model": getattr(config, 'REFINE_MODEL', "gpt-3.5-turbo"), # Default to 3.5-turbo
    "temperature": 0.5, # Keep specific refine temp
    "max_tokens": 1024 # Keep specific refine tokens
}


if 'plan_model_settings' not in st.session_state:
    st.session_state.plan_model_settings = default_plan_settings.copy()
if 'article_model_settings' not in st.session_state:
    st.session_state.article_model_settings = default_article_settings.copy()
if 'refine_model_settings' not in st.session_state:
    st.session_state.refine_model_settings = default_refine_settings.copy()

# Batch processing state
if 'batch_processor_instance' not in st.session_state: # Store the instance itself
     st.session_state.batch_processor_instance = services['batch_processor']
if 'batch_status_info' not in st.session_state:
     st.session_state.batch_status_info = services['batch_processor'].get_progress() # Get initial state

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")

    # --- Project Selection ---
    st.header("Projet")
    data_manager: DataManager = services['data_manager']
    try:
        projects = data_manager.list_projects() # Returns list of Project objects
        project_names = sorted([p.name for p in projects])
    except Exception as e:
        st.error(f"Erreur lors de la liste des projets: {e}")
        projects = []
        project_names = []

    current_project_name = st.session_state.current_project.name if st.session_state.current_project else None
    default_index = 0
    if current_project_name and current_project_name in project_names:
        try:
            default_index = project_names.index(current_project_name) + 1
        except ValueError:
            default_index = 0

    selected_project_option = st.selectbox(
        "Projet Actuel",
        ["Nouveau Projet"] + project_names,
        index=default_index,
        key="project_selector"
    )

    if selected_project_option == "Nouveau Projet":
        if st.session_state.current_project is not None:
             logger.info("Switched to New Project mode.")
             st.session_state.current_project = None
             st.session_state.current_keyword_obj = None
             st.session_state.current_plan_obj = None
             st.session_state.current_article_obj = None
             st.session_state.plan_generated = False
             st.session_state.article_generated = False
             st.session_state.current_keyword_input = ""

        new_project_name = st.text_input("Nom du Nouveau Projet", key="new_project_name_input")
        if st.button("Créer et Charger Projet"):
            if not new_project_name:
                st.error("Veuillez entrer un nom pour le nouveau projet.")
            elif new_project_name in project_names:
                 st.error(f"Un projet nommé '{new_project_name}' existe déjà.")
            else:
                with st.spinner(f"Création du projet '{new_project_name}'..."):
                    created_project = data_manager.create_project(new_project_name)
                    if created_project:
                        st.session_state.current_project = created_project
                        st.session_state.current_keyword_obj = None
                        st.session_state.current_plan_obj = None
                        st.session_state.current_article_obj = None
                        st.session_state.plan_generated = False
                        st.session_state.article_generated = False
                        st.session_state.current_keyword_input = ""
                        st.success(f"Projet '{created_project.name}' créé et chargé.")
                        st.rerun()
                    else:
                        st.error(f"Échec de la création du projet '{new_project_name}'. Vérifiez les logs.")

    elif selected_project_option != current_project_name:
        with st.spinner(f"Chargement du projet '{selected_project_option}'..."):
            loaded_project = data_manager.load_project(selected_project_option)
            if loaded_project:
                st.session_state.current_project = loaded_project
                st.session_state.current_keyword_obj = None
                st.session_state.current_plan_obj = None
                st.session_state.current_article_obj = None
                st.session_state.plan_generated = False
                st.session_state.article_generated = False
                st.session_state.current_keyword_input = ""
                logger.info(f"Project '{loaded_project.name}' loaded.")
                st.success(f"Projet '{loaded_project.name}' chargé.")
                st.rerun()
            else:
                st.error(f"Échec du chargement du projet '{selected_project_option}'.")
                st.session_state.current_project = None

    if st.session_state.current_project:
        st.info(f"Projet Actif: **{st.session_state.current_project.name}**")
    else:
         st.warning("Aucun projet chargé. Créez ou sélectionnez un projet.")

    with st.expander("🤖 Configuration des modèles", expanded=False):
        st.info("Configurez les modèles LLM par défaut pour chaque étape.")
        st.write("Configuration des modèles par défaut (à implémenter)")

    with st.expander("🔑 Clés API", expanded=False):
        st.info("Les clés API sont chargées depuis votre fichier `.env` ou les variables d'environnement.")
        loaded_keys_display = {k: f"{v[:4]}...{v[-4:]}" if v and len(v) > 8 else "Non définie" for k, v in st.session_state.api_keys.items()}
        st.json(loaded_keys_display)
        st.markdown("Pour mettre à jour, modifiez votre fichier `.env` et redémarrez l'application.")

    batch_processor_instance: BatchProcessor = st.session_state.batch_processor_instance
    st.session_state.batch_status_info = batch_processor_instance.get_progress()
    batch_status = st.session_state.batch_status_info["status"]

    if batch_status != BatchStatus.IDLE.value:
         st.sidebar.subheader("📦 Statut Traitement par Lots")
         st.sidebar.info(f"Statut: {batch_status}")
         if batch_status in [BatchStatus.RUNNING.value, BatchStatus.PAUSED.value]:
              progress_val = st.session_state.batch_status_info['processed_items'] / st.session_state.batch_status_info['total_items'] if st.session_state.batch_status_info['total_items'] > 0 else 0
              st.sidebar.progress(progress_val)
              st.sidebar.write(f"{st.session_state.batch_status_info['processed_items']} / {st.session_state.batch_status_info['total_items']} traités")

# --- Custom CSS ---
st.markdown("""<style>/* ... CSS styles ... */</style>""", unsafe_allow_html=True) # Keep CSS collapsed

# --- UI Components ---
def notification(message, type="info"):
    st.markdown(f'<div class="notification {type}">{message}</div>', unsafe_allow_html=True)

def model_selector(key_prefix: str, step_name: str, current_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Component for selecting an LLM model with provider-specific options."""
    llm: LLMService = services['llm_service']
    available_providers = llm.get_available_providers() # Returns correctly capitalized names
    logger.debug(f"[{key_prefix}] Available providers from LLMService: {available_providers}") # Add logging

    if not available_providers:
        st.warning("Aucun fournisseur LLM configuré avec clé API.") # Updated warning
        return current_settings

    provider_display_map = {
        "OpenAI": "OpenAI (GPT)", "Anthropic": "Anthropic (Claude)",
        "Google": "Google (Gemini)", "DeepSeek": "DeepSeek" # Use consistent capitalization
    }
    # Filter display map based on actual available providers
    available_providers_display_map = {p: provider_display_map.get(p, p) for p in available_providers}
    available_providers_display_list = list(available_providers_display_map.values())
    logger.debug(f"[{key_prefix}] Display map filtered: {available_providers_display_map}")
    logger.debug(f"[{key_prefix}] Display list for radio: {available_providers_display_list}")


    if not available_providers_display_list:
         st.warning("No display names found for available providers.")
         return current_settings

    # Determine default provider display name based on current_settings or first available
    default_provider_internal = current_settings.get("provider", available_providers[0])
    default_provider_display = available_providers_display_map.get(default_provider_internal, available_providers_display_list[0])
    logger.debug(f"[{key_prefix}] Default provider internal: {default_provider_internal}, display: {default_provider_display}")


    # Get the default model name from settings or config
    default_model_name = current_settings.get("model", config.DEFAULT_LLM_MODEL) # Use config default
    logger.debug(f"[{key_prefix}] Default model name: {default_model_name}")


    with st.container():
        st.markdown(f"#### Modèle pour {step_name}")
        show_advanced = st.checkbox("Afficher les options avancées", value=False, key=f"{key_prefix}_advanced")

        if show_advanced:
            # Use the filtered display map for the radio button
            try:
                # Ensure default_provider_display is actually in the list before finding index
                if default_provider_display not in available_providers_display_list:
                    logger.warning(f"[{key_prefix}] Default display name '{default_provider_display}' not in list {available_providers_display_list}. Defaulting index to 0.")
                    default_radio_index = 0
                else:
                    default_radio_index = available_providers_display_list.index(default_provider_display)
            except ValueError: # Should not happen if check above works, but as safety
                logger.error(f"[{key_prefix}] Error finding index for '{default_provider_display}' in {available_providers_display_list}. Defaulting index to 0.")
                default_radio_index = 0

            selected_provider_display = st.radio(
                "Fournisseur", available_providers_display_list,
                index=default_radio_index, # Use corrected index
                key=f"{key_prefix}_provider"
            )
            # Reverse map to get internal name (keys are display names, values are internal names)
            provider_internal_map_rev = {v: k for k, v in available_providers_display_map.items()}
            selected_provider = provider_internal_map_rev.get(selected_provider_display, available_providers[0]) # Fallback to first internal name
            logger.debug(f"[{key_prefix}] Selected provider display: {selected_provider_display}, internal: {selected_provider}")


            # Dynamically get models for the selected provider
            logger.debug(f"Fetching models for selected provider: {selected_provider}")
            provider_models = llm.get_models_for_provider(selected_provider)
            logger.debug(f"Models received: {provider_models}")

            if not provider_models:
                 st.warning(f"Impossible de récupérer les modèles pour {selected_provider_display}. Utilisation du modèle par défaut.")
                 provider_models = [default_model_name] # Fallback

            # Determine default model index
            current_model_in_settings = current_settings.get("model", default_model_name)
            try:
                 # Check if the current model is valid for the *selected* provider
                 if current_model_in_settings not in provider_models:
                      logger.warning(f"Model '{current_model_in_settings}' from settings not valid for selected provider {selected_provider}. Defaulting to first available model.")
                      current_model_in_settings = provider_models[0] if provider_models else default_model_name # Ensure fallback exists
                 model_index = provider_models.index(current_model_in_settings)
            except ValueError:
                 logger.warning(f"Model '{current_model_in_settings}' not found in fetched list {provider_models} for {selected_provider}. Defaulting to index 0.")
                 model_index = 0
                 current_model_in_settings = provider_models[0] if provider_models else default_model_name # Ensure fallback exists

            selected_model = st.selectbox(
                "Modèle", provider_models,
                index=model_index,
                key=f"{key_prefix}_model"
            )
            logger.debug(f"[{key_prefix}] Selected model: {selected_model}")


            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Température", 0.0, 1.0, current_settings.get("temperature", 0.7), 0.1, key=f"{key_prefix}_temperature")
            with col2:
                max_tokens = st.number_input("Tokens Max", 100, 128000, current_settings.get("max_tokens", 4000), 100, key=f"{key_prefix}_max_tokens")
        else:
            # Display the default provider and model correctly
            st.info(f"Modèle par défaut utilisé: {default_provider_display} / {default_model_name}")
            selected_provider = default_provider_internal
            selected_model = default_model_name
            temperature = current_settings.get("temperature", 0.7)
            max_tokens = current_settings.get("max_tokens", 4000)

        # Return the selected settings
        new_settings = {
            "provider": selected_provider, "model": selected_model,
            "temperature": temperature, "max_tokens": max_tokens
        }
        # Update the session state immediately
        st.session_state[f"{key_prefix}_model_settings"] = new_settings
        logger.debug(f"[{key_prefix}] Updated model settings in session state: {new_settings}")
        return new_settings


# --- Tab Setup Functions ---

def setup_plan_generator_tab():
    """Sets up the UI and logic for the Plan Generator tab."""
    st.header("Générateur de Plan SEO")
    data_manager: DataManager = services['data_manager']
    content_engine: ContentEngine = services['content_engine']
    serp_scraper: SERPScraper = services['serp_scraper']
    research_engine: ResearchEngine = services['research_engine']
    llm_service: LLMService = services['llm_service'] # Get LLM service for query gen

    # --- Keyword Input and Selection ---
    st.subheader("1. Sélection du Mot-Clé")
    current_project: Optional[Project] = st.session_state.current_project

    if not current_project:
        st.warning("Veuillez d'abord créer ou charger un projet dans la barre latérale.")
        return

    existing_keyword_objs = data_manager.list_keywords(current_project)
    existing_keyword_names = sorted([kw.name for kw in existing_keyword_objs])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.session_state.current_keyword_input = st.text_input(
            "Mot-clé principal", value=st.session_state.current_keyword_input,
            key="keyword_text_input", help="Le mot-clé principal pour lequel générer un plan."
        )
        keyword_name_input = st.session_state.current_keyword_input.strip()

    with col2:
        if existing_keyword_names:
            st.write("Ou sélectionnez un mot-clé existant:")
            options = ["-- Sélectionner --"] + existing_keyword_names
            current_input_index = 0
            if keyword_name_input in existing_keyword_names:
                 try: current_input_index = options.index(keyword_name_input)
                 except ValueError: pass

            selected_existing_kw_name = st.selectbox(
                "Mots-clés du projet", options, index=current_input_index,
                key="existing_keyword_selector"
            )

            if selected_existing_kw_name != "-- Sélectionner --":
                 if st.session_state.current_keyword_input != selected_existing_kw_name:
                      st.session_state.current_keyword_input = selected_existing_kw_name
                      keyword_name_input = selected_existing_kw_name
                      st.rerun()

    # Determine active keyword object and load data if changed
    active_keyword_obj: Optional[Keyword] = None
    if keyword_name_input:
         active_keyword_obj = data_manager.get_keyword(current_project, keyword_name_input)
         if keyword_name_input in existing_keyword_names:
              active_keyword_obj = next((kw for kw in existing_keyword_objs if kw.name == keyword_name_input), active_keyword_obj)
              if st.session_state.current_keyword_obj != active_keyword_obj:
                   with st.spinner(f"Chargement des données pour '{keyword_name_input}'..."):
                        st.session_state.current_keyword_obj = active_keyword_obj
                        st.session_state.current_plan_obj = data_manager.load_plan(active_keyword_obj)
                        st.session_state.current_article_obj = data_manager.load_article(active_keyword_obj)
                        st.session_state.plan_generated = st.session_state.current_plan_obj is not None
                        st.session_state.article_generated = st.session_state.current_article_obj is not None
                        st.success(f"Données chargées pour '{keyword_name_input}'.")
                        st.rerun() # Rerun to update display after loading
         else: # New keyword entered
              if st.session_state.current_keyword_obj is None or st.session_state.current_keyword_obj.name != keyword_name_input:
                   logger.info(f"Switched to new keyword: {keyword_name_input}")
                   st.session_state.current_keyword_obj = active_keyword_obj
                   st.session_state.current_plan_obj = None
                   st.session_state.current_article_obj = None
                   st.session_state.plan_generated = False
                   st.session_state.article_generated = False

    # --- Generation Parameters ---
    st.subheader("2. Paramètres de Génération")
    # Use the model_selector component, which now updates session state internally
    model_selector("plan", "Plan SEO", st.session_state.plan_model_settings)

    with st.form("plan_generation_form"):
        st.markdown("Options de contenu:")
        col1, col2 = st.columns(2)
        with col1: use_serp_analysis = st.checkbox("Analyse SERP Google", value=True, key="plan_use_serp")
        with col2: include_h3 = st.checkbox("Inclure H3", value=True, key="plan_include_h3")
        with st.expander("🔍 Recherche Approfondie (Optionnel)"):
            use_advanced_research = st.checkbox("Activer la recherche approfondie", value=False, key="plan_use_research")
        competitors_info = st.text_area("Informations manuelles sur les concurrents (optionnel)", value=st.session_state.competitors_info_input, key="competitors_info_input_area", height=100)
        st.session_state.competitors_info_input = competitors_info
        submit_button = st.form_submit_button("🚀 Générer le Plan")

    # --- Plan Generation Logic ---
    if submit_button:
        if not active_keyword_obj:
            st.error("Veuillez entrer ou sélectionner un mot-clé.")
        else:
            keyword_name = active_keyword_obj.name
            st.session_state.plan_generated = False
            st.session_state.current_plan_obj = None
            st.session_state.last_serp_results = None
            st.session_state.last_research_results = None

            serp_summary = "Analyse SERP non demandée."
            research_summary = "Recherche approfondie non demandée."
            serp_results_list = None
            research_results_list = []

            async def run_plan_pipeline():
                local_serp_summary = "Analyse SERP non demandée."
                local_research_summary = "Recherche approfondie non demandée."
                local_serp_results_list = None # Will hold list of dicts from SERPScraper
                local_research_results_list = []
                scraped_content_summary = "Aucun contenu de concurrent scrappé."

                if use_serp_analysis:
                    try:
                        logger.info(f"Starting SERP analysis for '{keyword_name}'...")
                        # SERPScraper now returns the standardized dict or None
                        serp_response = await serp_scraper.get_serp_results(keyword_name)
                        # Check for the correct key 'organic_results'
                        if serp_response and isinstance(serp_response.get('organic_results'), list):
                            local_serp_results_list = serp_response['organic_results'] # Extract the list using the correct key
                            local_serp_summary = f"Analyse SERP pour '{keyword_name}':\n- {len(local_serp_results_list)} résultats organiques trouvés.\nTitres principaux:\n"
                            # Use the correct key 'meta_title' from the scraper's output structure
                            for res in local_serp_results_list[:5]: local_serp_summary += f"  - {res.get('meta_title', 'N/A')}\n"
                            logger.info(f"Analyse SERP terminée ({len(local_serp_results_list)} résultats).")
                        else:
                            logger.warning("Aucun résultat SERP trouvé ou extraction échouée.")
                            local_serp_summary = "Analyse SERP demandée mais aucun résultat trouvé."
                    except Exception as e:
                        logger.error(f"Erreur lors de l'analyse SERP: {e}", exc_info=True)
                        local_serp_summary = f"Erreur lors de l'analyse SERP: {e}"
                    # --- Scrape content from top SERP results ---
                    if local_serp_results_list:
                        # Use the correct key 'url' from the scraper's output structure
                        top_urls = [res.get('url') for res in local_serp_results_list[:5] if res.get('url')] # Get top 3 URLs
                        if top_urls:
                            logger.info(f"Scraping content from top {len(top_urls)} URLs: {top_urls}")
                            scraped_data = await scrape_urls_content(top_urls)
                            if scraped_data:
                                scraped_content_summary = "\n\n# CONTENU DES CONCURRENTS (Extraits)\n"
                                for url, content in scraped_data.items():
                                     scraped_content_summary += f"## Source: {url}\n{content}\n\n"
                                logger.info(f"Successfully scraped content for {len(scraped_data)} URLs.")
                            else:
                                logger.warning("Web scraping yielded no content.")
                                scraped_content_summary = "Scraping des concurrents n'a retourné aucun contenu."
                        else:
                            logger.info("No valid URLs found in SERP results to scrape.")
                            scraped_content_summary = "Aucune URL valide trouvée dans la SERP pour le scraping."

                # --- Generate Research Queries Dynamically ---
                queries_to_research = []
                if use_advanced_research:
                    logger.info(f"Generating research queries for '{keyword_name}'...")
                    # Combine existing info for better query generation context
                    query_gen_context = f"Keyword: {keyword_name}\nSERP Summary: {local_serp_summary}\nCompetitor Content Snippets:\n{scraped_content_summary[:20000]}..." # Limit context length
                    query_gen_prompt = f"""
                    Based on the following context about the main keyword, generate a JSON list of 3-5 specific, diverse, and insightful search queries for Perplexity API to gather the latest information needed for a comprehensive article. Focus on recent trends, statistics, expert opinions, specific sub-topics, challenges, or solutions not already obvious from the context.

                    Context:
                    {query_gen_context}

                    Return ONLY a valid JSON list of strings. Example:
                    ["latest statistics {keyword_name} 2025", "expert predictions {keyword_name} future trends", "challenges implementing {keyword_name} solutions"]
                    """
                    try:
                        # Use a fast/cheap model for query generation
                        query_gen_model_settings = {"provider": "Google", "model": "models/gemini-1.5-flash-latest", "temperature": 0.6}
                        query_response_dict = await llm_service.generate_content(
                            prompt=query_gen_prompt, **query_gen_model_settings
                        )
                        query_response_content = query_response_dict.get("content", "")
                        if isinstance(query_response_content, str) and not query_response_content.startswith("Error:"):
                            try:
                                # Improved JSON extraction
                                json_match = re.search(r'```json\s*(\[.*?\])\s*```', query_response_content, re.DOTALL | re.IGNORECASE)
                                if json_match:
                                     json_str = json_match.group(1).strip()
                                else:
                                     # Try finding a list directly if no markdown block
                                     list_match = re.search(r'(\[.*?\])', query_response_content, re.DOTALL)
                                     json_str = list_match.group(1).strip() if list_match else query_response_content.strip()

                                generated_queries = json.loads(json_str)
                                if isinstance(generated_queries, list) and all(isinstance(q, str) for q in generated_queries):
                                    queries_to_research = generated_queries[:5] # Limit to 5 queries
                                    logger.info(f"Dynamically generated research queries: {queries_to_research}")
                                else:
                                     logger.warning("LLM did not return a valid JSON list for research queries.")
                                     queries_to_research = [] # Reset if invalid format
                            except json.JSONDecodeError:
                                 logger.warning(f"Failed to parse JSON for research queries: {query_response_content}")
                                 queries_to_research = [] # Reset on parse error
                            except Exception as query_e:
                                 logger.error(f"Error processing generated research queries: {query_e}")
                                 queries_to_research = [] # Reset on other errors
                        else:
                             logger.warning(f"LLM failed to generate research queries: {query_response_content}")
                             queries_to_research = [] # Reset if LLM failed

                    except Exception as e:
                        logger.error(f"Error during research query generation: {e}")
                        queries_to_research = [] # Reset on exception

                    # Fallback if dynamic generation fails or returns empty
                    if not queries_to_research:
                         logger.warning("Falling back to hardcoded research queries.")
                         queries_to_research = [f"tendances récentes {keyword_name} {datetime.now().year}", f"statistiques clés {keyword_name}", f"défis et solutions {keyword_name}"]

                # --- Perform Advanced Research ---
                if use_advanced_research and queries_to_research: # Only run if enabled AND queries exist
                    logger.info(f"Lancement de la recherche approfondie pour {len(queries_to_research)} requêtes...")
                    try:
                        local_research_results_list = await research_engine.perform_research(
                            queries=queries_to_research,
                            model_settings=st.session_state.plan_model_settings # Use settings from state
                        )
                        successful_searches = [r for r in local_research_results_list if r.success and not r.content.startswith("Error:")] # Exclude errors
                        failed_searches = [r for r in local_research_results_list if not r.success or r.content.startswith("Error:")]
                        local_research_summary = f"Résumé de la recherche approfondie pour '{keyword_name}':\n- {len(successful_searches)} recherches réussies.\n"
                        for i, res in enumerate(successful_searches[:5]): # Show more results
                             local_research_summary += f"  - Requête '{res.query[:40]}...': {res.content[:1500]}...\n"
                        if failed_searches:
                             local_research_summary += f"\n- {len(failed_searches)} recherches échouées ou bloquées.\n"
                             for res in failed_searches[:2]: local_research_summary += f"  - Échec requête '{res.query[:40]}...': {res.error or res.content}\n"
                        if not successful_searches and not failed_searches:
                             local_research_summary = "Recherche approfondie demandée mais aucun résultat obtenu."
                        logger.info(f"Recherche approfondie terminée ({len(successful_searches)} succès, {len(failed_searches)} échecs).")
                    except Exception as e:
                        logger.error(f"Erreur lors de la recherche approfondie: {e}", exc_info=True)
                        local_research_summary = f"Erreur lors de la recherche approfondie: {e}"

                logger.info("Génération du plan avec le LLM...")
                plan_gen_params = {"h2_count": 6, "include_h3": include_h3}
                combined_competitor_analysis = f"{local_serp_summary}\n{scraped_content_summary}"

                # Pass the correct model settings from session state
                generated_plan_obj_local = await content_engine.generate_plan(
                    keyword=keyword_name, model_settings=st.session_state.plan_model_settings, # Use settings from state
                    competitor_analysis_summary=combined_competitor_analysis,
                    research_summary=local_research_summary,
                    generation_params=plan_gen_params
                )
                # Return the original SERP list (list of dicts), not the summary string
                return generated_plan_obj_local, local_serp_summary, local_research_summary, local_serp_results_list, local_research_results_list

            # Run the pipeline
            with st.spinner("Génération du plan (SERP, Recherche, LLM)..."):
                try:
                    generated_plan_obj, serp_summary, research_summary, serp_results_list, research_results_list = asyncio.run(run_plan_pipeline())

                    if "Erreur" in serp_summary: st.error(serp_summary)
                    elif use_serp_analysis: st.success(f"Analyse SERP terminée ({len(serp_results_list or [])} résultats).")
                    if "Erreur" in research_summary: st.error(research_summary)
                    elif use_advanced_research: st.success(f"Recherche approfondie terminée ({len([r for r in research_results_list if r.success and not r.content.startswith('Error:')])} succès).")

                    st.session_state.current_plan_obj = generated_plan_obj
                    st.session_state.plan_generated = generated_plan_obj is not None # Check if plan object was created
                    st.session_state.last_serp_results = serp_results_list
                    st.session_state.last_research_results = research_results_list

                    if generated_plan_obj: # Only save if generation was successful
                        save_success = data_manager.save_plan(active_keyword_obj, generated_plan_obj)
                        if save_success: st.success("Plan généré et sauvegardé avec succès!")
                        else: st.warning("Plan généré mais échec de la sauvegarde.")
                    else:
                         st.error("La génération du plan a échoué. Vérifiez les logs pour plus de détails.")


                    st.rerun()

                except Exception as e:
                    logger.error(f"Erreur lors de l'exécution du pipeline de génération de plan: {e}", exc_info=True)
                    st.error(f"Erreur majeure lors de la génération du plan: {e}")
                    st.session_state.plan_generated = False
                    st.session_state.current_plan_obj = None

    # --- Display Generated Plan ---
    if st.session_state.plan_generated and st.session_state.current_plan_obj:
        st.subheader("3. Plan Généré")
        plan_obj = st.session_state.current_plan_obj
        keyword_obj = st.session_state.current_keyword_obj

        if not keyword_obj:
             st.error("Erreur interne: Objet mot-clé non défini alors que le plan est généré.")
             return

        col1, col2 = st.columns([3, 1])
        with col1:
            try:
                edited_plan_str = st.text_area(
                    "Éditer le plan (JSON)",
                    value=json.dumps(plan_obj.data, indent=2, ensure_ascii=False),
                    height=400,
                    key=f"edit_plan_{keyword_obj.name}"
                )
            except Exception as e:
                 st.error(f"Erreur d'affichage du plan JSON: {e}")
                 edited_plan_str = "{}"

        with col2:
            st.subheader("Actions")
            if st.button("Mettre à jour le plan", key="update_plan_button"):
                if not keyword_obj:
                     st.error("Erreur: Mot-clé non défini pour la sauvegarde.")
                else:
                    try:
                        updated_plan_data = json.loads(edited_plan_str)
                        updated_plan_obj = ArticlePlan(data=updated_plan_data)
                        st.session_state.current_plan_obj = updated_plan_obj

                        save_success = data_manager.save_plan(keyword_obj, updated_plan_obj)
                        if save_success: st.success("Plan mis à jour et sauvegardé!")
                        else: st.warning("Plan mis à jour mais échec de la sauvegarde.")
                    except json.JSONDecodeError: st.error("Format JSON invalide.")
                    except Exception as e: st.error(f"Erreur lors de la mise à jour: {e}")

            try:
                plan_md = plan_to_markdown(plan_obj.data)
                plan_json_export = json.dumps(plan_obj.data, indent=2, ensure_ascii=False)
                plan_html_export = markdown_to_html(plan_md)
                st.download_button("Exporter JSON", plan_json_export, f"plan_{keyword_obj.name}.json", "application/json", key="export_json")
                st.download_button("Exporter Markdown", plan_md, f"plan_{keyword_obj.name}.md", "text/markdown", key="export_md")
                st.download_button("Exporter HTML", plan_html_export, f"plan_{keyword_obj.name}.html", "text/html", key="export_html")
            except Exception as e: st.error(f"Erreur lors de la préparation de l'export: {e}")

            # Remove the button that relied on current_tab state
            # if st.button("Passer à la rédaction", key="goto_writer_button"):
            #     st.session_state.current_tab = "article_writer"
            #     st.rerun()

        # Display Tabs for plan details
        display_tabs = st.tabs(["📝 Aperçu Markdown", "🔢 Données JSON", "🔍 SERP", "📊 Recherche"])
        with display_tabs[0]:
            try: st.markdown(plan_to_markdown(plan_obj.data))
            except Exception as e: st.error(f"Erreur d'affichage Markdown: {e}")
        with display_tabs[1]:
            try: st.json(plan_obj.data)
            except Exception as e: st.error(f"Erreur d'affichage JSON: {e}")
        with display_tabs[2]:
            last_serp = st.session_state.get("last_serp_results")
            if last_serp: st.json(last_serp)
            else: st.info("Aucune donnée SERP générée ou chargée pour ce plan.")
        with display_tabs[3]:
            last_research = st.session_state.get("last_research_results")
            if last_research: st.json([r.to_dict() for r in last_research])
            else: st.info("Aucune donnée de recherche générée ou chargée pour ce plan.")

def setup_article_writer_tab():
    """Setup the Article Writer tab"""
    st.header("Rédacteur d'Articles")
    data_manager: DataManager = services['data_manager']
    content_engine: ContentEngine = services['content_engine']

    # Check if we have a plan object in state
    if not st.session_state.get('plan_generated') or not st.session_state.current_plan_obj:
        st.warning("Veuillez d'abord générer un plan dans l'onglet 'Générateur de Plan'.")
        # Remove button relying on current_tab state
        # if st.button("Aller au Générateur de Plan"):
        #     st.session_state.current_tab = "plan_generator"
        #     st.rerun()
        return

    current_plan_obj: ArticlePlan = st.session_state.current_plan_obj
    current_keyword_obj: Optional[Keyword] = st.session_state.current_keyword_obj
    keyword_name = current_keyword_obj.name if current_keyword_obj else "N/A"

    st.info(f"Rédaction d'un article pour le mot-clé : **{keyword_name}**")

    # Display relevant plan info
    with st.expander("Rappel du Plan", expanded=False):
        st.markdown(f"**H1:** {current_plan_obj.get('h1', 'Non défini')}")
        st.markdown(f"**Meta Title:** {current_plan_obj.get('metatitle', 'Non défini')}")
        st.markdown(f"**Meta Description:** {current_plan_obj.get('metadescription', 'Non défini')}")
        plan_structure = current_plan_obj.get('plan', [])
        if plan_structure:
            st.markdown("**Structure Principale (H2):**")
            for section in plan_structure: st.markdown(f"- {section.get('h2', 'Section sans titre')}")
        else: st.markdown("Aucune section H2 définie.")

    # Model selection
    st.subheader("1. Sélection du Modèle")
    # Use the model_selector component, which now updates session state internally
    model_selector("article", "Rédaction", st.session_state.article_model_settings)

    # Article generation form
    with st.form("article_generation_form"):
        st.subheader("2. Paramètres de Rédaction")
        col1, col2 = st.columns(2)
        with col1: tone = st.selectbox("Ton", ["Professionnel", "Décontracté", "Académique", "Persuasif", "Informatif"], key="article_tone")
        with col2: word_count = st.number_input("Mots cible", 500, 10000, 1500, 100, key="article_word_count")
        content_instructions = st.text_area("Instructions additionnelles", key="article_instructions", height=100)

        # Prepare research summary from state
        research_summary = "Aucune recherche complémentaire fournie."
        last_research = st.session_state.get("last_research_results")
        if last_research:
             successful_searches = [r for r in last_research if r.success and not r.content.startswith("Error:")] # Exclude errors
             if successful_searches:
                  research_summary = f"Résumé de la recherche approfondie pour '{keyword_name}':\n- {len(successful_searches)} recherches réussies.\n"
                  for i, res in enumerate(successful_searches[:5]): research_summary += f"  - Requête '{res.query[:30]}...': {res.content[:10000]}...\n"

        submit_button = st.form_submit_button("✍️ Rédiger l'Article")

    # Handle form submission
    if submit_button:
        if not current_plan_obj or not current_keyword_obj:
             st.error("Erreur: Plan ou mot-clé manquant.")
        else:
            with st.spinner("Rédaction de l'article en cours..."):
                try:
                    article_gen_params = {"tone": tone, "word_count": word_count, "content_instructions": content_instructions}
                    # Run async generation using settings from session state
                    generated_article_obj: GeneratedArticle = asyncio.run(content_engine.generate_article(
                        article_plan=current_plan_obj,
                        model_settings=st.session_state.article_model_settings, # Use settings from state
                        research_summary=research_summary,
                        generation_params=article_gen_params
                    ))

                    st.session_state.current_article_obj = generated_article_obj
                    st.session_state.article_generated = True
                    save_success = data_manager.save_article(current_keyword_obj, generated_article_obj)

                    if save_success: st.success("Article rédigé et sauvegardé!")
                    else: st.warning("Article rédigé mais échec de la sauvegarde.")
                    st.rerun()

                except Exception as e:
                    logger.error(f"Erreur lors de la rédaction de l'article: {e}", exc_info=True)
                    st.error(f"Erreur lors de la rédaction de l'article: {e}")
                    st.session_state.article_generated = False
                    st.session_state.current_article_obj = None

    # Display generated article
    if st.session_state.article_generated and st.session_state.current_article_obj:
        st.subheader("3. Article Généré")
        article_obj = st.session_state.current_article_obj
        keyword_obj = st.session_state.current_keyword_obj

        if not keyword_obj:
             st.error("Erreur interne: Mot-clé non défini pour l'article.")
             return

        # Article editing area
        edited_article_content = st.text_area(
            "Éditer l'article (Markdown)", value=article_obj.content, height=600,
            key=f"edit_article_{keyword_obj.name}"
        )

        # Actions column
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Mettre à jour l'article", key="update_article_button"):
                if edited_article_content != article_obj.content:
                    updated_article_obj = GeneratedArticle(content=edited_article_content)
                    st.session_state.current_article_obj = updated_article_obj
                    save_success = data_manager.save_article(keyword_obj, updated_article_obj)
                    if save_success: st.success("Article mis à jour et sauvegardé!")
                    else: st.warning("Article mis à jour mais échec de la sauvegarde.")
                else:
                    st.info("Aucune modification détectée.")

        with col2:
            export_format = st.selectbox("Format d'export", ["Markdown", "HTML", "Word", "WordPress"], key="export_format_article")
            if st.button("Exporter l'article", key="export_article_button"):
                file_name_base = f"article_{keyword_obj.name.replace(' ', '_')}"
                try:
                    if export_format == "Markdown":
                        st.download_button("Télécharger Markdown", edited_article_content, f"{file_name_base}.md", "text/markdown")
                    elif export_format == "HTML":
                        html_content = markdown_to_html(edited_article_content)
                        st.download_button("Télécharger HTML", html_content, f"{file_name_base}.html", "text/html")
                    elif export_format == "Word":
                        docx_bytes = markdown_to_docx(edited_article_content, keyword_obj.name)
                        st.download_button("Télécharger Word", docx_bytes, f"{file_name_base}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
                    elif export_format == "WordPress":
                        wp_html = prepare_for_wordpress(edited_article_content, keyword_obj.name)
                        st.download_button("Télécharger HTML (WordPress)", wp_html, f"{file_name_base}_wp.html", "text/html")
                except Exception as e:
                    st.error(f"Erreur d'exportation ({export_format}): {e}")

        # Preview Tabs
        preview_tabs = st.tabs(["📄 Aperçu Markdown", "🌐 Aperçu HTML"])
        with preview_tabs[0]:
            st.markdown(edited_article_content)
        with preview_tabs[1]:
            try: st.components.v1.html(markdown_to_html(edited_article_content), height=600, scrolling=True)
            except Exception as e: st.error(f"Erreur rendu HTML: {e}")


def setup_content_refiner_tab():
    """Setup the Content Refiner tab"""
    st.header("Raffineur de Contenu")
    content_engine: ContentEngine = services['content_engine']
    data_manager: DataManager = services['data_manager']

    # Check if we have an article object in state
    if not st.session_state.get('article_generated') or not st.session_state.current_article_obj:
        st.warning("Veuillez d'abord générer un article dans l'onglet 'Rédacteur d'Articles'.")
        # Remove button relying on current_tab state
        # if st.button("Aller au Rédacteur"):
        #     st.session_state.current_tab = "article_writer"
        #     st.rerun()
        return

    current_article_obj: GeneratedArticle = st.session_state.current_article_obj
    current_keyword_obj: Optional[Keyword] = st.session_state.current_keyword_obj
    keyword_name = current_keyword_obj.name if current_keyword_obj else "N/A"

    st.info(f"Raffinement de l'article pour : **{keyword_name}**")

    # Display current article
    with st.expander("Article Actuel (Markdown)", expanded=False):
        st.markdown(current_article_obj.content)

    # Model selection
    st.subheader("1. Sélection du Modèle")
    # Use the model_selector component, which now updates session state internally
    model_selector("refine", "Raffinement", st.session_state.refine_model_settings)

    # Refining options
    st.subheader("2. Options de Raffinement")
    refine_tabs = st.tabs(["Amélioration Ciblée", "Éléments Visuels", "Search & Replace", "Prompt Personnalisé"])

    with refine_tabs[0]:
        st.markdown("### Amélioration Ciblée")
        selected_text = st.text_area("Texte à améliorer", height=150, key="refine_select")
        improvement_type = st.selectbox("Type d'amélioration", ["Développer", "Simplifier", "Corriger"], key="refine_type")
        if st.button("Améliorer Section", key="refine_improve_btn"):
            if selected_text and st.session_state.refine_model_settings and current_keyword_obj:
                with st.spinner("Amélioration..."):
                    try:
                        # TODO: Implement call to a potential content_engine.refine_text_section method
                        # This method would need to be added to ContentEngine and use LLMService
                        st.info("Fonctionnalité d'amélioration à implémenter.")
                    except Exception as e:
                        st.error(f"Erreur: {e}")
            else:
                st.warning("Veuillez sélectionner du texte, un modèle et un mot-clé actif.")

    with refine_tabs[1]: st.markdown("### Ajout d'Éléments Visuels (à implémenter)")
    with refine_tabs[2]: st.markdown("### Search & Replace Assisté (à implémenter)")
    with refine_tabs[3]: st.markdown("### Prompt Personnalisé (à implémenter)")


def setup_batch_processor_tab():
    """Setup the Batch Processor tab"""
    st.header("Traitement par Lots")
    batch_processor_instance: BatchProcessor = st.session_state.batch_processor_instance
    data_manager: DataManager = services['data_manager']
    content_engine: ContentEngine = services['content_engine']
    serp_scraper: SERPScraper = services['serp_scraper']
    research_engine: ResearchEngine = services['research_engine']

    current_project = st.session_state.current_project
    if not current_project:
        st.warning("Veuillez d'abord créer ou charger un projet dans la barre latérale.")
        return

    # --- Batch Input ---
    st.subheader("1. Mots-Clés à Traiter")
    input_method = st.radio("Source des mots-clés", ["Manuel", "Fichier Texte", "CSV"], key="batch_input_method", horizontal=True)

    keywords_list = []
    if input_method == "Manuel":
        keywords_text = st.text_area("Entrez les mots-clés (un par ligne)", key="batch_manual_keywords", height=150)
        if keywords_text:
            keywords_list = [kw.strip() for kw in keywords_text.split('\n') if kw.strip()]
    elif input_method == "Fichier Texte":
        uploaded_file = st.file_uploader("Choisissez un fichier .txt", type="txt", key="batch_txt_upload")
        if uploaded_file is not None:
            try:
                keywords_list = [line.decode('utf-8').strip() for line in uploaded_file if line.strip()]
                st.success(f"{len(keywords_list)} mots-clés chargés depuis {uploaded_file.name}")
            except Exception as e:
                st.error(f"Erreur lecture fichier texte: {e}")
    elif input_method == "CSV":
        uploaded_file = st.file_uploader("Choisissez un fichier .csv", type="csv", key="batch_csv_upload")
        if uploaded_file is not None:
            st.warning("Import CSV non implémenté.")

    if keywords_list:
        st.info(f"{len(keywords_list)} mots-clés prêts pour le traitement.")
        with st.expander("Voir les mots-clés"):
            st.write(keywords_list)

    # --- Batch Configuration ---
    st.subheader("2. Configuration du Traitement")
    process_options = st.multiselect(
        "Tâches à effectuer",
        ["Générer Plan", "Générer Article"],
        key="batch_process_options"
    )

    batch_models = {}
    if "Générer Plan" in process_options:
        with st.expander("Options Génération Plan", expanded=False):
            # Use model_selector, which updates session state
            model_selector("batch_plan", "Plan SEO (Batch)", st.session_state.plan_model_settings)
            # Store the selected settings for the batch run
            batch_models["plan"] = st.session_state.batch_plan_model_settings
            st.session_state.batch_plan_use_serp = st.checkbox("Forcer Analyse SERP (Batch)", value=False, key="batch_plan_serp")
            st.session_state.batch_plan_use_research = st.checkbox("Forcer Recherche Approfondie (Batch)", value=False, key="batch_plan_research")

    if "Générer Article" in process_options:
        with st.expander("Options Génération Article", expanded=False):
            # Use model_selector, which updates session state
            model_selector("batch_article", "Rédaction Article (Batch)", st.session_state.article_model_settings)
            # Store the selected settings for the batch run
            batch_models["article"] = st.session_state.batch_article_model_settings
            st.session_state.batch_article_tone = st.selectbox("Ton par défaut (Batch)", ["Professionnel", "Décontracté"], key="batch_article_tone_select")

    # --- Batch Execution ---
    st.subheader("3. Exécution")

    if st.button("🚀 Démarrer le Traitement", key="start_batch_button"):
        if not keywords_list: st.error("Ajoutez des mots-clés.")
        elif not process_options: st.error("Sélectionnez au moins une tâche.")
        elif not current_project: st.error("Sélectionnez un projet actif.")
        else:
            items_to_process: List[Keyword] = []
            with st.spinner("Préparation des mots-clés..."):
                for kw_name in keywords_list:
                    kw_obj = data_manager.get_keyword(current_project, kw_name)
                    if kw_obj: items_to_process.append(kw_obj)
                    else: logger.warning(f"Mot-clé '{kw_name}' non trouvé, ignoré.")
            st.info(f"Traitement de {len(items_to_process)} mots-clés.")

            if not items_to_process:
                 st.error("Aucun mot-clé valide trouvé.")
                 return

            async def process_keyword_item(keyword_obj: Keyword, state_mgr: Optional[Any]) -> Dict[str, Any]:
                item_result = {"keyword": keyword_obj.name, "success": True, "error": None, "plan_generated": False, "article_generated": False}
                plan_obj: Optional[ArticlePlan] = None
                article_obj: Optional[GeneratedArticle] = None

                try:
                    if "Générer Plan" in process_options:
                        logger.info(f"Batch: Generating plan for {keyword_obj.name}")
                        serp_summary, research_summary = "N/A", "N/A"
                        plan_model_settings = batch_models.get("plan", st.session_state.plan_model_settings) # Get settings selected for batch

                        if st.session_state.get("batch_plan_use_serp", False):
                            try:
                                serp_response = await serp_scraper.get_serp_results(keyword_obj.name)
                                if serp_response and isinstance(serp_response.get('content'), list):
                                     serp_summary = f"SERP: {len(serp_response['content'])} results"
                                     # TODO: Potentially scrape content here too if needed for batch plan
                                else: serp_summary = "SERP: No results"
                            except Exception as e: serp_summary = f"SERP Error: {e}"

                        if st.session_state.get("batch_plan_use_research", False):
                            try:
                                queries = [f"tendances {keyword_obj.name}", f"statistiques {keyword_obj.name}"]
                                research_results = await research_engine.perform_research(queries, plan_model_settings)
                                research_summary = f"Research: {len([r for r in research_results if r.success])} successful"
                            except Exception as e: research_summary = f"Research Error: {e}"

                        plan_params = {"include_h3": True}
                        plan_obj = await content_engine.generate_plan(
                            keyword=keyword_obj.name, model_settings=plan_model_settings,
                            competitor_analysis_summary=serp_summary, research_summary=research_summary,
                            generation_params=plan_params
                        )
                        if plan_obj: data_manager.save_plan(keyword_obj, plan_obj); item_result["plan_generated"] = True; logger.info(f"Batch: Plan saved for {keyword_obj.name}")
                        else: raise ValueError("Plan generation returned None")

                    if "Générer Article" in process_options:
                        if not plan_obj: plan_obj = data_manager.load_plan(keyword_obj)
                        if not plan_obj: raise ValueError("Plan required but not found/generated.")

                        logger.info(f"Batch: Generating article for {keyword_obj.name}")
                        article_params = {"tone": st.session_state.get("batch_article_tone", "Professionnel"), "word_count": 1500}
                        article_model_settings = batch_models.get("article", st.session_state.article_model_settings) # Get settings selected for batch
                        research_summary_for_article = "N/A" # TODO: Get research if needed

                        article_obj = await content_engine.generate_article(
                            article_plan=plan_obj, model_settings=article_model_settings,
                            research_summary=research_summary_for_article, generation_params=article_params
                        )
                        if article_obj: data_manager.save_article(keyword_obj, article_obj); item_result["article_generated"] = True; logger.info(f"Batch: Article saved for {keyword_obj.name}")
                        else: raise ValueError("Article generation returned None")

                except Exception as e:
                    logger.error(f"Batch Error processing {keyword_obj.name}: {e}", exc_info=True)
                    item_result["success"] = False; item_result["error"] = str(e)
                return item_result

            def batch_progress_callback_sync(processed, total, item_id, result):
                 logger.info(f"Batch Progress Update: {processed}/{total} - Item: {item_id} - Success: {result.get('success') if result else 'N/A'}")
                 st.session_state.batch_status_info = batch_processor_instance.get_progress()
                 try: st.rerun()
                 except st.errors.StreamlitAPIException as e: logger.warning(f"Streamlit rerun failed: {e}")

            try:
                async def start():
                     await batch_processor_instance.start_batch(
                         items=items_to_process, item_processor=process_keyword_item,
                         progress_callback=batch_progress_callback_sync, item_id_func=lambda kw: kw.name
                     )
                asyncio.run(start())
                st.success("Traitement par lots démarré!")
                st.rerun()
            except Exception as e: st.error(f"Erreur démarrage batch: {e}")

    # --- Display Batch Progress ---
    progress_info = st.session_state.batch_status_info
    batch_status = progress_info["status"]

    if batch_status != BatchStatus.IDLE.value:
        st.subheader("4. Progression")
        progress_val = progress_info['processed_items'] / progress_info['total_items'] if progress_info['total_items'] > 0 else 0
        st.progress(progress_val)
        st.write(f"Statut: {progress_info['status']} | Traités: {progress_info['processed_items']}/{progress_info['total_items']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if batch_status == BatchStatus.RUNNING.value and st.button("⏸️ Pause", key="pause_batch"):
                asyncio.run(batch_processor_instance.pause_batch())
                st.session_state.batch_status_info = batch_processor_instance.get_progress(); st.rerun()
        with col2:
            if batch_status == BatchStatus.PAUSED.value and st.button("▶️ Reprendre", key="resume_batch"):
                asyncio.run(batch_processor_instance.resume_batch())
                st.session_state.batch_status_info = batch_processor_instance.get_progress(); st.rerun()
        with col3:
            if batch_status in [BatchStatus.RUNNING.value, BatchStatus.PAUSED.value] and st.button("⏹️ Arrêter", key="stop_batch"):
                asyncio.run(batch_processor_instance.stop_batch())
                st.session_state.batch_status_info = batch_processor_instance.get_progress(); st.rerun()

        with st.expander("Voir les résultats détaillés"):
             st.json({"results": progress_info.get("results",{}), "errors": progress_info.get("errors",{})})


def setup_content_refiner_tab():
    """Setup the Content Refiner tab"""
    st.header("Raffineur de Contenu")
    content_engine: ContentEngine = services['content_engine']
    data_manager: DataManager = services['data_manager']

    if not st.session_state.get('article_generated') or not st.session_state.current_article_obj:
        st.warning("Veuillez d'abord générer un article dans l'onglet 'Rédacteur d'Articles'.")
        return

    current_article_obj: GeneratedArticle = st.session_state.current_article_obj
    current_keyword_obj: Optional[Keyword] = st.session_state.current_keyword_obj
    keyword_name = current_keyword_obj.name if current_keyword_obj else "N/A"

    st.info(f"Raffinement de l'article pour : **{keyword_name}**")

    with st.expander("Article Actuel (Markdown)", expanded=False):
        st.markdown(current_article_obj.content)

    st.subheader("1. Sélection du Modèle")
    model_selector("refine", "Raffinement", st.session_state.refine_model_settings)

    st.subheader("2. Options de Raffinement")
    refine_tabs = st.tabs(["Amélioration Ciblée", "Éléments Visuels", "Search & Replace", "Prompt Personnalisé"])

    with refine_tabs[0]:
        st.markdown("### Amélioration Ciblée")
        selected_text = st.text_area("Texte à améliorer", height=150, key="refine_select")
        improvement_type = st.selectbox("Type d'amélioration", ["Développer", "Simplifier", "Corriger"], key="refine_type")
        if st.button("Améliorer Section", key="refine_improve_btn"):
            if selected_text and st.session_state.refine_model_settings and current_keyword_obj:
                with st.spinner("Amélioration..."):
                    try: st.info("Fonctionnalité d'amélioration à implémenter.")
                    except Exception as e: st.error(f"Erreur: {e}")
            else: st.warning("Veuillez sélectionner du texte, un modèle et un mot-clé actif.")

    with refine_tabs[1]: st.markdown("### Ajout d'Éléments Visuels (à implémenter)")
    with refine_tabs[2]: st.markdown("### Search & Replace Assisté (à implémenter)")
    with refine_tabs[3]: st.markdown("### Prompt Personnalisé (à implémenter)")


# --- Main Application Flow ---
def main():
    st.markdown("""
    # <span style='color:#4285F4;'>SEO</span> <span style='color:#34A853;'>Content</span> <span style='color:#FBBC05;'>Generator</span> <span style='color:#EA4335;'>Pro</span>
    ### Solution complète pour générer du contenu SEO optimisé basé sur l'analyse SERP et les données fraîches
    """, unsafe_allow_html=True)

    if not st.session_state.api_keys:
        st.warning("⚠️ Configuration requise: Aucune clé API détectée.")
        st.markdown("Configurez au moins une clé API dans votre fichier `.env` et redémarrez.")
        return

    tab_options = ["Générateur de Plan", "Rédacteur d'Articles", "Raffineur de Contenu", "Traitement par Lots"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_options)

    with tab1: setup_plan_generator_tab()
    with tab2: setup_article_writer_tab()
    with tab3: setup_content_refiner_tab()
    with tab4: setup_batch_processor_tab()

if __name__ == "__main__":
    main()
