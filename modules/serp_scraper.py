import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import aiohttp # Use aiohttp
from urllib.parse import urlencode, quote_plus

# Assuming core.serp_result defines SERPResult for caching
from core.serp_result import SERPResult
# Import config directly
import config
logger = logging.getLogger(__name__)

class SERPScraper:
    """
    Handles fetching and caching SERP results using the Bright Data SERP API
    via the Native Proxy-Based Access method (&brd_json=1), using aiohttp.
    """
    DEFAULT_COUNTRY = 'us'
    DEFAULT_TIMEOUT = 90 # Increased timeout

    def __init__(self, cache_dir: Path, proxy_config: Optional[Dict[str, Any]] = None, country: Optional[str] = None):
        """
        Initializes the SERPScraper for Proxy Access.

        Args:
            cache_dir: Path to the directory for caching SERP results.
            proxy_config: Dictionary containing Bright Data proxy details (username, password, host, port).
            country: The country code for the search (e.g., 'us', 'fr'). Defaults to config or 'us'.
        """
        self.cache_dir = Path(cache_dir)
        self.proxy_config = proxy_config # Store proxy config
        self.country = country or getattr(config, 'BRIGHTDATA_COUNTRY', self.DEFAULT_COUNTRY)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SERPScraper initialized for Proxy Access. Cache dir: {self.cache_dir}, Proxy configured: {bool(self.proxy_config)}, Country: {self.country}")

    def _get_cache_path(self, query: str) -> Path:
        """Generates a cache file path for a given query."""
        safe_filename = "".join(c if c.isalnum() else "_" for c in query) + f"_{self.country}.json"
        return self.cache_dir / safe_filename

    async def get_serp_results(self, query: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetches SERP results for a query using Bright Data SERP API (Proxy Method), using cache if available.

        Args:
            query: The search query.
            force_refresh: If True, bypasses the cache and fetches fresh results.

        Returns:
            A standardized response dictionary or None if an error occurs.
        """
        cache_path = self._get_cache_path(query)
        cached_standardized_response = SERPResult.load_from_cache(cache_path)

        if cached_standardized_response and not force_refresh:
            cache_expiry = getattr(config, 'SERP_CACHE_EXPIRY_SECONDS', 86400)
            timestamp_str = cached_standardized_response.data.get("metadata", {}).get("timestamp")
            if timestamp_str:
                 try:
                      cached_timestamp = float(timestamp_str)
                      if time.time() - cached_timestamp < cache_expiry:
                           logger.info(f"Using cached SERP results for '{query}' from {cache_path}")
                           return cached_standardized_response.data
                 except (ValueError, TypeError):
                      logger.warning(f"Invalid timestamp format in cache file {cache_path}")
            logger.info(f"Cache expired or invalid for '{query}'. Fetching fresh results.")

        logger.info(f"Fetching fresh SERP results for '{query}' via Bright Data Proxy (brd_json=1) using aiohttp...")

        if not self.proxy_config:
            logger.error("Cannot fetch SERP results: Bright Data proxy is not configured.")
            return None

        # Construct Bright Data proxy URL string for aiohttp
        try:
            proxy_auth = f"{self.proxy_config['username']}:{self.proxy_config['password']}"
            proxy_server = f"{self.proxy_config['host']}:{self.proxy_config['port']}"
            proxy_url = f"http://{proxy_auth}@{proxy_server}"
            logger.debug(f"Constructed proxy URL for aiohttp: {proxy_url}") # Log constructed URL
        except KeyError as e:
            logger.error(f"Missing key in proxy_config: {e}. Cannot construct proxy URL.")
            return None
        except Exception as e:
             logger.error(f"Error constructing proxy URL: {e}", exc_info=True)
             return None

        # Target Google URL, appending &brd_json=1 to trigger Bright Data's JSON response
        search_params = {'q': query, 'gl': self.country, 'hl': self.country, 'brd_json': '1'}
        target_url = f"https://www.google.com/search?{urlencode(search_params)}"

        headers = { # Standard headers might still be useful
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response_text = None # Initialize for error logging
        try:
            # Use aiohttp with the proxy URL
            # Disable SSL verification for the proxy connection using TCPConnector
            connector = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(connector=connector) as session:
                logger.debug(f"Requesting Bright Data SERP via Proxy (aiohttp): GET {target_url} via proxy")
                # Pass proxy URL to the get method
                async with session.get(target_url, headers=headers, proxy=proxy_url, timeout=self.DEFAULT_TIMEOUT) as response:
                    logger.debug(f"Bright Data SERP Response Status: {response.status}")
                    # Check for specific Bright Data error headers if needed
                    if 'x-luminati-error' in response.headers:
                         logger.error(f"Bright Data Error Header: {response.headers['x-luminati-error']}")
                    response.raise_for_status() # Raise HTTP errors based on status code
                    response_text = await response.text() # Read text first for potential JSON errors
                    brightdata_response_json = json.loads(response_text) # Then attempt JSON parsing

            # --- Parse the JSON response from Bright Data ---
            organic_results = []
            if isinstance(brightdata_response_json, dict):
                raw_organic = brightdata_response_json.get('organic', [])
                if isinstance(raw_organic, list):
                    for i, item in enumerate(raw_organic):
                        if isinstance(item, dict):
                            # Extract data similar to exemple_serp_scraper.py structure
                            link = item.get('link', '')
                            domain = ''
                            if link.startswith('http'):
                                try:
                                    domain = link.split('/')[2]
                                except IndexError:
                                    pass # Keep domain empty if split fails

                            organic_results.append({
                                'rank': item.get('rank', i + 1), # Use rank if available, else index
                                'url': link, # Bright Data uses 'link'
                                'meta_title': item.get('title', ''),
                                'meta_description': item.get('snippet', item.get('description', '')), # Use snippet or description
                                'domain': domain
                            })
                else:
                    logger.warning(f"Bright Data 'organic' results field is not a list: {type(raw_organic)}")
            else:
                logger.warning(f"Bright Data SERP API response was not a dictionary: {type(brightdata_response_json)}")

            if organic_results:
                logger.info(f"Successfully fetched and parsed {len(organic_results)} organic results via Bright Data SERP API for '{query}'.")

                # Create the final dictionary structure to return and cache
                # This structure aims to be usable by the rest of the application
                results_to_return = {
                    'query': query,
                    'organic_results': organic_results, # Use the parsed results list directly
                    'metadata': {
                        'timestamp': time.time(),
                        'source': 'brightdata_serp_proxy'
                    }
                }

                # Save the new structure to cache using SERPResult
                # Assuming SERPResult can handle saving this dictionary structure in its 'data' attribute
                serp_data_to_cache = SERPResult(query=query, data=results_to_return, cache_dir=self.cache_dir)
                serp_data_to_cache.save()
                logger.debug(f"Saved results to cache: {self._get_cache_path(query)}")

                return results_to_return # Return the dictionary containing the list of organic results
            else:
                logger.warning(f"Could not extract organic results from Bright Data SERP API response for '{query}'. Response: {str(brightdata_response_json)[:500]}...")
                return None # Return None if no results could be parsed

        except aiohttp.ClientError as e: # Catch aiohttp specific errors
            logger.error(f"AIOHTTP Client error fetching SERP via Bright Data for '{query}': {e}")
            return None
        # Handle potential JSONDecodeError if response is not valid JSON
        except json.JSONDecodeError as e:
             response_text_snippet = response_text[:500] if response_text else "N/A"
             logger.error(f"Failed to decode JSON response from Bright Data SERP API for '{query}': {e}. Response text snippet: {response_text_snippet}")
             return None
        except Exception as e:
             logger.error(f"Unexpected error fetching/parsing SERP via Bright Data for '{query}': {e}", exc_info=True)
             return None
