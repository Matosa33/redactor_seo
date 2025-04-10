import asyncio
import logging
from typing import List, Dict, Optional, Tuple
import httpx
from bs4 import BeautifulSoup, SoupStrainer # For parsing

logger = logging.getLogger(__name__)

# Basic content extraction (can be improved significantly)
def extract_main_content(html: str) -> str:
    """Tries to extract the main textual content from HTML."""
    try:
        # Focus on common main content tags, parse only these for efficiency
        # Add more tags if needed (e.g., 'div.content', 'div.post-body')
        parse_only = SoupStrainer(['main', 'article', 'p', 'h1', 'h2', 'h3', 'h4', 'li'])
        soup = BeautifulSoup(html, 'lxml', parse_only=parse_only)

        # Try common main content containers first
        main_content = soup.find('main') or soup.find('article')
        text = ""
        if main_content:
            # Get text from all relevant tags within main content
            tags_to_extract = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
            text = ' '.join(tag.get_text(strip=True) for tag in tags_to_extract)
        else:
            # Fallback: Get text from all paragraphs if no main container found
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs)

        # Basic cleaning (remove excessive whitespace)
        text = ' '.join(text.split())

        # Limit length to avoid excessive context
        MAX_LENGTH = 4000 # Increased limit
        return text[:MAX_LENGTH] + "..." if len(text) > MAX_LENGTH else text
    except Exception as e:
        logger.error(f"Error parsing HTML content: {e}", exc_info=True)
        return ""

async def fetch_url_content(client: httpx.AsyncClient, url: str) -> Tuple[str, Optional[str]]:
    """Fetches content for a single URL asynchronously."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        # Increased timeout, allow redirects
        response = await client.get(url, headers=headers, follow_redirects=True, timeout=20.0)
        response.raise_for_status() # Raise HTTP errors

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            logger.warning(f"Skipping non-HTML content at {url} (type: {content_type})")
            return url, None

        # Decode content (httpx handles gzip/deflate)
        html_content = response.text
        logger.info(f"Successfully fetched content from {url} ({len(html_content)} bytes)")
        return url, html_content
    except httpx.TimeoutException:
        logger.warning(f"Timeout fetching {url}")
    except httpx.RequestError as e:
        logger.warning(f"Request error fetching {url}: {e}")
    except httpx.HTTPStatusError as e:
        logger.warning(f"HTTP status error fetching {url}: {e.response.status_code}")
    except Exception as e:
        logger.warning(f"Unexpected error fetching {url}: {e}", exc_info=True)
    return url, None

async def scrape_urls_content(urls: List[str], max_concurrent: int = 3) -> Dict[str, str]:
    """
    Fetches and extracts main content from a list of URLs concurrently.

    Args:
        urls: List of URLs to scrape.
        max_concurrent: Maximum number of concurrent requests.

    Returns:
        A dictionary mapping URL to its extracted main content string.
    """
    scraped_content: Dict[str, str] = {}
    if not urls:
        return scraped_content

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []

    # Use a single client session for connection pooling
    async with httpx.AsyncClient() as client:
        for url in urls:
            async def task_wrapper(url_to_fetch):
                async with semaphore: # Control concurrency
                    fetched_url, html = await fetch_url_content(client, url_to_fetch)
                    if html:
                        main_text = extract_main_content(html)
                        if main_text:
                            return fetched_url, main_text
                return fetched_url, None # Return None content if fetch/parse fails

            tasks.append(task_wrapper(url))

        results = await asyncio.gather(*tasks, return_exceptions=True) # Gather results, catch task exceptions

    successful_scrapes = 0
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
        elif isinstance(result, tuple) and len(result) == 2:
            url, content = result
            if content:
                scraped_content[url] = content
                successful_scrapes += 1
        else:
             logger.warning(f"Unexpected result type from gather: {type(result)}")


    logger.info(f"Successfully scraped content from {successful_scrapes} out of {len(urls)} URLs.")
    return scraped_content
