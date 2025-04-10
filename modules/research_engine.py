import asyncio
import json
import logging
import re
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
# Removed aiohttp imports as we'll try the OpenAI client interface

# Import necessary services and config
from modules.llm_service import LLMService, QuotaError
import config

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchResult:
    """Represents the structured result of a single research query."""
    def __init__(self, query: str, content: str, sources: List[Dict[str, str]], insights: List[Dict[str, str]], success: bool = True, error: Optional[str] = None):
        self.query = query
        self.content = content # Raw content from search/LLM
        self.sources = sources # List of {"title": str, "url": str, "snippet": str}
        self.insights = insights # List of {"type": str, "content": str, "source": str, "date": str}
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "content": self.content,
            "sources": self.sources,
            "insights": self.insights,
            "success": self.success,
            "error": self.error
        }

class ResearchEngine:
    """
    Performs targeted research using external APIs (like Perplexity) or LLM simulation.
    Focuses on executing queries and returning structured, factual information.
    """

    def __init__(self, llm_service: LLMService, perplexity_api_key: Optional[str] = None):
        """
        Initializes the research engine.
        """
        if not isinstance(llm_service, LLMService):
            raise TypeError("llm_service must be an instance of LLMService")

        self.llm_service = llm_service
        self.perplexity_api_key = perplexity_api_key
        self.use_real_perplexity_api = bool(perplexity_api_key)
        self.api_base_url = "https://api.perplexity.ai"
        self._search_cache: Dict[str, ResearchResult] = {}
        self.request_delay = 3
        self._perplexity_client = None

        if self.use_real_perplexity_api:
            try:
                # Try initializing an OpenAI-compatible client for Perplexity
                from openai import AsyncOpenAI
                self._perplexity_client = AsyncOpenAI(api_key=self.perplexity_api_key, base_url=self.api_base_url)
                logger.info("ResearchEngine initialized with Perplexity API key (using OpenAI client interface).")
            except ImportError:
                 logger.warning("OpenAI library not found. Cannot use OpenAI interface for Perplexity.")
                 self.use_real_perplexity_api = False # Disable if library missing
            except Exception as e:
                 logger.error(f"Failed to initialize Perplexity client: {e}")
                 self.use_real_perplexity_api = False # Disable on error
        else:
            logger.warning("ResearchEngine initialized WITHOUT Perplexity API key. Research will be simulated.")

    # ... (rest of the methods remain the same initially) ...

    async def perform_research(self, queries: List[str], model_settings: Dict[str, Any], state_manager: Optional[Any] = None) -> List[ResearchResult]:
        """
        Performs research for a list of queries, either via Perplexity or LLM simulation.
        """
        logger.info(f"Starting research for {len(queries)} queries.")
        # self._update_status(state_manager, "in_progress", 0.0, f"Starting research for {len(queries)} queries.") # Status update removed for brevity

        results: List[ResearchResult] = []
        total_queries = len(queries)

        for i, query in enumerate(queries):
            # progress = (i / total_queries)
            # self._update_status(state_manager, "in_progress", progress, f"Researching query {i+1}/{total_queries}: '{query[:50]}...'")

            cache_key = query.lower().strip()
            if cache_key in self._search_cache:
                logger.info(f"Using cached result for query: '{query}'")
                results.append(self._search_cache[cache_key])
                continue

            try:
                search_result_data = await self._execute_single_search(query, model_settings)
                research_result = ResearchResult(**search_result_data)
                if research_result.success:
                    self._search_cache[cache_key] = research_result
                results.append(research_result)

                if self.use_real_perplexity_api and i < total_queries - 1:
                    await asyncio.sleep(self.request_delay)

            except Exception as e:
                logger.error(f"Failed research for query '{query}': {e}", exc_info=True)
                error_result = ResearchResult(query=query, content="", sources=[], insights=[], success=False, error=str(e))
                results.append(error_result)

        # final_progress = 1.0
        # self._update_status(state_manager, "completed", final_progress, f"Research completed for {len(results)} queries.")
        logger.info(f"Research finished. Returning {len(results)} results.")
        return results

    async def _execute_single_search(self, query: str, model_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single search query using Perplexity API or LLM simulation.
        """
        content = ""
        sources = []
        insights = []
        success = False
        error_msg = None

        try:
            if self.use_real_perplexity_api and self._perplexity_client:
                logger.debug(f"Attempting Perplexity API search for: '{query}'")
                try:
                    api_result = await self._call_perplexity_api_openai_sdk(query) # Use new method
                    content = api_result.get("content", "")
                    sources = api_result.get("sources", []) # Assuming _call method extracts sources
                    insights = self._create_direct_insights_from_perplexity(content)
                    success = True
                    logger.debug(f"Perplexity API search successful for: '{query}'")
                except Exception as api_error:
                    logger.warning(f"Perplexity API call failed for '{query}': {api_error}. Falling back to LLM simulation.")
                    error_msg = f"Perplexity API Error: {api_error}"
                    success = False # Ensure success is false if API fails

            if not success:
                logger.debug(f"Simulating research via LLM for: '{query}'")
                simulated_result = await self._simulate_search_with_llm(query, model_settings)
                content = simulated_result.get("content", "")
                sources = simulated_result.get("sources", [])
                if not content.startswith("Error:"):
                    insights = await self._extract_insights_from_content(query, content, model_settings)
                else:
                    # If simulation itself returned an error message, capture it
                    error_msg = content if error_msg is None else f"{error_msg}; Simulation Error: {content}"
                success = True # Simulation attempt is considered done
                logger.debug(f"LLM simulation completed for: '{query}'")

        except Exception as e:
            logger.error(f"Unhandled error during search execution for '{query}': {e}", exc_info=True)
            error_msg = str(e)
            success = False

        return {
            "query": query, "content": content, "sources": sources,
            "insights": insights, "success": success, "error": error_msg
        }

    async def _call_perplexity_api_openai_sdk(self, query: str) -> Dict[str, Any]:
        """Calls the Perplexity API using the OpenAI SDK interface."""
        if not self._perplexity_client:
             raise ValueError("Perplexity client (OpenAI interface) not initialized.")

        messages = [
            {"role": "system", "content": "You are an AI research assistant. Provide concise, factual, and up-to-date information with sources."},
            {"role": "user", "content": query}
        ]
        try:
            response = await self._perplexity_client.chat.completions.create(
                model="sonar-pro", # Use recommended sonar-pro
                messages=messages,
                # Perplexity doesn't use temp/max_tokens in the same way via this endpoint
            )
            content = response.choices[0].message.content or ""
            # Basic URL extraction as fallback for sources
            urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', content)
            sources = [{"title": f"Source {i+1}", "url": url, "snippet": "URL extracted"} for i, url in enumerate(set(urls))]
            return {"content": content, "sources": sources}
        except Exception as e:
             logger.error(f"Error calling Perplexity API via OpenAI SDK interface: {e}")
             raise # Re-raise the exception to be caught by _execute_single_search


    async def _simulate_search_with_llm(self, query: str, model_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates a web search using the LLM."""
        current_date = datetime.now()
        prompt = f"""
        Simulate an advanced web search for the query: "{query}"
        Provide a concise summary of the most relevant, factual, and up-to-date information (as of {current_date.strftime('%B %Y')}).
        Include specific statistics, dates, company/product names if applicable.
        Structure the response clearly. Include 2-3 plausible source URLs at the end, like [Source: https://example.com/article].
        Focus on objectivity. Do not state this is a simulation.
        """
        try:
            response_dict = await self.llm_service.generate_content(
                prompt=prompt,
                provider=model_settings.get("provider"),
                model=model_settings.get("model"),
                temperature=0.3,
                max_tokens=1000
            )
            content = response_dict.get("content", "")

            if isinstance(content, str) and content.startswith("Error:"):
                 logger.warning(f"LLM simulation for '{query}' returned an error message: {content}")
                 return {"content": content, "sources": []}
            elif not isinstance(content, str):
                 logger.error(f"LLM simulation for '{query}' received non-string content: {type(content)}")
                 return {"content": "Error: Invalid content type from LLM simulation.", "sources": []}

            sources = self._extract_simulated_sources(content)
            content_cleaned = re.sub(r'\[Source:\s*https?://.*?\]', '', content).strip()
            return {"content": content_cleaned, "sources": sources}
        except Exception as e:
            logger.error(f"LLM simulation failed for query '{query}': {e}", exc_info=True)
            return {"content": f"Error during LLM simulation: {e}", "sources": []}

    def _extract_simulated_sources(self, content: str) -> List[Dict[str, str]]:
        """Extracts source URLs added by the simulation prompt."""
        sources = []
        matches = re.findall(r'\[Source:\s*(https?://.*?)\s*\]', content)
        for i, url in enumerate(matches):
            context_search_area = content.rfind('\n', 0, content.find(f'[Source: {url}]'))
            context_start = max(0, context_search_area) if context_search_area != -1 else 0
            context = content[context_start:content.find(f'[Source: {url}]')].strip()
            title = context.split('.')[-1].strip() if '.' in context else f"Simulated Source {i+1}"
            sources.append({"title": title[:100], "url": url, "snippet": ""})
        return sources

    async def _extract_insights_from_content(self, query: str, content: str, model_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extracts structured insights from research content using an LLM."""
        if not content or content.startswith("Error"):
            return []

        prompt = f"""
        Analyze the following research content related to the query "{query}" and extract key factual insights.

        # RESEARCH CONTENT
        ```
        {content}
        ```

        # TASK
        Extract the 5-7 most important factual insights. Focus on:
        - Statistics (e.g., percentages, numbers, market size)
        - Key findings or conclusions
        - Specific dates or timeframes mentioned
        - Names of companies, products, or technologies involved
        - Comparisons or trends supported by data

        # OUTPUT FORMAT (Strict JSON Array)
        Provide ONLY a valid JSON array of objects, where each object has 'type', 'content', 'source', 'date'.
        - 'type': "statistique", "finding", "trend", "comparison", "mention"
        - 'content': The specific factual insight (concise).
        - 'source': The source mentioned in the text, or "Inferred from text".
        - 'date': Date mentioned (YYYY-MM or YYYY), or "Current".

        Example object:
        {{"type": "statistique", "content": "Market grew by 15% in 2023", "source": "Report X", "date": "2023"}}

        ```json
        [
          // insights here
        ]
        ```
        """
        try:
            response_dict = await self.llm_service.generate_content(
                prompt=prompt,
                provider=model_settings.get("provider"),
                model=model_settings.get("model"),
                temperature=0.2,
                max_tokens=1000
            )
            response_content = response_dict.get("content", "")
            if not isinstance(response_content, str) or response_content.startswith("Error:"):
                 logger.warning(f"Insight extraction LLM call failed or returned error: {response_content}")
                 return []

            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    insights = json.loads(json_str)
                    if isinstance(insights, list):
                        validated_insights = [i for i in insights if isinstance(i, dict) and all(k in i for k in ['type', 'content'])]
                        return validated_insights
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON for insights: {json_str}")
            else:
                 logger.warning(f"No JSON block found in insight extraction response: {response_content}")

        except Exception as e:
            logger.error(f"Failed to extract insights using LLM for query '{query}': {e}", exc_info=True)

        return self._create_direct_insights_from_perplexity(content)

    def _create_direct_insights_from_perplexity(self, content: str) -> List[Dict[str, str]]:
        """Creates basic insights directly from Perplexity/simulated content."""
        if not content or content.startswith("Error"): return []
        insights = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p) > 20]

        for para in paragraphs[:7]:
            insight_type = "finding"
            if re.search(r'\b\d+%|\b\d+\.\d+%', para): insight_type = "statistique"
            elif re.search(r'\bcompar|vs\.|\btrend\b|évolution', para, re.IGNORECASE): insight_type = "trend/comparison"
            date_match = re.search(r'\b(20\d{2})\b', para)
            date = date_match.group(1) if date_match else "Current"
            source = "Inferred from text"
            source_match = re.search(r'\[Source:\s*(.*?)\s*\]', para)
            if source_match: source = source_match.group(1)
            insights.append({"type": insight_type, "content": para[:300] + ("..." if len(para) > 300 else ""), "source": source, "date": date})
        return insights

# Example Usage
if __name__ == "__main__":
    async def main():
        print("Testing ResearchEngine...")
        from dotenv import load_dotenv
        load_dotenv()
        llm_api_keys = {k: os.getenv(f"{k.upper()}_API_KEY") for k in ["openai", "anthropic", "google", "deepseek"]}
        llm_api_keys = {k: v for k, v in llm_api_keys.items() if v}
        if not llm_api_keys: print("LLM API key required."); return

        llm = LLMService(api_keys=llm_api_keys)
        model_settings = {"provider": "OpenAI", "model": "gpt-3.5-turbo"} # Default for simulation/extraction

        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        engine_simulated = ResearchEngine(llm_service=llm, perplexity_api_key=None)
        engine_real = ResearchEngine(llm_service=llm, perplexity_api_key=perplexity_key) if perplexity_key else None

        test_queries = ["latest trends in large language models 2025", "market share comparison OpenAI vs Anthropic vs Google AI"]

        print("\n--- Testing with LLM Simulation ---")
        simulated_results = await engine_simulated.perform_research(test_queries, model_settings)
        for result in simulated_results:
            print(f"\nQuery: {result.query}\nSuccess: {result.success}")
            if result.success: print(f"Content Snippet: {result.content[:100]}...\nSources: {len(result.sources)}\nInsights: {len(result.insights)}")
            else: print(f"Error: {result.error}")

        if engine_real:
            print("\n--- Testing with Perplexity API ---")
            real_results = await engine_real.perform_research(test_queries, model_settings)
            for result in real_results:
                print(f"\nQuery: {result.query}\nSuccess: {result.success}")
                if result.success: print(f"Content Snippet: {result.content[:100]}...\nSources: {len(result.sources)}\nInsights: {len(result.insights)}")
                else: print(f"Error: {result.error}")
        else: print("\nSkipping Perplexity API test - API key not found.")

    asyncio.run(main())
