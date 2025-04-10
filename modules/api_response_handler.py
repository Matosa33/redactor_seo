import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class APIResponseHandler:
    """
    Handles and standardizes API responses across different services.
    Provides dynamic processing and sidebar integration capabilities.
    """
    
    @staticmethod
    def standardize_response(
        raw_response: Any,
        source: str,
        query: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Standardizes API responses into a common format.
        
        Args:
            raw_response: The raw API response data
            source: The API source/service name (e.g., "perplexity", "openai")
            query: The original query (if applicable)
            include_metadata: Whether to include processing metadata
            
        Returns:
            Standardized response dictionary
        """
        standardized = {
            "content": "",
            "sources": [],
            "insights": [],
            "metadata": {
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "error": None
            } if include_metadata else None
        }
        
        try:
            if source == "perplexity":
                standardized.update(APIResponseHandler._handle_perplexity(raw_response))
            elif source in ["openai", "anthropic", "google", "deepseek"]:
                standardized.update(APIResponseHandler._handle_llm(raw_response, source))
            elif source == "serp":
                standardized.update(APIResponseHandler._handle_serp(raw_response))
            else:
                standardized["content"] = str(raw_response)
                
            if query:
                standardized["query"] = query
                
        except Exception as e:
            logger.error(f"Error standardizing {source} response: {e}")
            if include_metadata and standardized.get("metadata"):
                standardized["metadata"].update({
                    "success": False,
                    "error": str(e)
                })
                
        return standardized
    
    @staticmethod
    def _handle_perplexity(response: Dict[str, Any]) -> Dict[str, Any]:
        """Handles Perplexity API response format."""
        result = {
            "content": response.get("content", ""),
            "sources": [],
            "insights": []
        }
        
        if response.get("citations"):
            result["sources"] = [
                {
                    "title": cit.get("title", f"Source {i+1}"),
                    "url": cit.get("url", ""),
                    "snippet": cit.get("text", "")
                }
                for i, cit in enumerate(response.get("citations", []))
            ]
            
        return result
    
    @staticmethod
    def _handle_llm(response: str, source: str) -> Dict[str, Any]:
        """Handles LLM provider responses."""
        return {
            "content": response,
            "sources": APIResponseHandler._extract_urls_from_text(response),
            "insights": []
        }
    
    @staticmethod
    def _handle_serp(response: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handles SERP scraping results."""
        return {
            "content": "\n".join(
                f"{res['position']}. {res['title']}\n{res['snippet']}\n{res['link']}"
                for res in response
            ),
            "sources": [
                {
                    "title": res["title"],
                    "url": res["link"],
                    "snippet": res["snippet"]
                }
                for res in response
            ],
            "insights": []
        }
    
    @staticmethod
    def _extract_urls_from_text(text: str) -> List[Dict[str, str]]:
        """Extracts URLs from text with basic context."""
        import re
        urls = re.findall(r'https?://[^\s]+', text)
        return [
            {
                "title": f"Reference {i+1}",
                "url": url,
                "snippet": "Extracted from content"
            }
            for i, url in enumerate(urls)
        ]
    
    @staticmethod
    def to_sidebar_format(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts standardized response to sidebar-compatible format.
        """
        return {
            "query": response.get("query", ""),
            "content_preview": response["content"][:200] + ("..." if len(response["content"]) > 200 else ""),
            "sources": response["sources"],
            "insights": response["insights"],
            "metadata": response.get("metadata", {})
        }
