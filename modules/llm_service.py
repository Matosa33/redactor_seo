import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import requests
import httpx # Added for API calls to list models
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import os

# Configure logging
try:
    import config
    log_level = getattr(config, 'LOG_LEVEL', logging.INFO)
except ImportError:
    log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Define specific exception types for retries if needed
class QuotaError(Exception):
    """Custom exception for quota/rate limit errors."""
    pass

class LLMService:
    """
    Service for interacting with various LLM providers.
    Handles API calls, basic error handling, and retries.
    Configuration (API keys, model details) should be passed during initialization.
    """

    SUPPORTED_PROVIDERS = ["OpenAI", "Anthropic", "Google", "DeepSeek"] # Ensure DeepSeek is here

    def __init__(self, api_keys: Dict[str, str]):
        """
        Initializes the LLM service.
        """
        if not isinstance(api_keys, dict):
            raise ValueError("api_keys must be a dictionary.")

        self.api_keys: Dict[str, str] = api_keys
        self._clients: Dict[str, Any] = {} # Store initialized library clients

        # Availability flags based on library imports (needed for generation)
        self.provider_library_available: Dict[str, bool] = {
            "openai": False, "anthropic": False, "google": False,
            "deepseek": True # DeepSeek uses httpx, always available if installed
        }
        self._import_and_initialize_clients()

    def _import_and_initialize_clients(self):
        """Imports libraries and initializes clients based on available API keys."""
        # OpenAI
        if self.api_keys.get("openai"):
            try:
                from openai import AsyncOpenAI
                self._clients["openai"] = AsyncOpenAI(api_key=self.api_keys["openai"])
                self.provider_library_available["openai"] = True
                logger.info("OpenAI client initialized.")
            except ImportError: logger.warning("OpenAI library not found.")
            except Exception as e: logger.error(f"Failed to initialize OpenAI client: {e}")

        # Anthropic
        if self.api_keys.get("anthropic"):
            try:
                from anthropic import AsyncAnthropic
                self._clients["anthropic"] = AsyncAnthropic(api_key=self.api_keys["anthropic"])
                self.provider_library_available["anthropic"] = True
                logger.info("Anthropic client initialized.")
            except ImportError: logger.warning("Anthropic library not found.")
            except Exception as e: logger.error(f"Failed to initialize Anthropic client: {e}")

        # Google Generative AI
        if self.api_keys.get("google"):
            try:
                import google.generativeai as genai
                self.genai = genai
                self.genai.configure(api_key=self.api_keys["google"])
                self.provider_library_available["google"] = True
                logger.info("Google Generative AI configured.")
            except ImportError: logger.warning("Google Generative AI library not found.")
            except Exception as e: logger.error(f"Failed to configure Google Generative AI: {e}")

    def update_api_keys(self, api_keys: Dict[str, str]):
        """Updates API keys and re-initializes corresponding clients."""
        for provider, key in api_keys.items():
            provider_lower = provider.lower()
            # Use SUPPORTED_PROVIDERS for validation
            supported_lower = [p.lower() for p in self.SUPPORTED_PROVIDERS]
            if key and provider_lower in supported_lower:
                self.api_keys[provider_lower] = key
            elif not key and provider_lower in self.api_keys:
                 self.api_keys[provider_lower] = ""
                 if provider_lower in self._clients: del self._clients[provider_lower]
        self._import_and_initialize_clients()
        logger.info("API keys updated and clients re-initialized.")

    def get_available_providers(self) -> List[str]:
        """Returns a list of providers for which API key is available."""
        available = []
        logger.debug(f"Checking available providers. Keys available: {list(self.api_keys.keys())}")
        for provider_lower, key in self.api_keys.items():
            if key:
                 # Ensure consistency with SUPPORTED_PROVIDERS capitalization
                 provider_name_standard = next((p for p in self.SUPPORTED_PROVIDERS if p.lower() == provider_lower), None)
                 if provider_name_standard:
                      logger.debug(f"Provider '{provider_name_standard}' has key and is supported. Adding.")
                      available.append(provider_name_standard) # Append the correctly capitalized name
                 else:
                      logger.debug(f"Provider '{provider_lower}' has key but not in SUPPORTED_PROVIDERS list.")
            else:
                 logger.debug(f"Provider '{provider_lower}' skipped, no API key.")
        logger.debug(f"Returning available providers: {available}")
        return available

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """
        Lists available models for a given provider using direct API calls.
        """
        provider_lower = provider_name.lower()
        api_key = self.api_keys.get(provider_lower)

        if not api_key:
            logger.warning(f"API key for {provider_name} not found. Cannot list models.")
            return []

        models = []
        try:
            if provider_lower == "openai":
                client = self._clients.get("openai")
                if client and self.provider_library_available.get("openai"):
                    try:
                         model_list = client.models.list()
                         models = sorted([model.id for model in model_list if "gpt" in model.id or "instruct" in model.id])
                    except Exception as e:
                         logger.error(f"Failed to list OpenAI models via client: {e}. Trying API call.")
                         headers = {"Authorization": f"Bearer {api_key}"}
                         response = httpx.get("https://api.openai.com/v1/models", headers=headers, timeout=10.0)
                         response.raise_for_status()
                         data = response.json()
                         models = sorted([model.get("id") for model in data.get("data", []) if model.get("id")])
                else:
                    headers = {"Authorization": f"Bearer {api_key}"}
                    response = httpx.get("https://api.openai.com/v1/models", headers=headers, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()
                    models = sorted([model.get("id") for model in data.get("data", []) if model.get("id")])

            elif provider_lower == "anthropic":
                headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
                response = httpx.get("https://api.anthropic.com/v1/models", headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                models = sorted([model.get("id") for model in data.get("data", []) if model.get("id")])

            elif provider_lower == "google":
                if self.provider_library_available["google"]:
                    models = sorted([m.name for m in self.genai.list_models() if 'generateContent' in m.supported_generation_methods])
                    models = [f"models/{m}" if not m.startswith("models/") else m for m in models]
                else:
                    logger.warning("Google client library not available for model listing.")
                    models = ["models/gemini-1.5-pro-latest", "models/gemini-pro"]

            elif provider_lower == "deepseek":
                headers = {"Authorization": f"Bearer {api_key}"}
                response = httpx.get("https://api.deepseek.com/models", headers=headers, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                models = sorted([model.get("id") for model in data.get("data", []) if model.get("id")])

            if not models:
                 logger.warning(f"API call for {provider_name} returned an empty model list. Returning fallback.")
                 if provider_lower == "openai": models = ["gpt-4-turbo", "gpt-3.5-turbo"]
                 elif provider_lower == "anthropic": models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
                 elif provider_lower == "google": models = ["models/gemini-1.5-pro-latest", "models/gemini-pro"]
                 elif provider_lower == "deepseek": models = ["deepseek-chat", "deepseek-coder"]

            logger.debug(f"Found models for {provider_name}: {models}")
            return models

        except httpx.RequestError as e: logger.error(f"HTTP Request error listing models for {provider_name}: {e}")
        except httpx.HTTPStatusError as e: logger.error(f"HTTP Status error listing models for {provider_name}: {e.response.status_code} - {e.response.text[:100]}")
        except Exception as e: logger.error(f"General error listing models for provider {provider_name}: {e}")

        logger.warning(f"Returning minimal fallback list for {provider_name} due to error.")
        if provider_lower == "openai": return ["gpt-4-turbo", "gpt-3.5-turbo"]
        if provider_lower == "anthropic": return ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
        if provider_lower == "google": return ["models/gemini-1.5-pro-latest", "models/gemini-pro"]
        if provider_lower == "deepseek": return ["deepseek-chat", "deepseek-coder"]
        return []

    def _is_provider_library_available_for_generation(self, provider_name: str) -> bool:
        """Checks if the library needed for *generation* for a provider is available."""
        provider_lower = provider_name.lower()
        if provider_lower == "deepseek": return True
        return self.provider_library_available.get(provider_lower, False)

    def _check_quota_error(self, error: Exception) -> bool:
        """Helper to check if an exception message indicates a quota/rate limit error."""
        error_str = str(error).lower()
        quota_error_messages = [
            "resource has been exhausted", "quota exceeded", "rate limit",
            "api rate limit", "insufficient_quota", "limit", "429"
        ]
        return any(msg in error_str for msg in quota_error_messages)

    _retry_strategy = retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, httpx.RequestError, ConnectionError, TimeoutError, QuotaError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )

    @_retry_strategy
    async def generate_content(
        self,
        prompt: str,
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates content using the specified provider and model. Includes retry logic.
        Requires the necessary client library (or httpx for DeepSeek) for the chosen provider.
        """
        provider_lower = provider.lower()
        logger.info(f"Attempting content generation with {provider} ({model})")

        if not self.api_keys.get(provider_lower):
             raise ValueError(f"API key for provider '{provider}' is missing.")
        if not self._is_provider_library_available_for_generation(provider_lower):
             raise ValueError(f"Required library/dependency for provider '{provider}' generation is not available.")

        try:
            from modules.api_response_handler import APIResponseHandler
            content = ""

            if provider_lower == "openai":
                content = await self._generate_openai(prompt, model, temperature, max_tokens, system_prompt)
            elif provider_lower == "anthropic":
                content = await self._generate_anthropic(prompt, model, temperature, max_tokens, system_prompt)
            elif provider_lower == "google":
                content = await self._generate_gemini(prompt, model, temperature, max_tokens, system_prompt)
            elif provider_lower == "deepseek":
                content = await self._generate_deepseek(prompt, model, temperature, max_tokens, system_prompt)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            return APIResponseHandler.standardize_response(
                raw_response=content, source=provider_lower, query=prompt[:100]
            )
        except ImportError:
             logger.error("APIResponseHandler module not found.")
             return {"content": content if 'content' in locals() else "Error: APIResponseHandler missing", "sources": [], "insights": [], "metadata": {}}
        except Exception as e:
            logger.error(f"Error during generation with {provider} ({model}): {e}")
            if self._check_quota_error(e):
                 raise QuotaError(f"Quota/Rate limit exceeded for {provider}: {e}") from e
            raise

    async def _generate_openai(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        client = self._clients.get("openai")
        if not client: raise ValueError("OpenAI client not initialized.")
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
        response = await client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return response.choices[0].message.content or ""

    async def _generate_anthropic(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        client = self._clients.get("anthropic")
        if not client: raise ValueError("Anthropic client not initialized.")
        messages = [{"role": "user", "content": prompt}]
        system_message = str(system_prompt) if system_prompt else None
        response = await client.messages.create(model=model, messages=messages, system=system_message, temperature=temperature, max_tokens=max_tokens)
        if response.content and isinstance(response.content, list) and len(response.content) > 0:
             return response.content[0].text
        return ""

    async def _generate_gemini(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        if not self.provider_library_available["google"]: raise ValueError("Google Generative AI library not available.")
        generation_config = {"temperature": temperature, "max_output_tokens": max_tokens}
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        if not model.startswith("models/"): model = f"models/{model}"
        gemini_model = self.genai.GenerativeModel(model_name=model, generation_config=generation_config)
        response = await asyncio.to_thread(lambda: gemini_model.generate_content(full_prompt))
        try:
            return response.text
        except ValueError as e:
            if hasattr(response, 'candidates') and not response.candidates:
                 feedback = getattr(response, 'prompt_feedback', 'N/A')
                 logger.warning(f"Gemini response blocked or empty (no candidates). Feedback: {feedback}. Prompt hash: {hash(prompt)}")
                 return f"Error: Gemini response blocked (possibly due to safety filters). Feedback: {feedback}"
            else:
                 logger.error(f"Unexpected ValueError processing Gemini response: {e}")
                 raise
        except Exception as e:
             logger.error(f"Error processing Gemini response: {e}")
             raise

    async def _generate_deepseek(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> str:
        api_key = self.api_keys.get("deepseek");
        if not api_key: raise ValueError("DeepSeek API key missing.")
        api_endpoint = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
        data = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(api_endpoint, headers=headers, json=data, timeout=120.0)
                response.raise_for_status()
                result = response.json()
        except Exception as e: logger.error(f"DeepSeek API call failed: {e}"); raise
        if "choices" not in result or not result["choices"]: raise ValueError(f"Invalid response format from DeepSeek: {json.dumps(result)}")
        return result["choices"][0]["message"]["content"] or ""

    # --- Fallback Logic (Consider refactoring out) ---
    async def complete(self, prompt: str, model_settings: Dict[str, Any]) -> str:
        logger.warning("The 'complete' method has specific fallback logic; consider refactoring.")
        cost_effective_options = [
            {"provider": "OpenAI", "model": "gpt-3.5-turbo", "priority": 1},
            {"provider": "Google", "model": "models/gemini-1.5-flash-latest", "priority": 2}
        ]
        primary_provider = model_settings.get("provider")
        primary_model = model_settings.get("model")
        try:
            response_dict = await self.generate_content(prompt=prompt, **model_settings)
            return response_dict.get("content", "")
        except QuotaError as e:
            logger.warning(f"Quota error with {primary_provider} ({primary_model}): {e}. Attempting fallback.")
            alternatives = sorted([opt for opt in cost_effective_options if not (opt["provider"] == primary_provider and opt["model"] == primary_model)], key=lambda x: x["priority"])
            if not alternatives: logger.error("No fallback providers."); raise e
            for option in alternatives:
                fb_provider, fb_model = option["provider"], option["model"]
                if self.api_keys.get(fb_provider.lower()):
                    try:
                        logger.info(f"Trying fallback: {fb_provider} with model {fb_model}")
                        fallback_args = {k:v for k,v in model_settings.items() if k not in ['provider','model']}
                        fb_response_dict = await self.generate_content(prompt=prompt, provider=fb_provider, model=fb_model, **fallback_args)
                        return fb_response_dict.get("content", "")
                    except Exception as fb_error: logger.warning(f"Fallback failed: {fb_error}"); continue
                else: logger.warning(f"Fallback provider {fb_provider} not available (missing API key).")
            logger.error("All fallback providers failed."); raise e
        except Exception as e: logger.error(f"Non-quota error: {e}"); raise

    async def generate_with_fallback(self, prompt: str, primary_provider: str, primary_model: str, fallback_providers: Optional[List[Tuple[str, str]]] = None, **kwargs) -> Tuple[Dict[str, Any], str]:
        logger.warning("The 'generate_with_fallback' method adds complexity; consider simplifying.")
        try:
            result_dict = await self.generate_content(prompt, primary_provider, primary_model, **kwargs)
            return result_dict, primary_provider
        except Exception as e:
            logger.warning(f"Primary provider {primary_provider} failed: {e}. Initiating fallback.")
            if fallback_providers is None: logger.error("Explicit fallback_providers list required."); raise e
            for fb_provider, fb_model in fallback_providers:
                if self.api_keys.get(fb_provider.lower()):
                    try:
                        logger.info(f"Attempting fallback with {fb_provider} ({fb_model})...")
                        result_dict = await self.generate_content(prompt, fb_provider, fb_model, **kwargs)
                        logger.info(f"Fallback successful with {fb_provider}.")
                        return result_dict, fb_provider
                    except Exception as fb_error: logger.warning(f"Fallback failed: {fb_error}"); continue
                else: logger.warning(f"Fallback provider {fb_provider} not available (missing API key).")
            logger.error("All providers failed."); raise Exception("All providers failed.") from e

    async def generate_image(
        self,
        prompt: str,
        provider: str, # Expecting "Google" for now
        model: str, # Expecting "gemini-2.0-flash-exp-image-generation" or similar
        # Add other potential parameters like aspect ratio, style, etc. if needed later
    ) -> Optional[bytes]: # Return image bytes or None on error
        """
        Generates an image using the specified provider and model.
        Currently supports Google Gemini image generation models.
        """
        provider_lower = provider.lower()
        logger.info(f"Attempting image generation with {provider} ({model}) for prompt: '{prompt[:50]}...'")

        if provider_lower != "google":
            logger.error(f"Image generation currently only supported for Google, not {provider}.")
            return None
        if not self.api_keys.get("google"):
            raise ValueError("API key for Google is missing.")
        if not self.provider_library_available["google"]:
            raise ValueError("Google Generative AI library not available.")

        try:
            # Use the specific model name provided by the user
            image_model = self.genai.GenerativeModel(model_name=model)

            # Generate image content using the prompt
            # The Google SDK handles image generation via generate_content for specific models
            response = await asyncio.to_thread(lambda: image_model.generate_content(prompt))

            # Process the response to extract image data
            # Based on Google SDK examples, image data is often in response.parts[0].blob.data
            if response.parts:
                # Find the first part that looks like an image blob
                image_part = next((part for part in response.parts if hasattr(part, 'blob') and hasattr(part.blob, 'mime_type') and 'image' in part.blob.mime_type), None)
                if image_part and hasattr(image_part.blob, 'data'):
                    logger.info(f"Successfully generated image for prompt: '{prompt[:50]}...'")
                    return image_part.blob.data # Return the image bytes
                else:
                    logger.warning(f"No valid image blob found in Gemini response parts for prompt: '{prompt[:50]}...'. Parts: {response.parts}")
                    return None
            elif hasattr(response, 'candidates') and not response.candidates:
                 # Handle cases where generation might be blocked (safety filters, etc.)
                 feedback = getattr(response, 'prompt_feedback', 'N/A')
                 logger.warning(f"Gemini image generation blocked or empty (no candidates). Feedback: {feedback}. Prompt: '{prompt[:50]}...'")
                 # Optionally, return an error message or raise a specific exception here
                 return None
            else:
                # Log unexpected response structure
                logger.warning(f"Unexpected response structure from Gemini image generation: {response}")
                return None

        except Exception as e:
            logger.error(f"Error generating image with {provider} ({model}): {e}", exc_info=True)
            # Consider specific error handling (e.g., for quota) if needed
            return None


# Example Usage
if __name__ == "__main__":
    async def main():
        print("Testing LLMService...")
        from dotenv import load_dotenv
        load_dotenv()
        test_api_keys = {k: os.getenv(f"{k.upper()}_API_KEY") for k in ["openai", "anthropic", "google", "deepseek"]}
        test_api_keys = {k: v for k, v in test_api_keys.items() if v}

        if not test_api_keys: print("No API keys found. Skipping live tests."); return
        print(f"Found API keys for: {list(test_api_keys.keys())}")
        llm_service = LLMService(api_keys=test_api_keys)
        available_providers = llm_service.get_available_providers()
        print(f"Available providers: {available_providers}")
        if not available_providers: print("No providers available. Cannot run tests."); return

        print("\n--- Testing Model Listing ---")
        for provider in available_providers:
             print(f"Models for {provider}:")
             models = llm_service.get_models_for_provider(provider)
             print(f"  -> Found {len(models)} models." + (f" Example: {models[0]}" if models else ""))

        test_provider = available_providers[0]
        models_for_provider = llm_service.get_models_for_provider(test_provider)
        test_model = models_for_provider[0] if models_for_provider else None
        if not test_model: print(f"No models for {test_provider}. Skipping generation."); return

        print(f"\n--- Testing basic generation with {test_provider} ({test_model}) ---")
        try:
            result_dict = await llm_service.generate_content(prompt="Short poem about coding.", provider=test_provider, model=test_model, max_tokens=50)
            print(f"Generated Poem:\n{result_dict.get('content')}")
        except Exception as e: print(f"Error: {e}"); import traceback; traceback.print_exc()

        print(f"\n--- Testing system prompt ({test_provider} / {test_model}) ---")
        try:
            result_pirate = await llm_service.generate_content(prompt="Weather today?", provider=test_provider, model=test_model, system_prompt="Speak like a pirate.", max_tokens=50)
            print(f"Pirate Response:\n{result_pirate.get('content')}")
        except Exception as e: print(f"Error: {e}")

        if len(available_providers) > 1:
             print("\n--- Testing Fallback ---")
             primary_provider_fb, fallback_provider_fb = available_providers[0], available_providers[1]
             primary_models = llm_service.get_models_for_provider(primary_provider_fb)
             primary_model_fb = primary_models[0] if primary_models else None
             fallback_models = llm_service.get_models_for_provider(fallback_provider_fb)
             fallback_model_fb = fallback_models[0] if fallback_models else None

             if primary_model_fb and fallback_model_fb:
                 original_keys = llm_service.api_keys.copy()
                 llm_service.update_api_keys({primary_provider_fb.lower(): "FAKE_KEY"})
                 print(f"Simulating failure for {primary_provider_fb}. Fallback to {fallback_provider_fb} ({fallback_model_fb})...")
                 try:
                     result_fb_dict, provider_used_fb = await llm_service.generate_with_fallback(prompt="Explain quantum physics simply.", primary_provider=primary_provider_fb, primary_model=primary_model_fb, fallback_providers=[(fallback_provider_fb, fallback_model_fb)], max_tokens=100)
                     print(f"Fallback Result (used {provider_used_fb}):\n{result_fb_dict.get('content')}")
                     assert provider_used_fb == fallback_provider_fb
                 except Exception as e: print(f"Error: {e}")
                 finally: llm_service.update_api_keys(original_keys)
             else: print("Skipping fallback test - couldn't get models for both providers.")
        else: print("\nSkipping fallback test - only one provider.")

    if __name__ == "__main__":
        asyncio.run(main())
