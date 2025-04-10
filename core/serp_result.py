import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import hashlib
import datetime

class SERPResult:
    """Represents the cached SERP (Search Engine Results Page) data for a query."""

    # Define a base cache directory (can be overridden)
    DEFAULT_CACHE_DIR = Path("./cache/serp")

    def __init__(self, query: str, data: Optional[Dict[str, Any]] = None, cache_dir: Optional[Path] = None):
        """
        Initializes a SERPResult instance.

        Args:
            query: The search query string.
            data: The SERP data dictionary (optional). If provided, it's used directly.
                  If None, attempts to load from cache.
            cache_dir: The directory to store/load cache files (defaults to DEFAULT_CACHE_DIR).
        """
        if not isinstance(query, str) or not query:
            raise ValueError("SERPResult requires a non-empty query string.")

        self.query: str = query
        self.cache_dir: Path = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_path: Path = self._get_cache_path(query, self.cache_dir)
        self._data: Optional[Dict[str, Any]] = None

        if data is not None:
            if not isinstance(data, dict):
                 raise ValueError("Initial data must be a dictionary.")
            self._data = data
            # Optionally save provided data to cache immediately? Or require explicit save?
            # self.save() # Let's require explicit save for now.
        else:
            self.load() # Attempt to load from cache if no data provided

    @staticmethod
    def _get_cache_filename(query: str) -> str:
        """Generates a safe filename hash for the query."""
        # Use SHA256 for a robust hash, less prone to collisions than simple slugify
        query_bytes = query.encode('utf-8')
        hash_object = hashlib.sha256(query_bytes)
        return f"{hash_object.hexdigest()}.json"

    @classmethod
    def _get_cache_path(cls, query: str, cache_dir: Path) -> Path:
        """Constructs the full path for the cache file."""
        filename = cls._get_cache_filename(query)
        return cache_dir / filename

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        """Returns the raw dictionary data of the SERP result, loading if necessary."""
        if self._data is None:
            self.load()
        return self._data

    def load(self) -> bool:
        """
        Loads SERP data from the cache file.

        Returns:
            True if loading was successful, False otherwise.
        """
        self._data = None # Reset data before attempting load
        if not self.cache_path.is_file():
            return False
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            # Optionally add timestamp check here for cache expiry
            return True
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from cache file: {self.cache_path}")
            return False
        except Exception as e:
            print(f"Error loading SERP cache for query '{self.query}': {e}")
            return False

    def save(self, data_to_save: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves the SERP data to the cache file.

        Args:
            data_to_save: Optional dictionary to save. If None, saves the current internal data.
        """
        if data_to_save is not None:
             if not isinstance(data_to_save, dict):
                 raise ValueError("Data to save must be a dictionary.")
             self._data = data_to_save # Update internal data if new data is provided

        if self._data is None:
            print(f"Warning: No data to save for query '{self.query}'.")
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True) # Ensure cache directory exists
            # Add metadata like query and timestamp to the saved data
            save_payload = {
                "_metadata": {
                    "query": self.query,
                    "cached_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                },
                "results": self._data # Store the actual results under a 'results' key
            }
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(save_payload, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving SERP cache for query '{self.query}': {e}")

    def exists_in_cache(self) -> bool:
        """Checks if a cache file exists for this query."""
        return self.cache_path.is_file()

    # Example accessors (adjust based on expected SERP data structure)
    @property
    def organic_results(self) -> Optional[List[Dict[str, Any]]]:
        """Returns the list of organic results, if available in the data."""
        if self.data and isinstance(self.data.get("results"), dict): # Check if 'results' key holds the actual data
             return self.data["results"].get("organic_results")
        elif self.data: # Fallback if data is directly the results dict (legacy?)
             return self.data.get("organic_results")
        return None

    @property
    def timestamp(self) -> Optional[str]:
         """Returns the timestamp when the data was cached, if available."""
         if self.data and isinstance(self.data.get("_metadata"), dict):
             return self.data["_metadata"].get("cached_at")
         return None


    def __repr__(self) -> str:
        status = "loaded" if self._data is not None else "not loaded"
        return f"SERPResult(query='{self.query}', status='{status}', cache_file='{self.cache_path.name}')"

    def __eq__(self, other: object) -> bool:
        # Equality based on query and potentially data content
        if not isinstance(other, SERPResult):
            return NotImplemented
        # Simple equality check based on query and loaded data state might suffice
        # For deep comparison, compare self._data == other._data
        return self.query == other.query and self._data == other._data

    def __hash__(self) -> int:
        # Hash based on the query primarily
        return hash(self.query)

    @classmethod
    def load_from_cache(cls, cache_path: Path) -> Optional['SERPResult']:
        """
        Loads SERP data from a specific cache file path.

        Args:
            cache_path: The full path to the cache file.

        Returns:
            A SERPResult instance if loading is successful, None otherwise.
        """
        if not cache_path.is_file():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_payload = json.load(f)

            # Extract query and results data from the saved payload
            metadata = cached_payload.get("_metadata", {})
            query = metadata.get("query")
            results_data = cached_payload.get("results") # Actual results are under 'results' key

            if not query or results_data is None:
                 print(f"Warning: Cache file {cache_path} is missing query or results data in payload.")
                 return None

            # Create an instance with the loaded data
            # Pass the cache_dir based on the cache_path's parent
            instance = cls(query=query, data=results_data, cache_dir=cache_path.parent)
            # Manually set the timestamp if needed, though it's mainly for expiry checks
            # instance.timestamp = metadata.get("cached_at") # Need to adjust how timestamp is stored/accessed if needed directly
            return instance

        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from cache file: {cache_path}")
            return None
        except Exception as e:
            print(f"Error loading SERP cache from path {cache_path}: {e}")
            return None


# Example Usage
if __name__ == "__main__":
    test_query = "example search query"
    test_cache_dir = Path("./temp_serp_cache") # Use a temporary dir for testing

    # Ensure clean state
    if test_cache_dir.exists():
        import shutil
        shutil.rmtree(test_cache_dir)

    print(f"Using test cache directory: {test_cache_dir.resolve()}")

    # 1. Test initialization without data (should try to load - fails first time)
    print("\n--- Test 1: Init without data (cache miss) ---")
    serp_res1 = SERPResult(test_query, cache_dir=test_cache_dir)
    print(serp_res1)
    print(f"Exists in cache? {serp_res1.exists_in_cache()}")
    print(f"Data loaded? {serp_res1.data is not None}")

    # 2. Test saving data
    print("\n--- Test 2: Saving data ---")
    sample_serp_data = {
        "search_information": {"query_displayed": test_query},
        "organic_results": [
            {"title": "Result 1", "link": "http://example.com/1", "snippet": "Snippet 1"},
            {"title": "Result 2", "link": "http://example.com/2", "snippet": "Snippet 2"},
        ]
    }
    serp_res1.save(sample_serp_data)
    print(f"Exists in cache after save? {serp_res1.exists_in_cache()}")
    print(f"Cache file path: {serp_res1.cache_path}")

    # 3. Test initialization with cache hit
    print("\n--- Test 3: Init with cache hit ---")
    serp_res2 = SERPResult(test_query, cache_dir=test_cache_dir)
    print(serp_res2)
    print(f"Data loaded? {serp_res2.data is not None}")
    if serp_res2.data:
        print(f"Loaded timestamp: {serp_res2.timestamp}")
        print(f"Number of organic results: {len(serp_res2.organic_results or [])}")
        # Accessing the actual results nested under 'results'
        actual_results = serp_res2.data.get("results", {})
        print(f"First organic result title: {actual_results.get('organic_results', [{}])[0].get('title')}")


    # 4. Test initialization with provided data
    print("\n--- Test 4: Init with provided data ---")
    provided_data = {"organic_results": [{"title": "Provided Result"}]}
    serp_res3 = SERPResult("another query", data=provided_data, cache_dir=test_cache_dir)
    print(serp_res3)
    print(f"Data loaded? {serp_res3.data is not None}")
    print(f"Organic results: {serp_res3.organic_results}")
    print(f"Exists in cache (before save)? {serp_res3.exists_in_cache()}")
    serp_res3.save() # Save the provided data
    print(f"Exists in cache (after save)? {serp_res3.exists_in_cache()}")


    # 5. Test error handling
    print("\n--- Test 5: Error Handling ---")
    try:
        invalid_query = SERPResult("")
    except ValueError as e:
        print(f"Caught expected error for empty query: {e}")

    try:
        invalid_data = SERPResult("test", data=[1, 2, 3])
    except ValueError as e:
        print(f"Caught expected error for invalid initial data type: {e}")

    # Clean up
    # if test_cache_dir.exists():
    #     shutil.rmtree(test_cache_dir)
    #     print(f"\nCleaned up test cache directory: {test_cache_dir}")
