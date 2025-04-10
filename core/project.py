import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class Project:
    """Represents a project within the application."""

    def __init__(self, project_path: Path):
        """
        Initializes a Project instance.

        Args:
            project_path: The root path of the project directory.
        """
        if not isinstance(project_path, Path):
            project_path = Path(project_path)

        self.project_path: Path = project_path
        self.metadata_path: Path = self.project_path / "metadata.json"
        self.keywords_path: Path = self.project_path / "keywords"
        self.metadata: Dict[str, Any] = {}

        if not self.project_path.is_dir():
            raise FileNotFoundError(f"Project directory not found: {self.project_path}")

        self.load_metadata()

    @property
    def name(self) -> str:
        """Returns the name of the project (directory name)."""
        return self.project_path.name

    def load_metadata(self) -> None:
        """Loads project metadata from metadata.json."""
        try:
            if self.metadata_path.is_file():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                # Initialize with default metadata if file doesn't exist
                self.metadata = {"name": self.name, "description": ""}
                self.save_metadata() # Create the file
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {self.metadata_path}. Using default metadata.")
            self.metadata = {"name": self.name, "description": ""}
        except Exception as e:
            print(f"Error loading metadata for project {self.name}: {e}")
            self.metadata = {"name": self.name, "description": ""} # Fallback

    def save_metadata(self) -> None:
        """Saves project metadata to metadata.json."""
        try:
            self.project_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata for project {self.name}: {e}")

    def list_keywords(self) -> List[str]:
        """Lists the names of all keywords (subdirectories) in the keywords directory."""
        keywords = []
        if self.keywords_path.is_dir():
            for item in self.keywords_path.iterdir():
                if item.is_dir():
                    keywords.append(item.name)
        return sorted(keywords)

    def get_keyword_path(self, keyword_name: str) -> Path:
        """Returns the path for a specific keyword directory."""
        # Basic sanitization - replace characters unsafe for directory names
        # A more robust slugify function might be needed depending on expected keyword inputs
        safe_keyword_name = keyword_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        return self.keywords_path / safe_keyword_name

    def __repr__(self) -> str:
        return f"Project(name='{self.name}', path='{self.project_path}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Project):
            return NotImplemented
        return self.project_path == other.project_path

    def __hash__(self) -> int:
        return hash(self.project_path)

# Example Usage (can be removed or placed under if __name__ == "__main__":)
if __name__ == "__main__":
    # Assuming 'projects/Mon_Projet_SEO' exists relative to the script execution location
    # Adjust the path as necessary for testing
    try:
        project_dir = Path("../projects/Mon_Projet_SEO") # Adjust path if running from core/
        if not project_dir.exists():
             project_dir = Path("projects/Mon_Projet_SEO") # Adjust path if running from root

        if project_dir.exists():
            proj = Project(project_dir)
            print(f"Loaded project: {proj.name}")
            print(f"Metadata: {proj.metadata}")
            print(f"Keywords: {proj.list_keywords()}")
            kw_path = proj.get_keyword_path("Test Keyword")
            print(f"Path for 'Test Keyword': {kw_path}")
        else:
            print(f"Project directory not found for testing: {project_dir.resolve()}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during testing: {e}")
