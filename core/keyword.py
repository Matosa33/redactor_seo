import json
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import issues with Project
if TYPE_CHECKING:
    from .project import Project

class Keyword:
    """Represents a specific keyword task within a project."""

    def __init__(self, project: 'Project', keyword_name: str):
        """
        Initializes a Keyword instance.

        Args:
            project: The parent Project object.
            keyword_name: The name of the keyword.
        """
        self.project: 'Project' = project
        self.name: str = keyword_name
        # Use the project's method to get the standardized path
        self.path: Path = self.project.get_keyword_path(self.name)
        self.plan_path: Path = self.path / "plan.json"
        self.article_path: Path = self.path / "article.md"

    def _ensure_dir_exists(self) -> None:
        """Ensures the keyword directory exists."""
        self.path.mkdir(parents=True, exist_ok=True)

    def load_plan(self) -> Optional[Dict[str, Any]]:
        """Loads the article plan from plan.json."""
        if not self.plan_path.is_file():
            return None
        try:
            with open(self.plan_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from {self.plan_path}.")
            return None
        except Exception as e:
            print(f"Error loading plan for keyword '{self.name}': {e}")
            return None

    def save_plan(self, plan_data: Dict[str, Any]) -> None:
        """Saves the article plan to plan.json."""
        self._ensure_dir_exists()
        try:
            with open(self.plan_path, 'w', encoding='utf-8') as f:
                json.dump(plan_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving plan for keyword '{self.name}': {e}")

    def load_article(self) -> Optional[str]:
        """Loads the generated article content from article.md."""
        if not self.article_path.is_file():
            return None
        try:
            with open(self.article_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading article for keyword '{self.name}': {e}")
            return None

    def save_article(self, article_content: str) -> None:
        """Saves the generated article content to article.md."""
        self._ensure_dir_exists()
        try:
            with open(self.article_path, 'w', encoding='utf-8') as f:
                f.write(article_content)
        except Exception as e:
            print(f"Error saving article for keyword '{self.name}': {e}")

    def exists(self) -> bool:
        """Checks if the keyword directory exists."""
        return self.path.is_dir()

    def has_plan(self) -> bool:
        """Checks if plan.json exists."""
        return self.plan_path.is_file()

    def has_article(self) -> bool:
        """Checks if article.md exists."""
        return self.article_path.is_file()

    def __repr__(self) -> str:
        return f"Keyword(name='{self.name}', project='{self.project.name}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Keyword):
            return NotImplemented
        return self.project == other.project and self.name == other.name

    def __hash__(self) -> int:
        return hash((self.project, self.name))

# Example Usage (can be removed or placed under if __name__ == "__main__":)
if __name__ == "__main__":
    from .project import Project # Import Project for testing

    try:
        # Adjust path as necessary
        project_dir = Path("../projects/Mon_Projet_SEO")
        if not project_dir.exists():
             project_dir = Path("projects/Mon_Projet_SEO")

        if project_dir.exists():
            proj = Project(project_dir)
            print(f"Testing with project: {proj.name}")

            # Test with an existing keyword (adjust name if needed)
            existing_kw_name = "Construire_un_RAG_pour_application_LLM"
            if existing_kw_name in proj.list_keywords():
                kw_existing = Keyword(proj, existing_kw_name)
                print(f"\nTesting existing keyword: {kw_existing.name}")
                print(f"Path: {kw_existing.path}")
                print(f"Exists? {kw_existing.exists()}")
                print(f"Has plan? {kw_existing.has_plan()}")
                plan = kw_existing.load_plan()
                # print(f"Plan content (first 100 chars): {str(plan)[:100]}...")
                print(f"Has article? {kw_existing.has_article()}")
                article = kw_existing.load_article()
                # print(f"Article content (first 100 chars): {str(article)[:100]}...")
            else:
                print(f"\nSkipping existing keyword test: '{existing_kw_name}' not found.")

            # Test creating/saving for a new keyword
            new_kw_name = "Test_Keyword_Save"
            kw_new = Keyword(proj, new_kw_name)
            print(f"\nTesting new keyword: {kw_new.name}")
            print(f"Path: {kw_new.path}")
            print(f"Exists before save? {kw_new.exists()}")

            test_plan = {"title": "Test Plan", "sections": ["Intro", "Body", "Conclusion"]}
            kw_new.save_plan(test_plan)
            print(f"Saved plan. Has plan now? {kw_new.has_plan()}")
            loaded_plan = kw_new.load_plan()
            print(f"Loaded plan matches saved? {loaded_plan == test_plan}")

            test_article = "# Test Article\n\nThis is the content."
            kw_new.save_article(test_article)
            print(f"Saved article. Has article now? {kw_new.has_article()}")
            loaded_article = kw_new.load_article()
            print(f"Loaded article matches saved? {loaded_article == test_article}")

            # Clean up test files (optional)
            # print(f"Cleaning up test files for {new_kw_name}...")
            # if kw_new.plan_path.exists(): kw_new.plan_path.unlink()
            # if kw_new.article_path.exists(): kw_new.article_path.unlink()
            # if kw_new.path.exists(): kw_new.path.rmdir() # Only removes if empty

        else:
            print(f"Project directory not found for testing: {project_dir.resolve()}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
