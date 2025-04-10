import os
import json
import logging
import shutil
from typing import List, Any, Optional
from pathlib import Path

# Import core objects
from core.project import Project
from core.keyword import Keyword
from core.article_plan import ArticlePlan
from core.generated_article import GeneratedArticle

# Configuration du logging (gardé pour l'instant)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages loading and saving of project data using core objects.
    Handles interactions with the file system for projects and keywords.
    """

    def __init__(self, base_directory: str = "projects"):
        """
        Initializes the data manager.

        Args:
            base_directory: The root directory where projects are stored.
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataManager initialized with base directory: {self.base_directory.resolve()}")

    def list_projects(self) -> List[Project]:
        """
        Lists all projects found in the base directory.

        Returns:
            A list of Project objects.
        """
        projects = []
        try:
            for item in self.base_directory.iterdir():
                if item.is_dir():
                    try:
                        # Attempt to instantiate a Project object.
                        # This will implicitly load metadata.
                        project = Project(item)
                        projects.append(project)
                    except FileNotFoundError:
                        logger.warning(f"Skipping directory {item.name}: Not a valid project structure (e.g., missing metadata or incorrect setup).")
                    except Exception as e:
                        logger.error(f"Error loading project from directory {item.name}: {e}")
            logger.info(f"Found {len(projects)} projects.")
            return sorted(projects, key=lambda p: p.name)
        except Exception as e:
            logger.error(f"Error listing projects in {self.base_directory}: {e}")
            return []

    def project_exists(self, project_name: str) -> bool:
        """
        Checks if a project directory exists.

        Args:
            project_name: The name of the project.

        Returns:
            True if the project directory exists, False otherwise.
        """
        # Note: Project names might need sanitization if they contain special chars.
        # The Project class itself doesn't sanitize the top-level dir name yet.
        project_dir = self.base_directory / project_name
        return project_dir.is_dir()

    def load_project(self, project_name: str) -> Optional[Project]:
        """
        Loads a project by its name.

        Args:
            project_name: The name of the project.

        Returns:
            A Project object if found and valid, None otherwise.
        """
        project_path = self.base_directory / project_name
        if not project_path.is_dir():
            logger.warning(f"Project directory not found: {project_path}")
            return None
        try:
            project = Project(project_path)
            logger.info(f"Successfully loaded project: {project.name}")
            return project
        except FileNotFoundError:
             logger.error(f"Project directory '{project_name}' exists but seems invalid (e.g., metadata missing).")
             return None
        except Exception as e:
            logger.error(f"Error loading project '{project_name}': {e}")
            return None

    def create_project(self, project_name: str, description: str = "") -> Optional[Project]:
        """
        Creates a new project directory and metadata file.

        Args:
            project_name: The desired name for the new project.
            description: An optional description for the project metadata.

        Returns:
            The newly created Project object, or None if creation failed.
        """
        project_path = self.base_directory / project_name
        if project_path.exists():
            logger.warning(f"Project '{project_name}' already exists at {project_path}")
            # Optionally load and return the existing project?
            return self.load_project(project_name)

        try:
            project_path.mkdir(parents=True, exist_ok=True)
            # Create a Project instance which will handle metadata creation
            project = Project(project_path)
            project.metadata["description"] = description
            project.save_metadata() # Explicitly save the initial metadata
            logger.info(f"Successfully created project: {project.name}")
            return project
        except Exception as e:
            logger.error(f"Error creating project '{project_name}': {e}")
            return None


    def delete_project(self, project: Project) -> bool:
        """
        Deletes a project directory and all its contents.

        Args:
            project: The Project object to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        if not isinstance(project, Project):
             logger.error("delete_project requires a Project object.")
             return False

        project_path = project.project_path
        project_name = project.name

        if not project_path.is_dir():
            logger.warning(f"Project directory for '{project_name}' not found at {project_path}. Cannot delete.")
            return False
        try:
            shutil.rmtree(project_path)
            logger.info(f"Successfully deleted project: {project_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting project '{project_name}': {e}")
            return False

    def get_keyword(self, project: Project, keyword_name: str) -> Keyword:
         """
         Gets a Keyword object associated with a project.
         Note: This does not create files/folders, just the object.

         Args:
             project: The parent Project object.
             keyword_name: The name of the keyword.

         Returns:
             A Keyword object.
         """
         return Keyword(project, keyword_name)

    def list_keywords(self, project: Project) -> List[Keyword]:
        """
        Lists all keywords for a given project by checking subdirectories.

        Args:
            project: The Project object.

        Returns:
            A list of Keyword objects found within the project.
        """
        keyword_objects = []
        keyword_names = project.list_keywords() # Gets names from directory listing
        for name in keyword_names:
            keyword_objects.append(Keyword(project, name))
        return keyword_objects


    def save_plan(self, keyword: Keyword, plan: ArticlePlan) -> bool:
        """
        Saves an article plan for a specific keyword.

        Args:
            keyword: The Keyword object.
            plan: The ArticlePlan object containing the plan data.

        Returns:
            True if saving was successful, False otherwise.
        """
        if not isinstance(keyword, Keyword) or not isinstance(plan, ArticlePlan):
            logger.error("save_plan requires Keyword and ArticlePlan objects.")
            return False
        try:
            keyword.save_plan(plan.data)
            logger.info(f"Plan saved for keyword '{keyword.name}' in project '{keyword.project.name}'.")
            return True
        except Exception as e:
            logger.error(f"Error saving plan for keyword '{keyword.name}': {e}")
            return False

    def load_plan(self, keyword: Keyword) -> Optional[ArticlePlan]:
        """
        Loads the article plan for a specific keyword.

        Args:
            keyword: The Keyword object.

        Returns:
            An ArticlePlan object if the plan exists, None otherwise.
        """
        if not isinstance(keyword, Keyword):
            logger.error("load_plan requires a Keyword object.")
            return None
        try:
            plan_data = keyword.load_plan()
            if plan_data:
                logger.info(f"Plan loaded for keyword '{keyword.name}'.")
                return ArticlePlan(plan_data)
            else:
                logger.warning(f"Plan file not found or invalid for keyword '{keyword.name}'.")
                return None
        except Exception as e:
            logger.error(f"Error loading plan for keyword '{keyword.name}': {e}")
            return None

    def save_article(self, keyword: Keyword, article: GeneratedArticle) -> bool:
        """
        Saves the generated article content for a specific keyword.

        Args:
            keyword: The Keyword object.
            article: The GeneratedArticle object containing the article content.

        Returns:
            True if saving was successful, False otherwise.
        """
        if not isinstance(keyword, Keyword) or not isinstance(article, GeneratedArticle):
            logger.error("save_article requires Keyword and GeneratedArticle objects.")
            return False
        try:
            keyword.save_article(article.content)
            logger.info(f"Article saved for keyword '{keyword.name}' in project '{keyword.project.name}'.")
            return True
        except Exception as e:
            logger.error(f"Error saving article for keyword '{keyword.name}': {e}")
            return False

    def load_article(self, keyword: Keyword) -> Optional[GeneratedArticle]:
        """
        Loads the generated article content for a specific keyword.

        Args:
            keyword: The Keyword object.

        Returns:
            A GeneratedArticle object if the article exists, None otherwise.
        """
        if not isinstance(keyword, Keyword):
            logger.error("load_article requires a Keyword object.")
            return None
        try:
            article_content = keyword.load_article()
            if article_content is not None: # Check for None explicitly, empty string is valid
                logger.info(f"Article loaded for keyword '{keyword.name}'.")
                return GeneratedArticle(article_content)
            else:
                logger.warning(f"Article file not found for keyword '{keyword.name}'.")
                return None
        except Exception as e:
            logger.error(f"Error loading article for keyword '{keyword.name}': {e}")
            return None

    # --- Commented out Export/Import ---
    # These need significant rework to align with the new object model
    # and potentially configuration management.

    # def export_project(self, project: Project, export_format: str = "json") -> Optional[bytes]:
    #     """ Exports a project using the Project object. """
    #     # Implementation needs to gather data from Project and Keyword objects
    #     logger.warning("export_project is not fully implemented yet.")
    #     return None

    # def import_project(self, project_name: str, project_data: bytes, import_format: str = "json") -> bool:
    #     """ Imports a project, creating Project and Keyword objects. """
    #     # Implementation needs to parse data and create/save objects
    #     logger.warning("import_project is not fully implemented yet.")
    #     return False


# Example Usage (for testing purposes)
if __name__ == "__main__":
    print("Testing DataManager...")
    # Use a temporary directory for testing to avoid modifying real projects
    test_base_dir = "temp_test_projects"
    manager = DataManager(base_directory=test_base_dir)

    # Clean up previous test run if necessary
    if Path(test_base_dir).exists():
        print(f"Cleaning up previous test directory: {test_base_dir}")
        shutil.rmtree(test_base_dir)

    # Test Project Creation
    print("\n--- Testing Project Creation ---")
    proj_name = "My Test Project"
    created_project = manager.create_project(proj_name, description="A project for testing.")
    if created_project:
        print(f"Project created: {created_project}")
        print(f"Metadata: {created_project.metadata}")
    else:
        print(f"Failed to create project: {proj_name}")
        exit() # Stop test if creation fails

    # Test Listing Projects
    print("\n--- Testing Project Listing ---")
    projects = manager.list_projects()
    print(f"Listed projects: {[p.name for p in projects]}")
    assert len(projects) == 1
    assert projects[0].name == proj_name

    # Test Loading Project
    print("\n--- Testing Project Loading ---")
    loaded_project = manager.load_project(proj_name)
    if loaded_project:
        print(f"Project loaded: {loaded_project}")
        assert loaded_project == created_project # Check equality based on path
    else:
        print(f"Failed to load project: {proj_name}")

    # Test Keyword Operations
    print("\n--- Testing Keyword Operations ---")
    kw_name = "Test Keyword One"
    keyword_obj = manager.get_keyword(loaded_project, kw_name)
    print(f"Got keyword object: {keyword_obj}")

    # Test Plan Save/Load
    print("\n--- Testing Plan Save/Load ---")
    plan_data = {"title": "Test Plan Title", "steps": [1, 2, 3]}
    article_plan_obj = ArticlePlan(plan_data)
    save_plan_success = manager.save_plan(keyword_obj, article_plan_obj)
    print(f"Save plan successful? {save_plan_success}")
    assert save_plan_success

    loaded_plan = manager.load_plan(keyword_obj)
    if loaded_plan:
        print(f"Loaded plan: {loaded_plan}")
        print(f"Loaded plan data: {loaded_plan.data}")
        assert loaded_plan == article_plan_obj # Check equality based on data
    else:
        print("Failed to load plan.")

    # Test Article Save/Load
    print("\n--- Testing Article Save/Load ---")
    article_content = "# Test Article\n\nContent goes here."
    generated_article_obj = GeneratedArticle(article_content)
    save_article_success = manager.save_article(keyword_obj, generated_article_obj)
    print(f"Save article successful? {save_article_success}")
    assert save_article_success

    loaded_article = manager.load_article(keyword_obj)
    if loaded_article:
        print(f"Loaded article: {loaded_article}")
        print(f"Loaded article content snippet: {loaded_article.content[:30]}...")
        assert loaded_article == generated_article_obj # Check equality based on content
    else:
        print("Failed to load article.")

    # Test Listing Keywords
    print("\n--- Testing Keyword Listing ---")
    keywords_list = manager.list_keywords(loaded_project)
    print(f"Listed keywords: {[kw.name for kw in keywords_list]}")
    assert len(keywords_list) == 1
    assert keywords_list[0].name == keyword_obj.path.name # Compare sanitized name used for path

    # Test Project Deletion
    print("\n--- Testing Project Deletion ---")
    delete_success = manager.delete_project(loaded_project)
    print(f"Delete project successful? {delete_success}")
    assert delete_success
    assert not Path(test_base_dir, proj_name).exists()

    # Clean up test directory
    print(f"\nCleaning up test directory: {test_base_dir}")
    shutil.rmtree(test_base_dir)
    print("DataManager tests completed.")
