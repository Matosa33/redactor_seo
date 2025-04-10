from typing import Dict, Any, Optional, List

class ArticlePlan:
    """Represents the structured plan for generating an article, typically loaded from plan.json."""

    def __init__(self, data: Dict[str, Any]):
        """
        Initializes an ArticlePlan instance.

        Args:
            data: The dictionary representation of the plan, usually loaded from JSON.
                  Expected structure might include keys like 'title', 'sections', 'keywords', etc.
        """
        if not isinstance(data, dict):
            raise ValueError("ArticlePlan must be initialized with a dictionary.")
        self._data: Dict[str, Any] = data

    @property
    def data(self) -> Dict[str, Any]:
        """Returns the raw dictionary data of the plan."""
        return self._data

    # Example properties to access common plan elements (adjust based on actual plan structure)
    @property
    def title(self) -> Optional[str]:
        """Returns the title from the plan, if available."""
        return self._data.get("title")

    @property
    def sections(self) -> Optional[List[Dict[str, Any]]]: # Assuming sections are dicts
        """Returns the list of sections from the plan, if available."""
        return self._data.get("sections")

    def get(self, key: str, default: Any = None) -> Any:
        """Provides dictionary-like access to plan data."""
        return self._data.get(key, default)

    def __repr__(self) -> str:
        # Provide a concise representation, perhaps showing the title or number of sections
        title_repr = f"title='{self.title}'" if self.title else "no title"
        sections_repr = f"{len(self.sections)} sections" if self.sections is not None else "no sections"
        return f"ArticlePlan({title_repr}, {sections_repr})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArticlePlan):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        # Hashing based on the content requires the dictionary to be hashable (e.g., no lists directly)
        # A simple approach is to hash the string representation, but be aware of potential collisions
        # or hash based on a primary identifier like title if guaranteed unique and present.
        # For simplicity, using the raw data's hash might fail if it contains unhashable types.
        # Let's use the repr for a basic hash, acknowledging limitations.
        return hash(repr(self._data)) # Or hash(self.title) if title is reliable

# Example Usage
if __name__ == "__main__":
    sample_plan_data = {
        "title": "Understanding Article Plans",
        "target_audience": "Developers",
        "keywords": ["planning", "structure", "content"],
        "sections": [
            {"heading": "Introduction", "content_points": ["Define article plan", "Importance"]},
            {"heading": "Core Components", "content_points": ["Title", "Sections", "Keywords"]},
            {"heading": "Conclusion", "content_points": ["Summary", "Next steps"]}
        ]
    }

    plan = ArticlePlan(sample_plan_data)

    print(f"Plan Representation: {plan}")
    print(f"Plan Title: {plan.title}")
    print(f"Plan Sections: {plan.sections}")
    print(f"Accessing 'target_audience': {plan.get('target_audience')}")
    print(f"Accessing a non-existent key: {plan.get('tone')}")

    # Test equality
    plan2 = ArticlePlan(sample_plan_data.copy())
    print(f"Plan == Plan2? {plan == plan2}")

    plan3_data = sample_plan_data.copy()
    plan3_data["title"] = "A Different Plan"
    plan3 = ArticlePlan(plan3_data)
    print(f"Plan == Plan3? {plan == plan3}")

    # Test initialization error
    try:
        invalid_plan = ArticlePlan(["not", "a", "dict"])
    except ValueError as e:
        print(f"\nCaught expected error: {e}")
