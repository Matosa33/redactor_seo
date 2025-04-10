from typing import Optional

class GeneratedArticle:
    """Represents the generated article content, typically loaded from article.md."""

    def __init__(self, content: str):
        """
        Initializes a GeneratedArticle instance.

        Args:
            content: The string content of the article (usually Markdown).
        """
        if not isinstance(content, str):
            raise ValueError("GeneratedArticle must be initialized with a string.")
        self._content: str = content

    @property
    def content(self) -> str:
        """Returns the raw string content of the article."""
        return self._content

    def __len__(self) -> int:
        """Returns the length (number of characters) of the article content."""
        return len(self._content)

    def __repr__(self) -> str:
        # Show the first ~50 characters as a preview
        preview = (self._content[:50] + '...') if len(self._content) > 53 else self._content
        return f"GeneratedArticle(content='{preview.replace(chr(10), ' ')}', length={len(self)})" # Replace newlines for cleaner repr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeneratedArticle):
            return NotImplemented
        return self._content == other._content

    def __hash__(self) -> int:
        return hash(self._content)

# Example Usage
if __name__ == "__main__":
    sample_content = """# My Awesome Article

This is the first paragraph. It explains the main topic.

## Section 1

More details here.
- Point 1
- Point 2

## Section 2

Further elaboration.
"""

    article = GeneratedArticle(sample_content)

    print(f"Article Representation: {article}")
    print(f"Article Length: {len(article)}")
    print(f"Full Content:\n---\n{article.content}\n---")

    # Test equality
    article2 = GeneratedArticle(sample_content)
    print(f"Article == Article2? {article == article2}")

    article3 = GeneratedArticle("Different content.")
    print(f"Article == Article3? {article == article3}")

    # Test initialization error
    try:
        invalid_article = GeneratedArticle(12345)
    except ValueError as e:
        print(f"\nCaught expected error: {e}")
