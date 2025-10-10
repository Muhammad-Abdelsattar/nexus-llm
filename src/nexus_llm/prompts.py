import json
from pathlib import Path
from typing import Protocol, List, Dict, Any, Union

import aiofiles

from .exceptions import TemplateNotFoundError


class PromptProvider(Protocol):

    def get_template(self, key: str) -> str:
        """Retrieves a string-based template by its key (synchronously)."""
        ...

    async def aget_template(self, key: str) -> str:
        """Retrieves a string-based template by its key (asynchronously)."""
        ...

    def get_few_shot_examples(self, key: str) -> List[Dict[str, Any]]:
        """Retrieves structured few-shot examples by their key (synchronously)."""
        ...

    async def aget_few_shot_examples(self, key: str) -> List[Dict[str, Any]]:
        """Retrieves structured few-shot examples by their key (asynchronously)."""
        ...


class FileSystemPromptProvider:
    """
    The default implementation of the PromptProvider protocol.

    It loads resources from a local filesystem relative to a given base path.
    The 'key' for a resource is its relative path from the base directory.
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        Initializes the provider with a root directory for all prompts.

        Args:
            base_path: The path to the root directory containing prompt files
                       and subdirectories.
        """
        self.base_path = Path(base_path)
        if not self.base_path.is_dir():
            raise FileNotFoundError(
                f"Prompt base directory not found at: {self.base_path.resolve()}"
            )

    def _resolve_path(self, key: str) -> Path:
        """Safely resolves a key to a full, validated file path."""
        if ".." in Path(key).parts:
            raise ValueError("Invalid key: path traversal is not allowed.")

        resource_path = self.base_path.joinpath(key).resolve()

        if not resource_path.is_file():
            raise TemplateNotFoundError(
                f"Resource with key '{key}' not found at path: {resource_path}"
            )

        return resource_path

    def get_template(self, key: str) -> str:
        """
        Reads and returns the text content of a file specified by the key.
        Example key: 'sql_agent/system.prompt'
        """
        path = self._resolve_path(key)
        return path.read_text("utf-8")

    async def aget_template(self, key: str) -> str:
        """Asynchronously reads and returns the text content of a file."""
        path = self._resolve_path(key)
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            return await f.read()

    def get_few_shot_examples(self, key: str) -> List[Dict[str, Any]]:
        """
        Reads a JSON file specified by the key and returns its content.
        Example key: 'sql_agent/few_shot_examples.json'
        """
        content = self.get_template(key)
        try:
            examples = json.loads(content)
            if not isinstance(examples, list):
                raise ValueError("Few-shot examples file must contain a JSON list.")
            return examples
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from key '{key}': {e}") from e

    async def aget_few_shot_examples(self, key: str) -> List[Dict[str, Any]]:
        """Asynchronously reads and parses a JSON file of few-shot examples."""
        content = await self.aget_template(key)
        try:
            examples = json.loads(content)
            if not isinstance(examples, list):
                raise ValueError("Few-shot examples file must contain a JSON list.")
            return examples
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from key '{key}': {e}") from e
