import importlib
from typing import TYPE_CHECKING

from .exceptions import ProviderNotFoundError, ConfigurationError

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from .config import Settings


BUILT_IN_PROVIDERS = {
    "google": "langchain_google_genai.ChatGoogleGenerativeAI",
    "openai": "langchain_openai.ChatOpenAI",
    "azure": "langchain_openai.AzureChatOpenAI",
    "anthropic": "langchain_anthropic.ChatAnthropic",
    "groq": "langchain_groq.ChatGroq",
    "ollama": "langchain_ollama.ChatOllama",
}


class LLMFactory:
    """
    A factory class responsible for creating LangChain LLM client instances.
    """

    def __init__(self, settings: "Settings"):
        """
        Initializes the factory with the application's validated settings.

        Args:
            settings: The validated Pydantic Settings object.
        """
        self.settings = settings

    def create_client(self, provider_key: str) -> "BaseChatModel":
        """
        Creates and returns an LLM client based on the provider key.

        This is the main public method of the factory. It orchestrates the
        entire client creation process.

        Args:
            provider_key: The key from the settings file (e.g., 'google_default').

        Returns:
            An instantiated and configured LangChain LLM client.

        Raises:
            ProviderNotFoundError: If the provider_key is not found in the settings.
            ConfigurationError: If the class cannot be imported or instantiated
                                due to configuration issues.
        """
        if provider_key not in self.settings.llm_providers:
            available_keys = list(self.settings.llm_providers.keys())
            raise ProviderNotFoundError(
                f"Provider key '{provider_key}' not found in settings. "
                f"Available providers: {available_keys}"
            )

        provider_config = self.settings.llm_providers[provider_key]

        if provider_config.type:
            if provider_config.type in BUILT_IN_PROVIDERS:
                provider_config.class_path = BUILT_IN_PROVIDERS[provider_config.type]
            else:
                valid_types = list(BUILT_IN_PROVIDERS.keys())
                raise ConfigurationError(
                    f"Invalid provider type '{provider_config.type}' for provider key '{provider_key}'. "
                    f"Available built-in types are: {valid_types}"
                )

        try:
            module_path, class_name = provider_config.class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            llm_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ConfigurationError(
                f"Failed to import LLM class for '{provider_key}'. "
                f"Check the class_path '{provider_config.class_path}' and ensure the "
                f"required package is installed. Error: {e}"
            ) from e

        try:
            return llm_class(**provider_config.params)
        except TypeError as e:
            raise ConfigurationError(
                f"Failed to instantiate LLM client for '{provider_key}'. "
                f"Check if the parameters in your settings file match the constructor "
                f"for the class '{provider_config.class_path}'. Error: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"An unexpected error occurred while creating the client for '{provider_key}': {e}"
            ) from e
