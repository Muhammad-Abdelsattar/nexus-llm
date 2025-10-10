class NexusError(Exception):
    """Base exception for the NexusLLM library."""

    pass


class ConfigurationError(NexusError):
    """Raised for configuration-related issues."""

    pass


class ProviderNotFoundError(NexusError):
    """Raised when a specified LLM provider key is not found in the settings."""

    pass


class TemplateNotFoundError(NexusError):
    """Raised when a prompt template file cannot be found."""

    pass
