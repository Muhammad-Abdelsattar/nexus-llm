import os
from pathlib import Path
from typing import Dict, Any, Union, Optional

from omegaconf import OmegaConf, ValidationError as OmegaValidationError
from pydantic import (
    BaseModel,
    Field,
    ValidationError as PydanticValidationError,
    model_validator,
)

from .exceptions import ConfigurationError


class LLMProviderSettings(BaseModel):
    type: Optional[str] = Field(
        None, description="A convenient alias for a popular provider."
    )
    class_path: Optional[str] = Field(
        None, description="The full Python import path for custom providers."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for the LLM class."
    )

    @model_validator(mode="after")
    def check_type_or_class_path_exclusive(self) -> "LLMProviderSettings":
        if self.type is not None and self.class_path is not None:
            raise ValueError("Provide either 'type' or 'class_path', not both.")
        if self.type is None and self.class_path is None:
            raise ValueError("Either 'type' or 'class_path' must be provided.")
        return self


class Settings(BaseModel):
    llm_providers: Dict[str, LLMProviderSettings]


def load_settings(source: Union[str, Path, Dict[str, Any]]) -> Settings:
    """
    Loads, resolves, and validates the library's settings from a source.

    This function accepts either a file path or a dictionary as its source,
    providing flexible configuration options. It performs three critical steps:
    1.  Loads the configuration from the source (file or dict).
    2.  Uses OmegaConf to resolve any environment variable interpolations
        (e.g., `${env:MY_API_KEY}`). This works for both source types.
    3.  Validates the resolved configuration against the Pydantic `Settings` model.

    Args:
        source: The configuration source. Can be a file path (str or Path)
                or a dictionary.

    Returns:
        A validated `Settings` object.

    Raises:
        FileNotFoundError: If the source is a path that does not exist.
        TypeError: If the source is not a path or a dictionary.
        ConfigurationError: If the configuration is invalid.
    """
    source_name = "the provided source"
    conf: OmegaConf

    if isinstance(source, (str, Path)):
        settings_path = Path(source)
        source_name = f"file '{settings_path.name}'"
        if not settings_path.is_file():
            raise FileNotFoundError(
                f"Settings file not found at: {settings_path.resolve()}"
            )
        conf = OmegaConf.load(settings_path)
    elif isinstance(source, dict):
        source_name = "the provided dictionary"
        conf = OmegaConf.create(source)
    else:
        raise TypeError(
            f"Source must be a file path or a dictionary, not {type(source).__name__}."
        )

    try:
        # register the 'env' resolver if it doesn't exist
        if not OmegaConf.has_resolver("env"):
            OmegaConf.register_new_resolver("env", os.getenv)

        # convert to a standard Python dict, resolving all interpolations
        resolved_config = OmegaConf.to_container(conf, resolve=True)

    except OmegaValidationError as e:
        raise ConfigurationError(
            f"Error resolving variables in {source_name}: {e}"
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load or parse settings from {source_name}: {e}"
        ) from e

    try:
        validated_settings = Settings.model_validate(resolved_config)
        return validated_settings

    except PydanticValidationError as e:
        raise ConfigurationError(
            f"Configuration from {source_name} is invalid:\n{e}"
        ) from e
