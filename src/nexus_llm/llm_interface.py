from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from .config import Settings
from .factory import LLMFactory


class LLMInterface:
    """
    The primary interface for executing tasks with a configured LLM.

    This class provides a three-tiered API for interacting with a language model:
    1.  Text Generation: For simple request/response string operations.
    2.  Structured Generation: For receiving validated Pydantic models.
    3.  Raw Invocation: For advanced, low-level access to the LangChain client.

    An instance of this class is configured for a single LLM provider and is
    designed to be stateless, making it safe for concurrent use.
    """

    def __init__(self, settings: Settings, provider_key: str):
        """
        Initializes the LLMInterface.

        Args:
            settings: The validated Pydantic Settings object for the library.
            provider_key: The key of the specific LLM provider to use for this
                          interface instance (e.g., 'google_gemini').
        """
        factory = LLMFactory(settings)
        self.llm_client: BaseChatModel = factory.create_client(provider_key)

    def _build_messages(
        self,
        system_prompt: str,
        user_input: str,
        human_prompt_template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> List[BaseMessage]:
        """A private helper to construct the full message list for an LLM call."""
        all_vars = {**(variables or {}), "user_input": user_input}
        messages: List[BaseMessage] = []

        messages.append(SystemMessage(content=system_prompt.format(**all_vars)))

        if few_shot_examples:
            for example in few_shot_examples:
                messages.append(
                    HumanMessage(content=example["user"].format(**all_vars))
                )
                messages.append(
                    AIMessage(content=example["assistant"].format(**all_vars))
                )

        if human_prompt_template:
            final_human_content = human_prompt_template.format(**all_vars)
        else:
            final_human_content = user_input

        messages.append(HumanMessage(content=final_human_content))
        return messages

    def generate_text(
        self,
        system_prompt: str,
        user_input: str,
        human_prompt_template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generates a raw text response.

        Args:
            system_prompt: The system message, which can be a template.
            user_input: The raw user input.
            human_prompt_template: An optional template for the human message.
            variables: Optional dictionary for formatting templates.
            few_shot_examples: Optional list of examples for few-shot prompting.
            **kwargs: Additional arguments passed to the LLM's `invoke` method.

        Returns:
            The string content of the LLM's response.
        """
        messages = self._build_messages(
            system_prompt,
            user_input,
            human_prompt_template,
            variables,
            few_shot_examples,
        )
        response = self.llm_client.invoke(messages, **kwargs)
        return response.content

    async def agenerate_text(
        self,
        system_prompt: str,
        user_input: str,
        human_prompt_template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronously generates a raw text response."""
        messages = self._build_messages(
            system_prompt,
            user_input,
            human_prompt_template,
            variables,
            few_shot_examples,
        )
        response = await self.llm_client.ainvoke(messages, **kwargs)
        return response.content

    def generate_structured(
        self,
        response_model: Type[BaseModel],
        system_prompt: str,
        user_input: str,
        human_prompt_template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Generates a structured response conforming to a Pydantic model.

        Args:
            response_model: The Pydantic model for the desired output.
            (Other parameters are the same as `generate_text`)

        Returns:
            An instantiated Pydantic model object.
        """
        messages = self._build_messages(
            system_prompt,
            user_input,
            human_prompt_template,
            variables,
            few_shot_examples,
        )
        structured_llm = self.llm_client.with_structured_output(response_model)
        return structured_llm.invoke(messages, **kwargs)

    async def agenerate_structured(
        self,
        response_model: Type[BaseModel],
        system_prompt: str,
        user_input: str,
        human_prompt_template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """Asynchronously generates a structured response."""
        messages = self._build_messages(
            system_prompt,
            user_input,
            human_prompt_template,
            variables,
            few_shot_examples,
        )
        structured_llm = self.llm_client.with_structured_output(response_model)
        return await structured_llm.ainvoke(messages, **kwargs)

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """
        Provides direct, low-level access to the LLM client's `invoke` method.

        Args:
            messages: A pre-constructed list of LangChain BaseMessage objects.
            **kwargs: Additional arguments for the `invoke` method.

        Returns:
            The raw BaseMessage object from the LLM.
        """
        return self.llm_client.invoke(messages, **kwargs)

    async def ainvoke(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """Asynchronously provides direct access to the `ainvoke` method."""
        return await self.llm_client.ainvoke(messages, **kwargs)
