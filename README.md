# Simple LLM Interface

A straightforward, high-level wrapper around LangChain for personal convenience.

### Philosophy

This package was created as a utility to standardize the way I interact with Large Language Models in my higher-level projects. It provides a clean, validated, and repeatable pattern for common LLM tasks.

It is **not** intended to be a replacement for LangChain or any other framework. Instead, it's a thin, opinionated layer that bundles configuration, prompt management, and execution into a simple interface that suits a specific workflow.

### Installation

The library is designed for a simple, all-in-one installation that includes support for all built-in LLM providers.

```bash
pip install "git+https://github.com/muhammad-abdelsattar/nexus-llm.git#egg=nexus-llm"
```

### Features

- **Configuration-Driven**: Uses a single YAML file and Pydantic for type-safe validation.
- **Convenient Provider Aliases**: Simple aliases for popular providers, with the flexibility to use any custom LangChain-compatible model.
- **Flexible Prompting**: A protocol-based `PromptProvider` for loading templates from any source.
- **Three-Tier API**: A powerful `LLMInterface` with methods for text, structured data, and raw message invocation.
- **First-Class Async Support**: Symmetrical `async` methods for all I/O operations.

### Configuration

You can configure providers using either a convenient `type` alias or a full `class_path`.

#### Built-in Provider Aliases (`type`)

This library includes built-in support for the following providers.

| Alias (`type`) | LangChain Class Path                            |
| :------------- | :---------------------------------------------- |
| `google`       | `langchain_google_genai.ChatGoogleGenerativeAI` |
| `openai`       | `langchain_openai.ChatOpenAI`                   |
| `azure`        | `langchain_openai.AzureChatOpenAI`              |
| `anthropic`    | `langchain_anthropic.ChatAnthropic`             |
| `groq`         | `langchain_groq.ChatGroq`                       |
| `ollama`       | `langchain_ollama.ChatOllama`                   |

#### Using a Custom Provider (`class_path`)

If a provider is not in the list, or you have your own custom model class, you can use the `class_path` property.

**Step 1: Install the necessary package**
First, ensure the Python package for your custom model is installed in your environment.

For example, if you want to install the langchain-aws bedrock, you can run:

```bash
pip install langchain-aws
```

**Step 2: Configure `settings.yaml`**
Reference the full import path in your settings file. The library will dynamically import and use it.

```yaml
# settings.yaml
llm_providers:
    bedrock:
        # No 'type' here, we use the full class_path
        class_path: "langchain_aws.bedrock.ChatBedrock"
        params:
            api_key: "${env:SPECIAL_API_KEY}"
            temperature: 0.5
```

**Step 3: Use the provider**

Now that you have the provider configured, you can use it in your LLM interface.

```python
from nexus_llm import LLMInterface, load_settings_from_yaml

settings = load_settings_from_yaml("settings.yaml")
bedrock_interface = LLMInterface(settings, "bedrock")

response = bedrock_interface.agenerate_text(
    system_prompt="You are a helpful assistant.",
    user_input="What is the capital of France?",
)
print(f"Response: {response}")
```

### Quick Start Example

This example demonstrates using both a built-in alias and a custom provider.

**1. Project Structure & Setup**

```my_project/
├── .env
├── settings.yaml
└── main.py
```

```bash
pip install "git+https://github.com/muhammad-abdelsattar/nexus-llm.git#egg=nexus-llm"
```

**2. Configuration (`settings.yaml`)**

```yaml
llm_providers:
    google_default:
        type: "google"
        params:
            model: "gemini-2.5-flash"
            google_api_key: "${env:GOOGLE_API_KEY}"
```

**3. Main Application (`main.py`)**

```python
import asyncio
from dotenv import load_dotenv
from nexus_llm import LLMInterface, load_settings_from_yaml

async def main():
    load_dotenv()
    settings = load_settings_from_yaml("settings.yaml")

    # use the built-in 'google' provider
    google_interface = LLMInterface(settings, "google_default")
    print("|> Calling Google Provider ...")
    response = await google_interface.agenerate_text(
        system_prompt="You are a helpful assistant.",
        user_input="What is the capital of France?",
    )
    print(f"Response: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

### Disclaimer

This is primarily a tool built to streamline my own development process. While it is designed to be robust for my use cases, it may not cover all possible scenarios. Feel free to adapt it for your own needs.
