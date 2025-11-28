# PutergenAI: Python SDK for Puter.js

[![Python Version](https://img.shields.io/badge/python-3.11--3.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/badge/pypi-2.0.1-blue)](https://pypi.org/project/putergenai/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://img.shields.io/badge/tests-passing-brightgreen)
[![Security](https://img.shields.io/badge/security-policy-important)](SECURITY.md)

Asynchronous Python client for interacting with the Puter.com API. This SDK provides access to Puter's AI models (including OpenAI GPT, Claude, Mistral, Grok, DeepSeek, and more), file system operations, image generation, OCR, and text-to-speech capabilities.

## Features

- **AI Chat Completions**: Support for 200+ AI models from various providers (OpenAI, Anthropic, Mistral, xAI, DeepSeek, Google, TogetherAI, OpenRouter, and more)
- **File System Operations**: Read, write, and delete files on Puter.com
- **Image Generation**: Create images from text prompts using various models
- **OCR**: Extract text from images
- **Text-to-Speech**: Convert text to MP3 audio
- **Streaming Support**: Real-time streaming responses for chat completions
- **Fallback & Retry Logic**: Automatic model fallback and retry mechanisms for reliability

## Installation

Install PutergenAI using pip:

```bash
pip install putergenai
```

Or from source:

```bash
git clone https://github.com/Nerve11/putergenai.git
cd putergenai
pip install .
```

## Quick Start

```python
import asyncio
from putergenai import PuterClient

async def main():
    async with PuterClient() as client:
        # Login to Puter.com
        await client.login("your_username", "your_password")

        # AI Chat with GPT-4o
        result = await client.ai_chat(
            prompt="Hello, how are you?",
            options={"model": "gpt-4o", "stream": False}
        )

        print(result["response"]["result"]["message"]["content"])

asyncio.run(main())
```

## Authentication

Authentication is required for most operations. Use your Puter.com username and password:

```python
await client.login("your_username", "your_password")
```

## AI Chat Completions

### Synchronous Chat

```python
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

result = await client.ai_chat(messages=messages, options={"model": "gpt-4o"})
print(result["response"]["result"]["message"]["content"])
```

### Streaming Chat

```python
async def stream_chat():
    messages = [{"role": "user", "content": "Tell me a story"}]

    gen = await client.ai_chat(
        messages=messages,
        options={"model": "claude-opus-4.5", "stream": True}
    )

    print("Assistant: ", end='', flush=True)
    async for content, model in gen:
        print(content, end='', flush=True)
    print()

asyncio.run(stream_chat())
```

### Vision/Chat with Images

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }
]

result = await client.ai_chat(messages=messages, options={"model": "gpt-4o"})
```

## Supported Models

The SDK supports models from:

- **OpenAI**: GPT-5, GPT-4o, o3 series
- **Anthropic**: Claude Opus, Sonnet, Haiku series
- **Mistral**: Large, Medium, Small models
- **xAI**: Grok series
- **DeepSeek**: Chat and Reasoner models
- **Google**: Gemini series
- **TogetherAI**: Various models including LLMs and image generation
- **OpenRouter**: Access to 100+ models from different providers

For the complete list of supported models, refer to the `model_to_driver` mapping in the source code.

## File System Operations

### Write a File

```python
await client.fs_write("hello.txt", "Hello, Puter!")
```

### Read a File

```python
content = await client.fs_read("hello.txt")
print(content.decode('utf-8'))
```

### Delete a File

```python
await client.fs_delete("hello.txt")
```

## Image Generation

```python
image_url = await client.ai_txt2img(
    "A beautiful sunset over mountains",
    model="pollinations-image"
)
print(image_url)
```

## OCR (Image to Text)

```python
text = await client.ai_img2txt("https://example.com/image.png")
print(text)

# Or with file upload
with open("image.png", "rb") as f:
    text = await client.ai_img2txt(f)
```

## Text-to-Speech

```python
audio_bytes = await client.ai_txt2speech("Hello, world!")
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

## Advanced Options

### Custom Parameters

```python
options = {
    "model": "gpt-5",
    "temperature": 1,
    "max_tokens": 2000,
    "stream": True
}

gen = await client.ai_chat(
    messages=messages,
    options=options,
    test_mode=False,  # Use test mode for debugging
    strict_model=True  # Enforce exact model usage
)
```

### Tools and Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    }
]

messages = [{"role": "user", "content": "What's the weather in Paris?"}]

result = await client.ai_chat(
    messages=messages,
    options={"model": "gpt-4o", "tools": tools}
)

# Check for tool calls in result
if "tool_calls" in result["response"]["result"]["message"]:
    # Handle tool calls...
    pass
```

## Error Handling

The SDK includes built-in error handling and retry logic:

- Automatic retries for transient failures
- Model fallback when preferred models are unavailable
- SSL verification options for debugging network issues

```python
try:
    result = await client.ai_chat(messages=messages)
except ValueError as e:
    print(f"Error: {e}")
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `examples/example.py`: Interactive chat terminal application
- `examples/example-ui.py`: GUI chat application with CustomTkinter

Run an example:

```bash
python examples/example.py
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Security

Security issues can be reported via the [Security Policy](SECURITY.md).

## License

PutergenAI is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ for the Puter.com platform
