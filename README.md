# PutergenAI

[![Python Version](https://img.shields.io/badge/python-3.11--3.14-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-3.5.0-blue)](https://pypi.org/project/putergenai/)

Asynchronous Python SDK for [Puter.com](https://puter.com) API — an open-source cloud platform with AI capabilities, file storage, and privacy-first design.

## Features

- **200+ AI Models**: GPT, Claude, Gemini, Mistral, Grok, DeepSeek, and more
- **File System**: Cloud storage operations (read/write/delete/copy/move/mkdir/readdir/stat/space/upload)
- **Key-Value Store**: Full KV API (set/get/list + add/incr/decr/update/remove/expire/flush)
- **Image Generation**: Text-to-image with multiple providers
- **OCR**: Extract text from images
- **Text-to-Speech**: Convert text to MP3 audio
- **Driver Calls**: Low-level `drivers.call()` equivalent for unsupported endpoints
- **Streaming Support**: Real-time chat completions
- **Async/Await**: Modern Python async architecture

## Quick Start

Install the package:

```bash
pip install putergenai
```

Basic usage:

```python
import asyncio
from putergenai import PuterClient

async def main():
    async with PuterClient() as client:
        await client.login("your_username", "your_password")
        
        result = await client.ai_chat(
            prompt="Hello, how are you?",
            options={"model": "gpt-4o"}
        )
        
        print(result["response"]["result"]["message"]["content"])

asyncio.run(main())
```

## CLI Usage

PuterGenAI now includes a command-line interface!

```bash
# Login
puter login

# Chat
puter chat --model gpt-4o

# List Models
puter models

# KV Store Operations
puter kv set mykey "hello world"
puter kv get mykey
```

## Documentation

- **[Installation Guide](docs/installation.md)** — Setup instructions and dependencies
- **[API Reference](docs/api.md)** — Complete method documentation

## Examples

### Streaming Chat

```python
async def stream_example():
    async with PuterClient() as client:
        await client.login("username", "password")
        
        stream = await client.ai_chat(
            prompt="Tell me a story",
            options={"model": "claude-opus-4.5", "stream": True}
        )
        
        async for chunk, model in stream:
            print(chunk, end='', flush=True)

asyncio.run(stream_example())
```

### File Operations

```python
async def file_example():
    async with PuterClient() as client:
        await client.login("username", "password")
        
        await client.fs_write("test.txt", "Hello, Puter!")
        content = await client.fs_read("test.txt")
        print(content.decode('utf-8'))
        await client.fs_delete("test.txt")

asyncio.run(file_example())
```

### Key-Value Store

```python
async def kv_example():
    async with PuterClient() as client:
        await client.login("username", "password")
        
        await client.kv_set("my_key", "my_value")
        value = await client.kv_get("my_key")
        print(value)

asyncio.run(kv_example())
```

### Vision (Image Analysis)

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
}]

result = await client.ai_chat(messages=messages, options={"model": "gpt-4o"})
```

### Image Generation

```python
image_url = await client.ai_txt2img(
    "A beautiful sunset over mountains",
    model="pollinations-image"
)
print(image_url)
```

### OCR (Image to Text)

```python
text = await client.ai_img2txt("https://example.com/image.png")
print(text)
```

### Text-to-Speech

```python
audio_bytes = await client.ai_txt2speech("Hello, world!")
with open("output.mp3", "wb") as f:
    f.write(audio_bytes)
```

## GUI Application

Try the interactive GUI built with CustomTkinter:

```bash
python examples/example-ui.py
```

Features:
- Secure login with encrypted credentials
- Multi-model chat interface
- Image generation with 4 free APIs
- Async operations (non-blocking UI)
- API key management via system keychain

## Supported Models

The SDK supports models from multiple providers:

| Provider | Examples |
|----------|----------|
| **OpenAI** | GPT‑5, GPT‑5 Mini, GPT‑4o, o3/o4‑mini |
| **Anthropic** | Claude Opus 4.1, Claude 3.7 Sonnet, Claude 3.5 Sonnet, Claude 3 Haiku |
| **Google** | Gemini 2.5 Pro, Gemini 2.5 Flash |
| **Mistral** | Mistral Large, Mistral Small, Pixtral, Codestral |
| **xAI** | Grok‑3, Grok‑2 Vision |
| **DeepSeek** | DeepSeek Chat, DeepSeek Reasoner |
| **MiniMax** | MiniMax M2, MiniMax M1 |
| **TogetherAI** | Various LLMs |
| **OpenRouter** | 100+ models |

For the complete list, see https://developer.puter.com/ai/models/ or run `puter models` / `await client.get_available_models()`.

## Security

- **Encrypted Storage**: API keys stored via system keychain or Fernet encryption
- **No Plain-Text Credentials**: Session tokens instead of passwords
- **SSL/TLS**: All connections secured by default
- **Input Validation**: Sanitized user inputs

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit your changes
4. Add tests for new functionality
5. Submit a pull request

Run tests:

```bash
python -m unittest discover tests
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Links

- **PyPI**: [pypi.org/project/putergenai](https://pypi.org/project/putergenai/)
- **GitHub**: [github.com/Nerve11/putergenai](https://github.com/Nerve11/putergenai)
- **Puter Platform**: [puter.com](https://puter.com)

---

**Maintainers**: [Nerve11](https://github.com/Nerve11) • [KernFerm](https://github.com/KernFerm)  
**Version**: 3.5.0 • Built with ❤️ for the Puter.com platform
