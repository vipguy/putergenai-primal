# PutergenAI: Python SDK for Puter.js

[![Python Version](https://img.shields.io/badge/python-3.11--3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/badge/pypi-1.5.1-blue)](https://pypi.org/project/putergenai/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://img.shields.io/badge/tests-passing-brightgreen)
[![Security](https://img.shields.io/badge/security-policy-important)](SECURITY.md)

## Overview

PutergenAI is a lightweight, robust Python SDK for interacting with the **Puter.js API**, an open-source cloud operating system focused on privacy and AI capabilities. This SDK provides a clean interface for:

- **AI Chat Completions** – Support for multiple models (GPT-5, Claude, Grok, DeepSeek, etc.) with automatic fallback
- **File System Operations** – Cloud storage integration (read/write/delete)
- **Utility AI Functions** – Image generation, OCR, text-to-speech
- **Vision Models** – Multi-modal AI with image support
- **Streaming** – Real-time response streaming for low-latency applications

**Version**: 1.5.1 (Latest - November 26, 2025)  
**Status**: ✅ Production Ready

## Key Features

✨ **Multi-Model Support** – Access 50+ AI models with intelligent fallback  
🔐 **Secure** – Sanitized inputs, encrypted key storage, no credential logging  
⚡ **Streaming** – Real-time response generation for interactive applications  
🌐 **File Operations** – Cloud-native file management  
🎨 **Vision & Multimedia** – Image-to-text, text-to-image, text-to-speech  
🛡️ **Error Resilience** – Automatic retries, model fallback, graceful degradation  
📦 **Zero-Dependency Core** – Core SDK requires only `requests`  
🐍 **Python 3.11+** – Modern Python support with full type hints

## Installation

### From PyPI (Recommended)
```bash
pip install putergenai==1.5.1
```

### For Development
```bash
git clone https://github.com/Nerve11/putergenai.git
cd putergenai
pip install -e .
```

### With GUI Support
```bash
pip install putergenai[gui]
# or manually: pip install customtkinter pillow keyring cryptography flask
```

### System Requirements
- Python 3.11 or higher (tested on 3.11, 3.12)
- Linux, macOS, or Windows
- Internet connection for API calls

**Dependencies:**
- **Core**: `requests>=2.32.5`
- **GUI** (optional): `customtkinter>=5.2.2`, `pillow>=12.0.0`, `keyring>=25.7.0`, `cryptography>=46.0.3`
- **Web Demo** (optional): `flask>=3.1.2`

**Pro Tip:** Use a virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install putergenai
```

## Quick Start

### Basic Chat
```python
from putergenai import PuterClient

# Initialize client
client = PuterClient()
client.login("your_username", "your_password")

# Simple chat
response = client.ai_chat(
    messages=[{"role": "user", "content": "What is AI?"}],
    options={"model": "gpt-5"}
)
print("Response:", response["response"]["result"]["message"]["content"])
print("Model used:", response["used_model"])
```

### Streaming Chat
```python
# Real-time streaming response
gen = client.ai_chat(
    messages=[{"role": "user", "content": "Write a poem"}],
    options={"model": "claude-sonnet-4", "stream": True}
)
print("Poem: ", end='', flush=True)
for content, model in gen:
    print(content, end='', flush=True)
print("\nStreamed from:", model)
```

### File Operations
```python
# Write a file
client.fs_write("greetings.txt", "Hello, Puter!")

# Read a file
content = client.fs_read("greetings.txt").decode('utf-8')
print("Content:", content)

# Delete a file
client.fs_delete("greetings.txt")
```

### Image Generation & OCR
```python
# Generate an image
image_url = client.ai_txt2img("A sunset over mountains", model="pollinations-image")
print("Image URL:", image_url)

# Image to text (OCR)
text = client.ai_img2txt("https://example.com/image.jpg", test_mode=True)
print("Extracted text:", text)

# Text to speech
audio = client.ai_txt2speech("Hello world!")
with open("greeting.mp3", "wb") as f:
    f.write(audio)
```

## API Reference

### Client Initialization
```python
client = PuterClient(token: Optional[str] = None)
```
- **token**: Pre-existing authentication token (optional, skips login)

### Authentication
```python
token = client.login(username: str, password: str) -> str
```
- Authenticates with Puter.js
- Returns auth token
- Raises `ValueError` on failure

### AI Chat
```python
client.ai_chat(
    messages: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    test_mode: bool = False,
    image_url: Optional[Union[str, List[str]]] = None,
    prompt: Optional[str] = None,
    strict_model: bool = False
) -> Union[Dict[str, Any], Generator[Tuple[str, str], None, None]]
```

**Parameters:**
- `messages`: Chat message history (e.g., `[{"role": "user", "content": "Hi"}]`)
- `options`: Dict with `model` (str), `stream` (bool), `temperature` (float 0-2), `max_tokens` (int)
- `test_mode`: Use test API (no credits consumed)
- `image_url`: URL(s) for vision models
- `prompt`: Simple text prompt (alternative to messages)
- `strict_model`: Raise error on model fallback

**Returns:**
- **Streaming**: Generator yielding `(content, used_model)` tuples
- **Non-streaming**: `{"response": dict, "used_model": str}`

### File System
```python
client.fs_write(path: str, content: Union[str, bytes, Any]) -> Dict[str, Any]
client.fs_read(path: str) -> bytes
client.fs_delete(path: str) -> None
```

### AI Utilities
```python
client.ai_img2txt(image: Union[str, Any], test_mode: bool = False) -> str
client.ai_txt2img(prompt: str, model: str = "pollinations-image", test_mode: bool = False) -> str
client.ai_txt2speech(text: str, options: Optional[Dict[str, Any]] = None) -> bytes
```

## Use Cases

### 1. Interactive ChatBot
```python
messages = [{"role": "system", "content": "You are helpful."}]
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit": break
    messages.append({"role": "user", "content": user_input})
    
    gen = client.ai_chat(messages, options={"model": "gpt-5", "stream": True})
    print("Assistant: ", end='')
    for content, _ in gen:
        print(content, end='', flush=True)
    print()
```

### 2. Content Generation Pipeline
```python
# Generate blog post
response = client.ai_chat(
    prompt="Write a 500-word blog post about AI ethics",
    options={"model": "claude-3-5-sonnet"}
)
blog_content = response["response"]["result"]["message"]["content"]

# Convert to speech
audio = client.ai_txt2speech(blog_content)
with open("blog.mp3", "wb") as f:
    f.write(audio)

# Save to cloud
client.fs_write("blog_post.txt", blog_content)
```

### 3. Document OCR & Analysis
```python
# Extract text from image
extracted_text = client.ai_img2txt("https://example.com/document.jpg")

# Analyze with AI
analysis = client.ai_chat(
    prompt=f"Analyze this document:\n\n{extracted_text}",
    options={"model": "gpt-5"}
)
print("Analysis:", analysis["response"]["result"]["message"]["content"])
```

## Error Handling & Best Practices

### Common Errors
```python
try:
    client.login("user", "pass")
except ValueError as e:
    print(f"Login failed: {e}")  # Invalid credentials
except requests.RequestException as e:
    print(f"Network error: {e}")  # Connection issues
```

### Best Practices
```python
import os
import logging

# 1. Use environment variables for secrets
username = os.getenv("PUTER_USERNAME")
password = os.getenv("PUTER_PASSWORD")

# 2. Enable debug logging (only in dev)
logging.basicConfig(level=logging.DEBUG)

# 3. Use test mode in development
response = client.ai_chat(
    messages=[...],
    options={"model": "gpt-5"},
    test_mode=True  # No credits consumed
)

# 4. Handle model fallback
response = client.ai_chat(
    messages=[...],
    options={"model": "gpt-5"},
    strict_model=False  # Allow fallback
)
if response["used_model"] != "gpt-5":
    print(f"Fell back to: {response['used_model']}")

# 5. Wrap streaming in try-except
try:
    gen = client.ai_chat(messages=[...], options={"stream": True})
    for content, model in gen:
        print(content, end='', flush=True)
except Exception as e:
    print(f"Stream error: {e}")

# 6. Pin version in production
# requirements.txt: putergenai==1.5.1
```

## Supported Models

PutergenAI supports 50+ models across multiple providers:

**OpenAI**: GPT-5, GPT-5.1, GPT-4o, o1, o3, etc.  
**Anthropic**: Claude Opus, Sonnet, Haiku (latest versions)  
**Meta**: Llama 3.3, Llama Guard  
**Google**: Gemini 2.5, Gemma 3  
**Other**: DeepSeek, Mistral, Grok, Qwen, and more  

See `PuterClient.model_to_driver` for the full list.

## Security

✅ **Input Sanitization** – All user inputs validated and sanitized  
✅ **Encrypted Storage** – API keys stored encrypted with Fernet  
✅ **No Secret Logging** – Sensitive data never logged  
✅ **Secure Defaults** – HTTPS enforced, TLS verified  
✅ **Dependency Monitoring** – All dependencies pinned to secure versions  

**For security issues**: See [SECURITY.md](SECURITY.md)

## GUI Application

PutergenAI includes a modern GUI (`examples/example-ui.py`) with:
- Asynchronous login (no UI freezing)
- Multi-model support
- Image generation (Hugging Face, Replicate, DeepAI, OpenAI)
- Encrypted API key storage
- Streaming chat responses
- Error recovery & fallbacks

**Usage:**
```bash
python examples/example-ui.py
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/xyz`
3. Add tests and update docs
4. Submit a pull request

**Style Guide**: PEP 8 (enforced with Black, flake8)  
**Tests**: `python -m pytest tests/`  

## Changelog

### v1.5.1 (November 26, 2025) – LATEST
- ✅ Fixed type hint imports (`Tuple` added to typing imports)
- ✅ Removed unused imports
- ✅ Updated all dependencies to latest secure versions
- ✅ Enhanced Python 3.11+ compatibility
- ✅ Improved error handling and logging
- ✅ Updated security policy and documentation

### v1.5.0 (November 26, 2025)
- 🚀 Major release: Consolidated from 0.1.x to 1.5.x
- ✅ Production-ready designation
- ✅ Expanded model support (50+ models)
- ✅ New: Vision & multimedia support
- ✅ Enhanced security and error resilience

### v0.1.5 (Earlier)
- Initial releases with core SDK functionality

## Troubleshooting

**Login Issues:**
- Verify Puter.js account is active
- Check credentials are correct
- Ensure network connectivity
- Try `test_mode=True` for debugging

**Model Fallback:**
- Check if requested model is supported
- Use `strict_model=False` to allow fallback (default)
- Check `response["used_model"]` for actual model used

**File Operations:**
- Verify cloud paths are valid
- Ensure sufficient storage quota
- Check file permissions

**API Rate Limits:**
- Implement exponential backoff on retries
- Use `test_mode=True` to avoid consuming credits
- Monitor usage via Puter.js dashboard

**Memory/Performance:**
- Use streaming for large responses
- Implement pagination for file operations
- Monitor resource usage for long-running processes

## License

MIT License – See [LICENSE](LICENSE) for details.

## Support & Community

- **GitHub Issues**: [Report bugs](https://github.com/Nerve11/putergenai/issues)
- **Discussions**: [Ask questions](https://github.com/Nerve11/putergenai/discussions)
- **Security**: [Report vulnerabilities](SECURITY.md)

## Maintainers

- **[Nerve11](https://github.com/Nerve11)** – Core SDK
- **[BubblesTheDev](https://github.com/KernFerm)** – GUI & examples

**Last Updated**: November 26, 2025  
**Version**: 1.5.1  
**Status**: ✅ Production Ready

---

⭐ If this SDK helps you, please star the repo! Questions? Open an issue or discussion.

## API Syntax and Reference

The SDK centers around the `PuterClient` class. All methods are synchronous for simplicity; for async, wrap in `asyncio` or use threading.

### Initialization
```python
client = PuterClient(token="optional_pre_existing_token")
```
- If `token` is provided, skips login. Otherwise, call `login()`.

### Authentication
```python
client.login(username: str, password: str) -> str
```
- Returns the auth token.
- Raises `ValueError` on failure (e.g., invalid credentials).

### AI Chat
```python
client.ai_chat(
    messages: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    test_mode: bool = False,
    image_url: Optional[Union[str, List[str]]] = None,
    prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
    strict_model: bool = False
) -> Union[Dict[str, Any], Generator[tuple[str, str], None, None]]
```
- **messages**: List of chat messages (e.g., `[{"role": "user", "content": "Hi"}]`).
- **options**: Dict with `model` (str, e.g., "gpt-5"), `stream` (bool), `temperature` (float 0-2).
- **test_mode**: Use test API (no credits consumed).
- **image_url**: For vision models (e.g., GPT-4o).
- **prompt**: Alternative to messages for simple queries.
- **strict_model**: If True, raises error on model fallback.
- **Returns**:
  - Non-stream: `{"response": dict, "used_model": str}`.
  - Stream: Generator yielding `(content_chunk, used_model)`.

**Syntax Notes**:
- Models are passed explicitly in payload for reliability.
- Handles server fallbacks (e.g., GPT-5 → GPT-4.1-nano) with warnings or errors.
- Retries up to 3 times on availability issues, auto-enabling `test_mode` if needed.

### File System Operations
```python
client.fs_write(path: str, content: Union[str, bytes, Any]) -> Dict[str, Any]
client.fs_read(path: str) -> bytes
client.fs_delete(path: str) -> None
```
- `path`: Cloud path (e.g., "test.txt").
- `content`: String, bytes, or file-like object.
- Raises `ValueError` or `requests.RequestException` on failure.

### Other AI Utilities
```python
client.ai_img2txt(image: Union[str, Any], test_mode: bool = False) -> str
client.ai_txt2img(prompt: str, test_mode: bool = False) -> str
client.ai_txt2speech(text: str, options: Optional[Dict[str, Any]] = None) -> bytes
```
- `ai_img2txt`: OCR from URL or file.
- `ai_txt2img`: Generates image URL from prompt.
- `ai_txt2speech`: Returns MP3 bytes.

**Error Handling**: All methods raise exceptions on failure. Use try-except for resilience.

## Use Cases


### 1. **Interactive AI Chat Bot** (e.g., Customer Support)
Use streaming for real-time responses. Handle model fallbacks for reliability.
```python
messages = [{"role": "system", "content": "You are a helpful assistant."}]
while True:
    user_input = input("You: ")
    if user_input == "exit": break
    messages.append({"role": "user", "content": user_input})
    gen = client.ai_chat(messages, options={"model": "gpt-5", "stream": True}, strict_model=False)
    print("Assistant: ", end='')
    for content, _ in gen:
        print(content, end='', flush=True)
    print()
```

**Why it works**: Streaming reduces latency; `strict_model=False` ensures uptime if GPT-5 is unavailable.

### 2. **File Backup Tool** (Cloud Storage Integration)
Sync local files to Puter.js FS.
```python
def backup_file(local_path: str, cloud_path: str):
    with open(local_path, 'rb') as f:
        client.fs_write(cloud_path, f)
    print(f"Backed up {local_path} to {cloud_path}")
```

**Pro Tip**: Implement hashing (e.g., SHA-256) to avoid unnecessary uploads. Use threading for large files.

### 3. **AI Content Generation Pipeline** (e.g., Blog Post Generator)
Generate text, convert to speech, and store.
```python
prompt = "Write a blog post about AI ethics."
response = client.ai_chat(prompt=prompt, options={"model": "claude-3-5-sonnet"})
content = response["response"]["result"]["message"]["content"]
client.fs_write("blog_post.txt", content)
audio = client.ai_txt2speech(content)
with open("blog_post.mp3", "wb") as f:
    f.write(audio)
```

**Best Practice**: Batch requests for high-volume use; monitor costs via Puter.js dashboard.

### 4. **Vision-Based Analysis** (e.g., Image Description)
```python
description = client.ai_img2txt("https://example.com/image.jpg", test_mode=True)
print("Image description:", description)
```

**Limitation Note**: Vision models (e.g., GPT-4o) may require specific drivers; test with `test_mode=True` first.


## Error Handling and Best Practices

- **Common Errors**:
    - `ValueError`: Invalid credentials or model mismatch (with `strict_model=True`).
    - `requests.RequestException`: Network issues; implement retries with exponential backoff.
    - Model fallback: Server may use a different model (e.g., GPT-5 → GPT-4.1-nano); check `used_model` in response.

- **Best Practices**:
    - **Security**: Never hardcode credentials; use environment variables (e.g., `os.getenv("PUTER_USERNAME")`). All user input is sanitized.
    - **Performance**: For streaming, use in async contexts (e.g., `asyncio`) to avoid blocking.
    - **Costs**: Always set `test_mode=True` in dev; monitor usage via Puter.js API.
    - **Testing**: Write unit tests for your integration (e.g., mock responses with `responses` lib).
    - **Versioning**: Pin to a specific SDK version in `requirements.txt` (e.g., `putergenai==0.1.01`).
    - **Scalability**: For multi-user apps, pool clients or use session tokens.

If you encounter issues, check logs (enable DEBUG via `logging.basicConfig(level=logging.DEBUG)`) and verify your Puter.js account status. Contributions welcome—see below.

## Contributing

Fork the repo, create a branch (`git checkout -b feature/xyz`), commit changes, and open a PR. Follow PEP 8 for style. Include tests (use `unittest`) and update docs if needed.

Run tests:
```bash
python -m unittest discover tests
```

# PutergenAI GUI Application

This repository includes a CustomTkinter-based GUI for PutergenAI, allowing you to chat with AI models, generate images, and manage your Puter.js account visually. The GUI lives at `examples/example-ui.py` and now implements modern secret management and web security best practices.

## Features
- **Secure Login:** Asynchronous login with Puter credentials to prevent UI freezing
- **Multi-Model Support:** Select from multiple AI models for chat and image generation
- **Image Generation APIs:** Support for 4 free APIs: Hugging Face, Replicate, DeepAI, OpenAI
- **Smart API Key Management:** Environment variables → system keychain (keyring) → encrypted-file fallback with automatic prompts
- **Popup Notifications:** Helpful reminders when selecting APIs that require keys
- **Dynamic UI:** API key entry section appears when needed with disable/enable dropdowns
- **Asynchronous Operations:** Non-blocking chat and image generation to maintain UI responsiveness
- **Enhanced Error Handling:** Robust error handling with user-friendly feedback
- **Security Features:** Encrypted API key storage and secure session management
- **Sign Out Functionality:** Clean logout and return to login screen
- **Consistent UX:** Fixed window sizing (800x600) for optimal user experience

## How to Use
1. **Run the app:**
   ```bash
    # Minimal GUI example
    python examples/example.py

    # Enhanced GUI with secure secret management
    python examples/example-ui.py
   ```
2. **Login:** Enter your Puter username and password. The login process runs asynchronously to prevent UI freezing.
3. **Select Model:** Choose an AI model for chat or image tasks from the dropdown menu.
4. **Select Image Generation API:** Pick one of the free APIs. A popup will remind you to add your API key if required.
5. **Enter API Key:** When prompted, enter your API key in the provided field and click "Save Key". Keys are encrypted and stored securely.
6. **Chat or Generate Images:** Use the chat box and buttons to interact with the AI or generate images. All operations are asynchronous.
7. **Monitor Progress:** Watch the status updates and progress indicators during operations.
8. **Handle Errors:** The application provides clear error messages and fallback options for failed operations.
9. **Sign Out:** Click the Sign Out button to securely log out and return to the login screen.

## Requirements
- Python 3.8+
- `putergenai` (see SDK instructions above)
- `customtkinter`, `Pillow`, `requests`, `cryptography`, `keyring` (for secure system keychain storage)

Install dependencies:
```bash
    pip install customtkinter pillow requests putergenai cryptography keyring
```

### Optional: Configure environment variables for API keys and secrets

- Windows PowerShell:
    ```powershell
    $env:HUGGINGFACE_API_TOKEN = "hf_..."
    $env:REPLICATE_API_TOKEN   = "r8_..."
    $env:DEEPAI_API_KEY        = "quickstart-..."
    $env:OPENAI_API_KEY        = "sk-..."

    # Optional: Flask demo secret and Fernet key for encrypted file fallback
    $env:PUTERGENAI_FLASK_SECRET_KEY = "long-random-hex"
    $env:PUTERGENAI_FERNET_KEY       = "base64url-fernet-key"
    ```

- Linux/macOS (bash/zsh):
    ```bash
    export HUGGINGFACE_API_TOKEN="hf_..."
    export REPLICATE_API_TOKEN="r8_..."
    export DEEPAI_API_KEY="quickstart-..."
    export OPENAI_API_KEY="sk-..."

    export PUTERGENAI_FLASK_SECRET_KEY="long-random-hex"
    export PUTERGENAI_FERNET_KEY="base64url-fernet-key"
    ```

## Notes
- **API Keys:** Required for Hugging Face, Replicate, and OpenAI image generation. DeepAI may work with a demo key.
- **Security:** API keys are resolved in this order: environment variables → system keychain via `keyring` → encrypted-file fallback (`api_keys.cfg` with Fernet). Never store plain-text credentials.
- **Performance:** All network operations (login, chat, image generation) run asynchronously to prevent UI freezing.
- **Error Handling:** The application includes comprehensive error handling with user-friendly messages and fallback mechanisms.
- **Window Management:** Window size is fixed (800x600) and cannot be resized for consistent UX.
- **Threading:** Login and API operations use threading to maintain UI responsiveness.
- **Input Validation:** All user input is sanitized for security.
- **Session Management:** Secure session handling with proper cleanup on logout.

For SDK usage and advanced features, see below.

## GUI Example Usage

Below is a sample workflow using the included `example.py` GUI application:

```python
from putergenai import PuterClient
import customtkinter as ctk
import tkinter as tk
import tkinter.messagebox as mbox

# Launch the GUI
if __name__ == "__main__":
    PuterApp().mainloop()
```

### Main Features in the GUI
- **Secure Login:** Asynchronous authentication with Puter credentials using threading to prevent UI blocking.
- **Model Selection:** Choose from available AI models (chat and image) with dynamic dropdown management.
- **Image Generation API Selection:** Pick from Hugging Face, Replicate, DeepAI, or OpenAI with smart API key prompts.
- **Encrypted API Key Management:** Secure storage and retrieval of API keys using Fernet encryption.
- **Asynchronous Operations:** Non-blocking chat and image generation with progress indicators.
- **Error Recovery:** Comprehensive error handling with fallback mechanisms and user-friendly messages.
- **Session Management:** Secure login/logout functionality with proper session cleanup.
- **Responsive UI:** Threading ensures the interface remains responsive during all operations.
- **Smart Dropdown Control:** Dropdowns automatically disable/enable based on operation state.
- **Window Management:** Fixed window size (800x600) with optimized layout for consistent user experience.

### Example GUI Flow
1. **Launch:** Run the app with `python example.py`
2. **Authentication:** Login with your Puter username and password (asynchronous process)
3. **Configuration:** Select a model and an image generation API from the dropdown menus
4. **API Setup:** Enter your API key if prompted and save it (encrypted storage)
5. **Interaction:** Type a message or image prompt and use the buttons to chat or generate images
6. **Monitor:** Watch progress indicators and status updates during operations
7. **Error Handling:** Receive clear feedback if operations fail with suggested solutions
8. **Session End:** Sign out securely when finished, which clears all session data

## Advanced Usage

### Asynchronous Operations
The GUI implements threading for all network operations to prevent freezing:
- Login authentication runs in a separate thread
- Image generation uses asynchronous processing with progress updates
- Chat operations are non-blocking with real-time response streaming
- Error handling provides immediate feedback without blocking the UI

### Security Implementation
- **Secret resolution order:** ENV → system keychain (`keyring`) → encrypted-file fallback
- **Encryption:** Fallback secrets are encrypted using Fernet symmetric encryption
- **File Security:** Sensitive files have restricted permissions (user read/write only)
- **No Logging:** Sensitive data is never logged or printed to console
- **Session Tokens:** Secure session management; demo web route uses session tokens instead of storing passwords
- **Input Sanitization:** All user input is validated and sanitized before processing

## Security Features

- **Key Management Hierarchy:** Environment variables → system keychain (`keyring`) → encrypted-file fallback (`api_keys.cfg`).
- **Encrypted Fallback Storage:** When keyring is unavailable, secrets are stored encrypted with Fernet; key material is in `api_keys.key` or `PUTERGENAI_FERNET_KEY`.
- **File Permissions:** Sensitive files default to user-only permissions on Windows.
- **No Sensitive Logging:** Secrets are never logged.
- **Asynchronous Authentication:** Login uses threading to avoid UI blocking.
- **Session Management:** No plain-text credential storage; web demo uses session tokens.
- **Input Validation:** Sanitized inputs, safe filenames, and constrained image options.
- **Secure Cookies and Headers:** Flask demo sets Secure/HttpOnly/SameSite cookies and adds basic security headers.

### Secure Cookie Example
The Flask demo now avoids storing passwords in cookies and instead sets a short-lived session token cookie, plus security headers:

```python
from flask import Flask, make_response, request
from cryptography.fernet import Fernet
import os, secrets

app = Flask("Secure Example")
app.secret_key = os.environ.get("PUTERGENAI_FLASK_SECRET_KEY", secrets.token_hex(32))
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Strict',
)
fernet = Fernet(Fernet.generate_key())

@app.route('/')
def index():
    password = request.args.get("password")
    if password:
        session_token = secrets.token_urlsafe(32)
        resp = make_response("Authentication token created (password not stored in cookie)")
        resp.set_cookie("session_token", session_token, secure=True, httponly=True, samesite='Strict', max_age=3600)
        return resp
    return "No password provided"

@app.after_request
def set_secure_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp
```

**Security Best Practices:**
- Never store credentials in cookies; use opaque session tokens
- Always set Secure/HttpOnly/SameSite on sensitive cookies
- Add basic security headers; prefer a CSP and further hardening in production
- Use environment variables for secrets; rotate keys/tokens regularly
- Implement rate limiting, logging, and session expiry

## Troubleshooting & Performance

### Common Issues and Solutions

**Login Freezing:**
- **Solution:** The application now uses asynchronous login with threading to prevent UI freezing
- **Implementation:** Login operations run in background threads while maintaining UI responsiveness

**Image Generation Timeouts:**
- **Solution:** Asynchronous image generation with multiple API fallbacks
- **Features:** Progress indicators, timeout handling, and automatic retry mechanisms

**API Key Errors:**
- **Solution:** Enhanced error handling with clear user feedback
- **Features:** Encrypted storage, validation, and helpful error messages

**Memory Issues:**
- **Solution:** Proper resource cleanup and image optimization
- **Implementation:** PIL image handling with memory-efficient processing

### Performance Optimizations

**Threading Implementation:**
- All network operations use dedicated threads to prevent UI blocking
- Main UI thread remains responsive during all operations
- Proper thread synchronization to prevent race conditions

**Error Recovery:**
- Multiple API fallbacks for image generation (Hugging Face → Replicate → DeepAI → OpenAI)
- Graceful degradation when services are unavailable
- User-friendly error messages with suggested actions

**Resource Management:**
- Efficient memory usage for image processing
- Proper cleanup of network connections
- Optimized UI updates to prevent rendering lag

**Security Performance:**
- Fast encryption/decryption of API keys using Fernet
- Minimal impact file permission checks
- Secure session management without performance overhead


## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Built on top of Puter.js—kudos to the team for an innovative API. Inspired by real-world needs for privacy-focused AI tools.

**Maintainers**:
- [Nerve11](https://github.com/Nerve11)
- [BubblesTheDev](https://github.com/KernFerm)
- **Last Updated**: November 26, 2025  
- **Version**: `1.5.1`

- If this SDK saves you time, star the repo! Questions? Open an issue.
