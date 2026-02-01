# Installation

## Requirements
- Python 3.9 or higher
- A Puter.com account and API access token
- Recommended: virtual environment (`venv`, `virtualenv`, or similar)

## Installing from PyPI
Install the latest released version from PyPI:

```bash
pip install --upgrade pip
pip install putergenai
```

Verify that the package is available:

```bash
python -m pip show putergenai
```

## CLI Tool

The package includes a command-line interface (CLI) for quick interaction with Puter.

```bash
# Login
puter login

# Start a chat
puter chat --model gpt-4o

# List available models
puter models

# Manage Key-Value store
puter kv set mykey "value"
puter kv get mykey
```

## Installing from source
If you want to track the latest changes from the repository:

```bash
git clone https://github.com/Nerve11/putergenai.git
cd putergenai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

## Configuration
The client expects a Puter access token.

### Via environment variable

1. Copy example file:

```bash
cp .env.example .env
```

2. Edit `.env` and set:

```bash
PUTER_TOKEN="your_token_here"
```

3. Load it in your application (for example, using `python-dotenv`) or pass the value directly to `PuterClient`.

### Via explicit argument
You can also pass the token directly when constructing the client:

```python
from putergenai import PuterClient

client = PuterClient(token="your_token_here")
```

## Quick sanity check
This minimal script checks that authentication and a simple chat call work:

```python
import asyncio
from putergenai import PuterClient

async def main():
    async with PuterClient(token="your_token_here") as client:
        result = await client.ai_chat(prompt="ping", options={"model": "gpt-5-nano"})
        print(result["used_model"], "OK")

asyncio.run(main())
```
