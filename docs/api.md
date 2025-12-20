# putergenai API reference

The `putergenai` package provides an asynchronous high‑level client for the Puter.com API.
The main entry point is the `PuterClient` class.

```python
from putergenai import PuterClient
```

## PuterClient

### Constructor

```python
PuterClient(
    token: str | None = None,
    ignore_ssl: bool = False,
    auto_update_models: bool = False,
)
```

- `token`: Bearer token for the Puter API. If not provided, you must call `login`.
- `ignore_ssl`: When `True`, disables SSL verification (not recommended in production).
- `auto_update_models`: When `True`, updates the internal model mapping on context‑manager entry.

Typical usage:

```python
client = PuterClient(token="your_token_here")
```

or with an async context manager:

```python
async with PuterClient(token="your_token_here") as client:
    ...
```

### Authentication

```python
await client.login(username: str, password: str) -> str
```

Logs in to Puter using username and password and stores the returned token inside the client.
Returns the token string or raises `ValueError` on failure.

### Model discovery

```python
await client.get_available_models(force_refresh: bool = False) -> dict
```

Returns a dictionary with the key `"models"`, which is a list of available models.
The result is cached for 1 hour; set `force_refresh=True` to bypass cache.

```python
await client.update_model_mappings() -> None
```

Refreshes the internal `model_to_driver` mapping according to the current model list.
This is useful when Puter adds or removes models.

```python
client.get_model_list(models_data: dict | None = None) -> list[str]
```

Extracts a plain list of model IDs from `get_available_models` data.
If `models_data` is omitted, the last cached value is used.

```python
await client.is_model_available(model_name: str) -> bool
```

Returns `True` if the given model name is present in the current model list.

### Filesystem operations

All filesystem methods work with paths inside the Puter virtual filesystem.

```python
await client.fs_write(path: str, content: str | bytes | IO[Any]) -> dict
```

Writes the given content to a file at `path`.
Returns a JSON dictionary from the API or raises an error on failure.

```python
await client.fs_read(path: str) -> bytes
```

Reads the file at `path` and returns its raw bytes.

```python
await client.fs_delete(path: str) -> None
```

Deletes a file or directory at `path`.

#### Filesystem example

```python
import asyncio
from putergenai import PuterClient

async def main():
    async with PuterClient(token="your_token_here") as client:
        await client.fs_write("/hello.txt", "Hello from putergenai!\n")
        data = await client.fs_read("/hello.txt")
        print(data.decode("utf-8"))
        await client.fs_delete("/hello.txt")

asyncio.run(main())
```

### Chat completions

```python
await client.ai_chat(
    prompt: str | list[dict] | None = None,
    options: dict | None = None,
    test_mode: bool = False,
    image_url: str | list[str] | None = None,
    messages: list[dict] | None = None,
    strict_model: bool = False,
) -> dict | AsyncGenerator[tuple[str, str], None]
```

High‑level chat API on top of the Puter driver endpoint.

Key parameters:

- `prompt`: Simple string prompt. If `messages` is provided, it can be omitted.
- `options`:
  - `model`: Target model name (for example, `"gpt-5-nano"`).
  - `stream`: When `True`, returns an async generator instead of a single response.
  - `temperature`: Sampling temperature. Some models are forced to `1` by design.
  - `max_tokens`: Maximum number of tokens to generate.
  - `tools`: Optional tool calling definition (see Puter/OpenAI schema).
- `test_mode`: Enables Puter test routing when supported.
- `image_url`: Single URL or list of URLs for multimodal models.
- `messages`: Low‑level messages list in OpenAI/Claude style.
- `strict_model`: When `True`, raises an error if the backend substitutes another model.

Return value:

- Non‑streaming (`options["stream"]` is `False`): a dictionary with keys
  - `"response"`: raw JSON response from Puter.
  - `"used_model"`: the actual model ID used by the backend.
- Streaming (`options["stream"]` is `True`): an async generator yielding pairs
  `(chunk_text, used_model)` for each partial update.

#### Non‑streaming example

```python
import asyncio
from putergenai import PuterClient

async def main():
    async with PuterClient(token="your_token_here") as client:
        result = await client.ai_chat(
            prompt="Explain what Puter is in one sentence.",
            options={"model": "gpt-5-nano", "max_tokens": 128},
        )
        print("Used model:", result["used_model"])
        print("Raw response:", result["response"])

asyncio.run(main())
```

#### Streaming example

```python
import asyncio
from putergenai import PuterClient

async def main():
    async with PuterClient(token="your_token_here") as client:
        stream = await client.ai_chat(
            prompt="Stream a short introduction to Puter.",
            options={"model": "gpt-5-nano", "stream": True},
        )
        async for chunk, used_model in stream:
            print(chunk, end="")
        print("\n(model:", used_model, ")")

asyncio.run(main())
```

### Vision and media helpers

#### `ai_img2txt`

```python
await client.ai_img2txt(image: str | IO[Any], test_mode: bool = False) -> str
```

Runs OCR on a remote image URL or a file‑like object and returns extracted text.

Example:

```python
text = await client.ai_img2txt("https://example.com/image.png")
print(text)
```

#### `ai_txt2img`

```python
await client.ai_txt2img(
    prompt: str,
    model: str = "pollinations-image",
    test_mode: bool = False,
) -> str
```

Generates an image from text using the specified image driver.
Returns an image URL or other identifier provided by the backend.

Example:

```python
url = await client.ai_txt2img("A small blue robot on a desk")
print("Image URL:", url)
```

#### `ai_txt2speech`

```python
await client.ai_txt2speech(text: str, options: dict | None = None) -> bytes
```

Converts text to speech and returns audio bytes (MP3).
The `options` dictionary may contain the key `"testMode"`.

Example:

```python
import asyncio
from putergenai import PuterClient

async def main():
    async with PuterClient(token="your_token_here") as client:
        audio = await client.ai_txt2speech("Hello from putergenai!")
        with open("output.mp3", "wb") as f:
            f.write(audio)

asyncio.run(main())
```
