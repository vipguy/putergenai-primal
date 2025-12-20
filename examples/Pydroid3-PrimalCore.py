# -*- coding: utf-8 -*-
"""
Putergenai + Pollinations All-in-One CLI (Pydroid3 ready)
- Logs into Puter once
- Chat (Puter models)
- Chat (Pollinations)  <-- separate code path.
- Generate Image (Puter)
- Generate Image (Pollinations)  <-- separate code path
- OCR (Puter)
- TTS (Puter)

Notes:
- PrimalCore was here.
"""

import html
import json
import logging
import os
import re
import sys
import time
from getpass import getpass
from typing import Any, Dict, Generator, List, Optional, Union
from urllib.parse import quote_plus, urlparse

import requests

# ---------- Logging ----------
LOG_FILE = "putergenai.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("puter_pollinations_cli")


# ---------- Helpers ----------
def sanitize_string(s, allow_empty=False, allow_path=False):
    if not isinstance(s, str):
        raise ValueError("Input must be a string.")
    s = s.strip()
    if not allow_empty and not s:
        raise ValueError("Input cannot be empty.")
    if allow_path:
        pattern = r"^[\w\-\.\/]+$"
        if not re.match(pattern, s):
            raise ValueError("Input contains invalid characters for path.")
    return s


def sanitize_float(val, min_value=0.0, max_value=2.0, default=0.7):
    try:
        f = float(val)
    except (ValueError, TypeError):
        f = default
    if f < min_value or f > max_value:
        raise ValueError(f"Value must be between {min_value} and {max_value}.")
    return f


def sanitize_int(val, min_value: int, max_value: int, default: int):
    try:
        n = int(val)
    except (ValueError, TypeError):
        n = default
    if n < min_value or n > max_value:
        raise ValueError(f"Value must be between {min_value} and {max_value}.")
    return n


def sanitize_url(url):
    if not isinstance(url, str):
        raise ValueError("Invalid URL: not a string.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Invalid URL.")
    return url


def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def get_password(prompt: str) -> str:
    try:
        return getpass(prompt)
    except Exception:
        # Some Android terminals might not support getpass masking
        return input(prompt)


def strip_html_to_text(s: str) -> str:
    """Convert simple HTML to plain text for Pollinations replies."""
    if not s:
        return ""
    # Normalize breaks/paragraphs to newlines
    s = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", s)
    s = re.sub(r"(?i)</\s*p\s*>", "\n\n", s)
    s = re.sub(r"(?i)<\s*p\s*>", "", s)
    # Remove other tags
    s = re.sub(r"<[^>]+>", "", s)
    # Unescape entities
    s = html.unescape(s)
    # Collapse excessive blank lines
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


# ---------- Puter Client ----------
class PuterClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.api_base = "https://api.puter.com"
        self.login_url = "https://puter.com/login"
        self.headers = {
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Origin": "https://puter.com",
            "Referer": "https://puter.com/",
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"
            ),
        }
        # Puter-provided/composite mapping (no pollinations-text here; handled separately)
        self.model_to_driver = {
            "gpt-4o-mini": "openai-completion",
            "gpt-4o": "openai-completion",
            "gpt-4.1": "openai-completion",
            "gpt-4.1-mini": "openai-completion",
            "gpt-4.1-nano": "openai-completion",
            "o1": "openai-completion",
            "o1-mini": "openai-completion",
            "o1-pro": "openai-completion",
            "o3": "openai-completion",
            "o3-mini": "openai-completion",
            "o4-mini": "openai-completion",
            "gpt-5-2025-08-07": "openai-completion",
            "gpt-5-mini-2025-08-07": "openai-completion",
            "gpt-5-nano-2025-08-07": "openai-completion",
            "gpt-5-chat-latest": "openai-completion",
            "claude-3-7-sonnet-latest": "claude",
            "claude-3-5-sonnet-latest": "claude",
            "claude-sonnet-4": "claude",
            "claude-opus-4-latest": "claude",
            "claude-opus-4-1-20250805": "claude",
            "deepseek-chat": "deepseek",
            "deepseek-reasoner": "deepseek",
        }
        self.force_temperature_1_models = {
            "gpt-5-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
            "o3",
            "o1",
        }
        self.fallback_models = [
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "claude-3-5-sonnet-latest",
            "deepseek-chat",
        ]
        self.max_retries = 3

    def login(self, username: str, password: str) -> str:
        username = sanitize_string(username)
        password = sanitize_string(password)
        payload = {"username": username, "password": password}
        try:
            r = requests.post(self.login_url, headers=self.headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if data.get("proceed") and data.get("token"):
                self.token = data["token"]
                logger.info("Login successful, token acquired")
                return self.token
            raise ValueError("Login failed. Please check your credentials.")
        except requests.RequestException as e:
            raise ValueError(f"Login error: {e}")

    def _auth_headers(self) -> Dict[str, str]:
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        return {
            **self.headers,
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def fs_write(self, path: str, content: Union[str, bytes, Any]) -> Dict[str, Any]:
        path = sanitize_string(path, allow_path=True)
        headers = self._auth_headers()
        headers.pop("Content-Type", None)
        if isinstance(content, str):
            content = content.encode("utf-8")
        elif hasattr(content, "read"):
            content = content.read()
        elif not isinstance(content, bytes):
            raise ValueError("Content must be str, bytes, or file-like.")
        try:
            r = requests.post(
                f"{self.api_base}/write",
                params={"path": path},
                data=content,
                headers=headers,
                timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise ValueError(f"fs_write error: {e}")

    def fs_read(self, path: str) -> bytes:
        path = sanitize_string(path, allow_path=True)
        headers = self._auth_headers()
        try:
            r = requests.get(
                f"{self.api_base}/read", params={"path": path}, headers=headers, timeout=60
            )
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            raise ValueError(f"fs_read error: {e}")

    def fs_delete(self, path: str) -> None:
        path = sanitize_string(path, allow_path=True)
        headers = self._auth_headers()
        try:
            r = requests.post(
                f"{self.api_base}/delete", params={"path": path}, headers=headers, timeout=30
            )
            r.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"fs_delete error: {e}")

    def ai_chat(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        options: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
        image_url: Optional[Union[str, List[str]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        retry_count: int = 0,
        strict_model: bool = False,
    ) -> Union[Dict[str, Any], Generator[tuple[str, str], None, None]]:
        if options is None:
            options = {}
        model = options.get("model", self.fallback_models[0])
        driver = self.model_to_driver.get(model, "openai-completion")
        stream = options.get("stream", False)
        temperature = options.get("temperature", 0.7)
        if model in self.force_temperature_1_models:
            temperature = 1
        max_tokens = options.get("max_tokens", 1000)

        if messages is None:
            messages = []
            if prompt:
                content = prompt
                if image_url:
                    if not isinstance(image_url, list):
                        image_url = [image_url]
                    parts = (
                        [{"type": "text", "text": content}] if isinstance(content, str) else content
                    )
                    for url in image_url:
                        parts.append({"type": "image_url", "image_url": {"url": sanitize_url(url)}})
                    content = parts
                messages.append({"role": "user", "content": content})

        args = {
            "messages": messages,
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": options.get("tools"),
        }
        payload = {
            "interface": "puter-chat-completion",
            "driver": driver,
            "method": "complete",
            "args": args,
            "stream": stream,
            "testMode": test_mode,
        }

        headers = self._auth_headers()

        def check_used_model(data: Dict[str, Any], requested_model: str, strict: bool) -> str:
            used_model = None
            if "result" in data and "usage" in data["result"] and data["result"]["usage"]:
                used_model = data["result"]["usage"][0].get("model", "unknown")
            elif "metadata" in data and "service_used" in data["metadata"]:
                used_model = data["metadata"].get("service_used", "unknown")
            if used_model and used_model != requested_model:
                msg = f"Requested model {requested_model}, but server used {used_model}"
                if strict:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
            return used_model or requested_model

        try:
            r = requests.post(
                f"{self.api_base}/drivers/call",
                json=payload,
                headers=headers,
                stream=stream,
                timeout=120,
            )
            r.raise_for_status()
            if stream:
                it = r.iter_lines()

                def process_line(line: bytes, requested_model: str, strict: bool):
                    if not line:
                        return None
                    try:
                        data = json.loads(line)
                        if not data.get("success", True):
                            error_data = data.get("error", {})
                            raise ValueError(
                                f"API error: {error_data.get('message','Unknown')} (code: {error_data.get('code')})"
                            )
                        used_model = check_used_model(data, requested_model, strict)
                        if "type" in data and data["type"] == "text" and "text" in data:
                            return data["text"], used_model
                        if "result" in data and "message" in data["result"]:
                            mc = data["result"]["message"].get("content", "")
                            if isinstance(mc, str) and mc:
                                return mc, used_model
                            if isinstance(mc, list) and mc and mc[0].get("text"):
                                return mc[0]["text"], used_model
                        if "choices" in data:
                            delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                return delta, used_model
                    except json.JSONDecodeError:
                        if line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                used_model = check_used_model(data, requested_model, strict)
                                t = data.get("text", "")
                                if t:
                                    return t, used_model
                            except json.JSONDecodeError:
                                return None
                        else:
                            decoded = line.decode("utf-8", errors="ignore")
                            if decoded.strip():
                                return decoded, requested_model
                    return None

                first = next(it, None)
                if first:
                    # early error inspection
                    try:
                        _ = process_line(first, model, strict_model)
                    except ValueError as e:
                        msg = str(e)
                        if (
                            any(
                                code in msg for code in ("no_implementation_available", "forbidden")
                            )
                            and retry_count < self.max_retries
                        ):
                            if strict_model:
                                raise ValueError("Strict model enforced; no fallback.")
                            if not test_mode:
                                return self.ai_chat(
                                    prompt=prompt,
                                    options=options,
                                    test_mode=True,
                                    image_url=image_url,
                                    messages=messages,
                                    retry_count=retry_count + 1,
                                    strict_model=strict_model,
                                )
                            # fallback to next model
                            idx = (
                                self.fallback_models.index(model)
                                if model in self.fallback_models
                                else -1
                            )
                            nxt = (
                                self.fallback_models[idx + 1]
                                if idx >= 0 and idx + 1 < len(self.fallback_models)
                                else None
                            )
                            if nxt:
                                options["model"] = nxt
                                time.sleep(1)
                                return self.ai_chat(
                                    prompt=prompt,
                                    options=options,
                                    test_mode=test_mode,
                                    image_url=image_url,
                                    messages=messages,
                                    retry_count=retry_count + 1,
                                    strict_model=strict_model,
                                )
                            raise

                def generator():
                    if first:
                        proc = process_line(first, model, strict_model)
                        if proc:
                            yield proc
                    for line in it:
                        proc = process_line(line, model, strict_model)
                        if proc:
                            yield proc

                return generator()
            else:
                data = r.json()
                if not data.get("success", True):
                    err = data.get("error", {})
                    code = err.get("code")
                    msg = err.get("message", "Unknown error")
                    if (
                        code in ("no_implementation_available", "forbidden")
                        and retry_count < self.max_retries
                    ):
                        if strict_model:
                            raise ValueError("Strict model enforced; no fallback.")
                        if not test_mode:
                            return self.ai_chat(
                                prompt=prompt,
                                options=options,
                                test_mode=True,
                                image_url=image_url,
                                messages=messages,
                                retry_count=retry_count + 1,
                                strict_model=strict_model,
                            )
                        # fallback chain
                        idx = (
                            self.fallback_models.index(model)
                            if model in self.fallback_models
                            else -1
                        )
                        nxt = (
                            self.fallback_models[idx + 1]
                            if idx >= 0 and idx + 1 < len(self.fallback_models)
                            else None
                        )
                        if nxt:
                            options["model"] = nxt
                            time.sleep(1)
                            return self.ai_chat(
                                prompt=prompt,
                                options=options,
                                test_mode=test_mode,
                                image_url=image_url,
                                messages=messages,
                                retry_count=retry_count + 1,
                                strict_model=strict_model,
                            )
                        raise ValueError(f"No implementation available for {model}: {msg}")
                used_model = check_used_model(data, model, strict_model)
                return {"response": data, "used_model": used_model}
        except requests.RequestException as e:
            raise ValueError(f"ai_chat request failed: {e}")

    def ai_img2txt(self, image: Union[str, Any], test_mode: bool = False) -> str:
        headers = self._auth_headers()
        try:
            if isinstance(image, str):
                payload = {"image_url": sanitize_url(image), "testMode": test_mode}
                r = requests.post(
                    f"{self.api_base}/ai/img2txt", json=payload, headers=headers, timeout=120
                )
            else:
                files = {"image": image}
                r = requests.post(
                    f"{self.api_base}/ai/img2txt",
                    files=files,
                    data={"testMode": test_mode},
                    headers=headers,
                    timeout=120,
                )
            r.raise_for_status()
            return r.json().get("text", "")
        except requests.RequestException as e:
            raise ValueError(f"ai_img2txt error: {e}")

    def ai_txt2img(
        self, prompt: str, model: str = "pollinations-image", test_mode: bool = False
    ) -> str:
        payload = {
            "interface": "puter-image-generation",
            "driver": model,
            "method": "generate",
            "args": {"prompt": prompt},
            "testMode": test_mode,
        }
        headers = self._auth_headers()
        try:
            r = requests.post(
                f"{self.api_base}/drivers/call", json=payload, headers=headers, timeout=120
            )
            r.raise_for_status()
            data = r.json()
            if "result" in data:
                if isinstance(data["result"], dict):
                    return (
                        data["result"].get("image_url")
                        or data["result"].get("url")
                        or data["result"].get("data", "")
                    )
            raise ValueError(f"Unexpected response format: {data}")
        except requests.RequestException as e:
            raise ValueError(f"ai_txt2img error: {e}")

    def ai_txt2speech(self, text: str, options: Optional[Dict[str, Any]] = None) -> bytes:
        if options is None:
            options = {}
        payload = {"text": text, "testMode": options.get("testMode", False)}
        headers = self._auth_headers()
        try:
            r = requests.post(
                f"{self.api_base}/ai/txt2speech", json=payload, headers=headers, timeout=120
            )
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            raise ValueError(f"ai_txt2speech error: {e}")


# ---------- Pollinations (direct) ----------
class PollinationsClient:
    """
    Minimal direct client.
    - Chat: tries POST JSON, falls back to simple GET /{prompt}
    - Image: GET image URL and save file
    """

    def __init__(self):
        self.text_base = "https://text.pollinations.ai"
        self.image_base = "https://image.pollinations.ai/prompt"

        self.headers = {
            "Accept": "*/*",
            "User-Agent": "Mozilla/5.0 (Linux; Android 10) Pydroid3/1.0",
        }

    def chat(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system: Optional[str] = None,
    ) -> str:
        prompt = prompt.strip()
        if not prompt:
            return ""
        # 1) Try JSON POST (works for many deployments)
        try:
            payload = {
                "messages": ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = requests.post(self.text_base, headers=self.headers, json=payload, timeout=60)
            r.raise_for_status()
            txt = r.text
            return strip_html_to_text(txt).strip()
        except Exception as e:
            logger.warning(f"Pollinations POST failed, falling back to GET: {e}")

        # 2) Fallback: GET /{prompt}
        try:
            url = f"{self.text_base}/{quote_plus(prompt)}"
            r = requests.get(url, headers=self.headers, timeout=60)
            r.raise_for_status()
            return strip_html_to_text(r.text).strip()
        except Exception as e:
            raise ValueError(f"Pollinations chat failed: {e}")

    def generate_image(
        self, prompt: str, size: str = "1024x1024", out_path: Optional[str] = None
    ) -> str:
        """
        Downloads the generated image locally and returns the local file path.
        """
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        url = f"{self.image_base}/{quote_plus(prompt)}?n=1&size={quote_plus(size)}"
        try:
            r = requests.get(url, headers=self.headers, timeout=120)
            r.raise_for_status()
            if out_path is None:
                safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prompt)[:40]
                out_path = f"pollinations_{safe_name or 'image'}.png"
            with open(out_path, "wb") as f:
                f.write(r.content)
            return out_path
        except Exception as e:
            raise ValueError(f"Pollinations image failed: {e}")


# ---------- CLI ----------
def ask_yes_no(prompt: str) -> bool:
    while True:
        ans = safe_input(prompt).strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please enter y or n.")


def ask_float(prompt: str, minv: float, maxv: float, default: float) -> float:
    while True:
        raw = safe_input(prompt).strip()
        try:
            return sanitize_float(raw or default, minv, maxv, default)
        except ValueError as e:
            print(e)


def ask_int(prompt: str, minv: int, maxv: int, default: int) -> int:
    while True:
        raw = safe_input(prompt).strip()
        try:
            return sanitize_int(raw or default, minv, maxv, default)
        except ValueError as e:
            print(e)


def choose_from_list(title: str, items: List[str]) -> int:
    print(f"\n{title}")
    for i, it in enumerate(items, 1):
        print(f"{i}. {it}")
    while True:
        raw = safe_input("Select a number: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                return idx
            print(f"Please select a number between 1 and {len(items)}.")
        except ValueError:
            print("Please enter a valid number.")


def chat_puter(client: PuterClient):
    # Choose model from Puter map (no pollinations-text here)
    models = list(client.model_to_driver.keys())
    idx = choose_from_list("Available models:", models)
    selected_model = models[idx]
    stream = ask_yes_no("Enable streaming? (y/n): ")
    test_mode = ask_yes_no("Enable test mode? (y/n): ")
    strict_model = ask_yes_no("Enforce strict model usage? (y/n): ")
    temperature = ask_float("Enter temperature (0-2, default 0.7): ", 0.0, 2.0, 0.7)
    max_tokens = ask_int("Max tokens (50-8000, default 1000): ", 50, 8000, 1000)
    show_used = ask_yes_no("Show used model after response? (y/n): ")

    options = {
        "model": selected_model,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    messages: List[Dict[str, Any]] = []
    print("\nChat started. Type 'exit' to quit.")
    while True:
        user_input = safe_input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        try:
            if stream:
                gen = client.ai_chat(
                    messages=messages,
                    options=options,
                    test_mode=test_mode,
                    strict_model=strict_model,
                )
                print("Assistant: ", end="", flush=True)
                buf = ""
                used_model = selected_model
                for chunk, used in gen:
                    if chunk:
                        chunk = str(chunk).replace("\x1b", "")
                        print(chunk, end="", flush=True)
                        buf += chunk
                        used_model = used
                print()
                if show_used:
                    print(f"(Used model: {used_model})")
                messages.append({"role": "assistant", "content": buf})
            else:
                res = client.ai_chat(
                    messages=messages,
                    options=options,
                    test_mode=test_mode,
                    strict_model=strict_model,
                )
                content = res["response"].get("result", {}).get("message", {}).get("content", "")
                used_model = res["used_model"]
                safe_content = str(content).replace("\x1b", "")
                print(f"Assistant: {safe_content}")
                if show_used:
                    print(f"(Used model: {used_model})")
                messages.append({"role": "assistant", "content": safe_content})
        except Exception as e:
            print(f"Error: {e}")


def chat_pollinations(pclient: PollinationsClient):
    temperature = ask_float("Enter temperature (0-2, default 0.7): ", 0.0, 2.0, 0.7)
    max_tokens = ask_int("Max tokens (50-8000, default 512): ", 50, 8000, 512)
    print("\nPollinations Chat started. Type 'exit' to quit.")
    while True:
        user_input = safe_input("\nYou: ")
        if user_input.strip().lower() == "exit":
            break
        try:
            reply = pclient.chat(user_input, temperature=temperature, max_tokens=max_tokens)
            print("Assistant:", reply)
        except Exception as e:
            print(f"Error: {e}")


def gen_image_puter(client: PuterClient):
    prompt = safe_input("Image prompt: ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return
    try:
        url = client.ai_txt2img(prompt, model="pollinations-image", test_mode=False)
        print("Puter driver returned image URL/data:")
        print(url)
    except Exception as e:
        print(f"Error: {e}")


def gen_image_pollinations(pclient: PollinationsClient):
    prompt = safe_input("Image prompt: ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return
    size = safe_input("Size (e.g., 512x512, 1024x1024). Default 1024x1024: ").strip() or "1024x1024"
    try:
        path = pclient.generate_image(prompt, size=size)
        print(f"Saved image to: {os.path.abspath(path)}")
    except Exception as e:
        print(f"Error: {e}")


def ocr_puter(client: PuterClient):
    src = safe_input("Image URL or local path for OCR: ").strip()
    if not src:
        print("Input required.")
        return
    try:
        if os.path.exists(src):
            with open(src, "rb") as f:
                text = client.ai_img2txt(f, test_mode=False)
        else:
            text = client.ai_img2txt(src, test_mode=False)
        print("\nOCR Result:\n", text or "(empty)")
    except Exception as e:
        print(f"Error: {e}")


def tts_puter(client: PuterClient):
    text = safe_input("Text to synthesize: ").strip()
    if not text:
        print("Input required.")
        return
    try:
        data = client.ai_txt2speech(text, options={"testMode": False})
        out = "puter_tts.mp3"
        with open(out, "wb") as f:
            f.write(data)
        print(f"Saved audio to: {os.path.abspath(out)}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    print("== Putergenai+PrimalCore All-in-One CLI ==")
    # --- Login loop ---
    attempts = 0
    client = PuterClient()
    while attempts < 3:
        username = safe_input("Enter your username: ")
        password = get_password("Enter your password: ")
        try:
            username = sanitize_string(username)
            password = sanitize_string(password)
            client.login(username, password)
            print("Login successful!")
            break
        except Exception as e:
            attempts += 1
            print(f"Login failed: {e}")
            if attempts >= 3:
                print("Maximum login attempts exceeded. Exiting.")
                sys.exit(1)

    if ask_yes_no("Enable debug logging to file? (y/n): "):
        print(f"(Debug logs in {LOG_FILE})")
        logger.setLevel(logging.DEBUG)
    else:
        # Quiet file logs
        logger.setLevel(logging.ERROR)

    pclient = PollinationsClient()

    # --- Main Menu loop ---
    while True:
        print("\n--- Main Menu ---")
        print("1) Chat (Puter models)")
        print("2) Chat (Pollinations)")
        print("3) Generate Image (Puter)")
        print("4) Generate Image (Pollinations)")
        print("5) OCR: Image → Text (Puter)")
        print("6) TTS: Text → Speech (Puter)")
        print("7) Exit")
        choice = safe_input("Select an option: ").strip()

        if choice == "1":
            chat_puter(client)
        elif choice == "2":
            chat_pollinations(pclient)
        elif choice == "3":
            gen_image_puter(client)
        elif choice == "4":
            gen_image_pollinations(pclient)
        elif choice == "5":
            ocr_puter(client)
        elif choice == "6":
            tts_puter(client)
        elif choice == "7":
            print("Bye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()
