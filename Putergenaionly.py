# -*- coding: utf-8 -*-
"""
Putergenai + PrimalCore AI Chat Only (Pydroid3 ready)
- Logs into Puter once
- Chat (ALL available Puter text models, Trinity Large default)
"""

import json
import logging
import re
import sys
import time
from getpass import getpass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests

# ---------- Logging ----------
LOG_FILE = "putergenai.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("puter_chat_cli")


# ---------- Helpers ----------
def sanitize_string(s, allow_empty: bool = False, allow_path: bool = False) -> str:
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


def sanitize_float(val, min_value=0.0, max_value=2.0, default=0.7) -> float:
    try:
        f = float(val)
    except (ValueError, TypeError):
        f = default
    if f < min_value or f > max_value:
        raise ValueError(f"Value must be between {min_value} and {max_value}.")
    return f


def sanitize_int(val, min_value: int, max_value: int, default: int) -> int:
    try:
        n = int(val)
    except (ValueError, TypeError):
        n = default
    if n < min_value or n > max_value:
        raise ValueError(f"Value must be between {min_value} and {max_value}.")
    return n


def sanitize_url(url: str) -> str:
    from urllib.parse import urlparse as _urlparse

    if not isinstance(url, str):
        raise ValueError("Invalid URL: not a string.")
    parsed = _urlparse(url)
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
        return input(prompt)


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
        # Static fallback mapping (used if live metadata is missing).
        self.model_to_driver: Dict[str, str] = {
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
            # Trinity Large Preview free model via Puter.
            "arcee-ai/trinity-large-preview:free": "openai-completion",
        }
        self.force_temperature_1_models = {
            "gpt-5-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
            "o3",
            "o1",
        }
        # Default fallback order: Trinity first.
        self.fallback_models: List[str] = [
            "arcee-ai/trinity-large-preview:free",
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "claude-3-5-sonnet-latest",
            "deepseek-chat",
        ]
        self.max_retries = 3
        self.live_models: List[Dict[str, Any]] = []

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

    # ---- Live model discovery (chat/text models) ----
    def fetch_live_models(self) -> List[Dict[str, Any]]:
        """
        Fetch list of available AI chat/completion models from the same source
        that backs puter.ai.listModels().
        """
        headers = self._auth_headers()
        try:
            r = requests.get(
                f"{self.api_base}/puterai/chat/models/details",
                headers=headers,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                self.live_models = data
            else:
                self.live_models = data.get("models", [])
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch live models, falling back to static list: {e}")
            self.live_models = []
        return self.live_models

    def get_all_text_models(self) -> List[Tuple[str, str]]:
        """
        Return a list of (model_id, provider) tuples for all chat/text models.
        If live metadata is available, use that; otherwise fall back to static mapping.
        """
        models = self.fetch_live_models()
        out: List[Tuple[str, str]] = []
        if models:
            for m in models:
                mid = m.get("id")
                provider = m.get("provider", "unknown")
                if not mid:
                    continue
                out.append((mid, provider))
        else:
            out = [(mid, drv) for mid, drv in self.model_to_driver.items()]

        # Ensure Trinity Large is present & first.
        if "arcee-ai/trinity-large-preview:free" not in [m[0] for m in out]:
            out.insert(0, ("arcee-ai/trinity-large-preview:free", "arcee"))
        else:
            out = sorted(
                out,
                key=lambda x: 0 if x[0] == "arcee-ai/trinity-large-preview:free" else 1,
            )

        return out

    # ---- Chat ----
    def ai_chat(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        options: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
        image_url: Optional[Union[str, List[str]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        retry_count: int = 0,
        strict_model: bool = False,
    ) -> Union[Dict[str, Any], Generator[Tuple[str, str], None, None]]:
        if options is None:
            options = {}
        model = options.get("model", self.fallback_models[0])

        # Resolve driver: from static map if available, else from live provider, else default.
        driver = self.model_to_driver.get(model)
        if not driver and self.live_models:
            for m in self.live_models:
                if m.get("id") == model:
                    driver = m.get("provider", "openai-completion")
                    break
        if not driver:
            driver = "openai-completion"

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


def choose_from_list_with_default(title: str, items: List[str], default_index: int = 0) -> int:
    print(f"\n{title}")
    for i, it in enumerate(items, 1):
        prefix = "*" if i - 1 == default_index else " "
        print(f"{prefix} {i}. {it}")
    while True:
        raw = safe_input("Select a number (Enter for default *): ").strip()
        if not raw:
            return default_index
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(items):
                return idx
            print(f"Please select a number between 1 and {len(items)}.")
        except ValueError:
            print("Please enter a valid number.")


def chat_puter(client: PuterClient):
    # Fetch all available chat models (live, with Trinity first).
    model_tuples = client.get_all_text_models()
    model_ids = [m[0] for m in model_tuples]

    # Default index is Trinity if present (get_all_text_models already puts it first).
    default_index = 0

    idx = choose_from_list_with_default(
        "Available Puter AI models (Trinity Large *default*):",
        model_ids,
        default_index=default_index,
    )
    selected_model = model_ids[idx]

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
    print(f"\nChat started with model: {selected_model}")
    print("Type 'exit' to quit.")
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


def main():
    print("== Putergenai+PrimalCore AI Chat Only CLI (All Puter Models, Trinity Default) ==")
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
        logger.setLevel(logging.ERROR)

    while True:
        print("\n--- Main Menu ---")
        print("1) Chat (All Puter models, Trinity default)")
        print("2) Exit")
        choice = safe_input("Select an option: ").strip()

        if choice == "1":
            chat_puter(client)
        elif choice == "2":
            print("Bye!")
            break
        else:
            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()

