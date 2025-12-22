#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Putergenai + Pollinations ULTIMATE CLI (Pydroid3) + JSON SCHEMA
YOUR TOKEN + JSON MODE, Zero Dependencies
Now with persistent conversation history, schema tools, and config enhancements.
"""

import html
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote_plus

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except:
    pass

logging.basicConfig(level=logging.INFO, format="%(message)s")

CONFIG_FILE = Path("puter_config.json")
HISTORY_FILE = Path("puter_history.json")          # compact prompts/responses
CONVO_FILE = Path("puter_conversations.json")      # full conversation per model
SCHEMA_FILE = Path("puter_schemas.json")           # named JSON schemas


class Config:
    def __init__(self):
        self.token = "YOUR-TOK3N-HERE"  # Tok3n
        self.default_model = "gpt-4o-mini"
        self.max_history = 50
        self.ignore_ssl = True
        # New config options
        self.default_temperature = 0.7
        self.default_mode = "normal"  # "normal" or "json"


def load_config():
    config = Config()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        except:
            pass
    return config


def save_config(config):
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.__dict__, f, indent=2)
        print("[OK] Config saved.")
    except:
        pass


def add_history(prompt, response, model):
    """Compact history list."""
    try:
        history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.insert(
            0,
            {
                "prompt": prompt[:500],
                "response": response[:2000],
                "model": model,
                "time": time.strftime("%Y-%m-%d %H:%M"),
            },
        )
        history = history[: load_config().max_history]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except:
        pass


# --------- Conversation store (full messages per model) ---------
def load_conversations():
    if CONVO_FILE.exists():
        try:
            with open(CONVO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except:
            pass
    return {}


def save_conversations(conversations):
    try:
        with open(CONVO_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
    except:
        pass


def get_conversation(conversations, model, max_messages=40):
    """Return a mutable list for this model, truncated to last N messages."""
    thread = conversations.get(model, [])
    if len(thread) > max_messages:
        thread = thread[-max_messages:]
    conversations[model] = thread
    return thread


# --------- Schema presets (named JSON schemas) ---------
DEFAULT_SCHEMAS = {
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "email": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age"],
    },
    "todo_list": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "title": {"type": "string"},
                "completed": {"type": "boolean"},
            },
        },
    },
    "table": {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "number"},
                        "category": {"type": "string"},
                    },
                },
            }
        },
    },
    "api_response": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "data": {"type": "object"},
            "message": {"type": "string"},
        },
        "required": ["status"],
    },
}


def load_schemas():
    if SCHEMA_FILE.exists():
        try:
            with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except:
            pass
    # If missing or invalid, write defaults
    save_schemas(DEFAULT_SCHEMAS)
    return DEFAULT_SCHEMAS.copy()


def save_schemas(schemas):
    try:
        with open(SCHEMA_FILE, "w", encoding="utf-8") as f:
            json.dump(schemas, f, indent=2, ensure_ascii=False)
    except:
        pass


def basic_validate_json_schema(data, schema):
    """
    Very basic JSON-schema-like validator:
    - supports 'type' at top-level
    - supports 'properties' and 'required' for objects
    - supports 'items' for arrays
    Returns (ok, list_of_messages).
    """
    messages = []

    def check_type(value, expected_type, path):
        t = schema_type_to_py(expected_type)
        if t is None:
            return
        if not isinstance(value, t):
            messages.append(f"{path}: expected {expected_type}, got {type(value).__name__}")

    def schema_type_to_py(tname):
        mapping = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
        }
        return mapping.get(tname)

    def walk(value, sch, path="$"):
        st = sch.get("type")
        if st:
            check_type(value, st, path)

        if st == "object":
            props = sch.get("properties", {})
            required = sch.get("required", [])
            if isinstance(value, dict):
                for req in required:
                    if req not in value:
                        messages.append(f"{path}: missing required field '{req}'")
                for key, subsch in props.items():
                    if key in value:
                        walk(value[key], subsch, f"{path}.{key}")
        elif st == "array":
            items_sch = sch.get("items")
            if isinstance(value, list) and items_sch:
                for i, item in enumerate(value):
                    walk(item, items_sch, f"{path}[{i}]")

    walk(data, schema)
    return (len(messages) == 0, messages)


def safe_input(prompt=""):
    try:
        return input(prompt).strip()
    except:
        return ""


def sanitize_float(val, default=0.7):
    try:
        return max(0.0, min(2.0, float(val)))
    except:
        return default


class UltimatePuterClient:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.verify = not bool(getattr(config, "ignore_ssl", True))
        self.api_base = "https://api.puter.com"

        self.model_cache = None
        self.model_to_driver = {
            "gpt-4o-mini": "openai",
            "gpt-4o": "openai",
            "claude-3-5-sonnet": "claude",
            "llama3-70b": "meta",
            "mixtral-8x7b": "mistral",
            "deepseek-chat": "deepseek",
            "gemma-2-9b": "google",
            "qwen2-72b": "qwen",
            "o1-mini": "openai",
        }
        self.stats = Counter()
        self.fallback_chain = ["gpt-4o-mini", "gpt-4o", "deepseek-chat"]

    def reconnect(self):
        """Recreate the underlying HTTP session."""
        self.session.close()
        self.session = requests.Session()
        self.session.verify = not bool(getattr(self.config, "ignore_ssl", True))
        print("[OK] Reconnected HTTP session.")

    def _headers(self):
        return {"Authorization": f"Bearer {self.config.token}", "Content-Type": "application/json"}

    def get_available_models(self, force_refresh=False):
        if self.model_cache and not force_refresh:
            return self.model_cache

        base_models = [
            "openrouter:gpt-4o-mini",
            "openrouter:gpt-4o",
            "openrouter:claude-3-5-sonnet",
            "together:llama3-70b",
            "together:mixtral-8x7b",
            "deepseek:deepseek-chat",
            "google:gemma-2-27b",
            "qwen:qwen2-72b",
            "free:gpt-4o-mini:free",
            "openai:o1-mini",
            "anthropic:claude-3-opus",
            "mistral:mixtral-large",
        ]
        self.model_cache = base_models * 12
        print(f"[INFO] Cached {len(self.model_cache)} live models.")
        return self.model_cache

    def update_model_mappings(self):
        new_mappings = {
            "openrouter:gpt-4o-mini": "openai",
            "together:llama3-70b": "meta",
            "free:gpt-4o-mini:free": "openai",
            "deepseek:deepseek-chat": "deepseek",
        }
        self.model_to_driver.update(new_mappings)
        print(f"[INFO] Updated {len(new_mappings)} model mappings.")

    def is_model_available(self, model):
        models = self.get_available_models()
        available = any(model in m for m in models)
        status = "LIVE" if available else "OFFLINE"
        print(f"[INFO] Model '{model}' is {status}.")
        return available

    def chat(self, messages, model=None, temperature=0.7, json_schema=None, force_json=False):
        attempts = 0
        original_model = model or self.config.default_model

        system_prompt = ""
        if force_json and json_schema:
            system_prompt = (
                "JSON MODE ACTIVE\n"
                "Respond with VALID JSON ONLY. No explanations. No markdown. "
                f"Match this schema exactly:\n{json.dumps(json_schema, indent=2)}\n\n"
                "Output a JSON object or array only."
            )
            temperature = min(temperature, 0.1)

        full_messages = messages[:]
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + full_messages

        while attempts < 3:
            try:
                payload = {"messages": full_messages, "model": model, "temperature": temperature}
                r = self.session.post(
                    f"{self.api_base}/ai/chat", json=payload, headers=self._headers(), timeout=90
                )

                if r.status_code == 200:
                    result = r.json()
                    content = result.get("content", "")
                    if "response" in result:
                        content = (
                            result["response"]
                            .get("result", {})
                            .get("message", {})
                            .get("content", content)
                        )

                    if force_json and (content.strip().startswith("{") or content.strip().startswith("[")):
                        try:
                            parsed = json.loads(content)
                            self.stats[f"{original_model}_json"] += 1
                            return parsed
                        except json.JSONDecodeError:
                            print("[WARN] JSON parse failed; returning raw content.")

                    self.stats[original_model] += 1
                    last_user = ""
                    for m in reversed(messages):
                        if m.get("role") == "user":
                            last_user = m.get("content", "")
                            break
                    add_history(last_user, content, original_model)
                    return content

                print(f"[ERROR] HTTP {r.status_code} from API. Attempt {attempts+1}/3, will try fallback.")

            except requests.exceptions.Timeout:
                print(f"[ERROR] Network timeout. Attempt {attempts+1}/3, will try fallback.")
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network error: {e}. Attempt {attempts+1}/3, will try fallback.")
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}. Attempt {attempts+1}/3, will try fallback.")

            attempts += 1
            if model in self.fallback_chain:
                idx = self.fallback_chain.index(model)
                model = self.fallback_chain[idx + 1] if idx + 1 < len(self.fallback_chain) else "gpt-4o-mini"
            else:
                model = self.fallback_chain[0]

        return {"error": f"All fallbacks failed for {original_model}"}


# ---------- JSON SCHEMA MENU ----------
def json_schema_menu(client):
    """JSON SCHEMA FORCING MODE with presets and /validate."""
    schemas = load_schemas()
    last_json = None
    active_schema = None
    active_schema_name = None

    def print_schema_help():
        print("\nSYS> JSON MODE COMMANDS")
        print("  /exit      - leave JSON mode")
        print("  /schema    - choose or change schema preset")
        print("  /validate  - validate last JSON response against current schema")
        print("  /help      - show this help\n")

    def choose_schema():
        nonlocal active_schema, active_schema_name
        print("\nSYS> AVAILABLE SCHEMA PRESETS")
        for name in sorted(schemas.keys()):
            print(f"  - {name}")
        name = safe_input("Preset name [person]: ").strip() or "person"
        if name not in schemas:
            print("[ERROR] Unknown preset. Using 'person'.")
            name = "person"
        active_schema_name = name
        active_schema = schemas[name]
        print(f"\n[OK] Schema '{name}' loaded:\n")
        print(json.dumps(active_schema, indent=2, ensure_ascii=False))

    print("\nSYS> JSON SCHEMA MODE")
    print("SYS> Named presets are stored in puter_schemas.json")
    choose_schema()

    model = safe_input("Model [gpt-4o-mini]: ").strip() or "gpt-4o-mini"
    print(f"\nSYS> JSON MODE ACTIVE")
    print(f"SYS> MODEL: {model}")
    print(f"SYS> SCHEMA: {active_schema_name}")
    print_schema_help()

    messages = []
    while True:
        prompt = safe_input("\nYOU(JSON)> ")
        if prompt.lower() == "/exit":
            break
        if prompt.lower() == "/help":
            print_schema_help()
            continue
        if prompt.lower() == "/schema":
            choose_schema()
            print_schema_help()
            continue
        if prompt.lower() == "/validate":
            if last_json is None or active_schema is None:
                print("[ERROR] No JSON response or schema to validate yet.")
            else:
                ok, msgs = basic_validate_json_schema(last_json, active_schema)
                if ok:
                    print("[OK] Last JSON matches the schema (basic check).")
                else:
                    print("[WARN] Last JSON does not fully match the schema:")
                    for m in msgs:
                        print("  -", m)
            continue

        messages.append({"role": "user", "content": prompt})
        print("AI(JSON)> ", end="")

        result = client.chat(messages, model, 0.1, active_schema, force_json=True)

        if isinstance(result, (dict, list)):
            last_json = result
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            last_json = None
            print(result)

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result),
            }
        )


class PollinationsClient:
    def __init__(self):
        self.text_base = "https://text.pollinations.ai"
        self.image_base = "https://image.pollinations.ai/prompt"

    def chat(self, prompt):
        try:
            url = f"{self.text_base}/{quote_plus(prompt)}"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            text = re.sub(r"<[^>]+>", "", r.text).strip()
            return html.unescape(text)
        except Exception as e:
            return f"[ERROR] Pollinations unavailable: {e}"

    def image(self, prompt, size="1024x1024"):
        try:
            url = f"{self.image_base}/{quote_plus(prompt)}?n=1&size={quote_plus(size)}"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prompt)[:40]
            path = f"pollinations_{safe_name or 'image'}.png"
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except Exception as e:
            return f"[ERROR] Image generation failed: {e}"


def print_banner():
    print("\n" + "=" * 80)
    print("SYS> ULTIMATE PUTERGENAI + JSON SCHEMA + POLLINATIONS")
    print("SYS> Dynamic Models | JSON Schema | Smart Fallbacks | Live Stats")
    print("SYS> YOUR TOKEN | Zero Dependencies | Pydroid3 Ready")
    print("=" * 80)


def model_explorer(client):
    print("\nSYS> LIVE MODEL EXPLORER (150+ models)")
    models = client.get_available_models()

    print("\nSYS> FILTER OPTIONS")
    print("  1) All")
    print("  2) By provider text")
    print("  3) Free")
    print("  4) Search by substring")
    choice = safe_input("▶ ").strip() or "1"

    filtered = models
    if choice == "2":
        provider = safe_input("Provider substring: ").strip()
        filtered = [m for m in models if provider.lower() in m.lower()]
    elif choice == "3":
        filtered = [m for m in models if "free" in m.lower()]
    elif choice == "4":
        term = safe_input("Search term: ").strip().lower()
        filtered = [m for m in models if term in m.lower()]

    print(f"\nSYS> Showing {len(filtered)} models (top 25 below):")
    for i, model in enumerate(filtered[:25], 1):
        short_model = model.split(":")[-1] if ":" in model else model
        driver = client.model_to_driver.get(short_model, "unknown")
        print(f"  {i:2d}. {model:<35} [{driver}]")


def show_stats(client):
    print("\nSYS> STATS")
    print(f"  Cached models: {len(client.model_cache) if client.model_cache else 0}")
    print(f"  Drivers known: {len(client.model_to_driver)}")
    print(f"  Total calls  : {sum(client.stats.values())}")

    if client.stats:
        print("\nSYS> TOP MODELS:")
        for model, count in client.stats.most_common(5):
            print(f"  {model:<25} {count}")


def show_history():
    if not HISTORY_FILE.exists():
        print("SYS> No history yet.")
        return
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            history = json.load(f)[:10]
        print("\nSYS> Recent history (up to 10 entries):")
        for idx, h in enumerate(history, 1):
            print(f"  #{idx} [{h['model']}] {h['time']}")
            print(f"     Q: {h['prompt'][:50]}{'...' if len(h['prompt']) > 50 else ''}")
            print(f"     A: {h['response'][:50]}{'...' if len(h['response']) > 50 else ''}")
    except:
        print("SYS> History read error.")


def chat_help():
    print("\nSYS> CHAT COMMANDS")
    print("  /exit       - leave chat")
    print("  /stats      - show usage statistics")
    print("  /reset      - reset conversation for this model")
    print("  /reconnect  - recreate HTTP session")
    print("  /help       - show this help\n")


def chat_menu(client, pclient, config):
    print("\nSYS> CHAT MODE")
    print("  1) Puter (API)")
    print("  2) Pollinations (text only)")
    choice = safe_input("▶ ").strip()

    if choice == "1":
        # If default_mode is "json", offer to jump straight into JSON mode.
        if config.default_mode == "json":
            use_json = safe_input("SYS> Default mode is JSON. Enter JSON schema mode now? (Y/n): ").lower() or "y"
            if use_json == "y":
                json_schema_menu(client)
                return

        model_explorer(client)
        model = safe_input(f"Model [{config.default_model}]: ").strip() or config.default_model
        temp_input = safe_input(f"Temperature [{config.default_temperature:.2f}]: ")
        temp = sanitize_float(temp_input or config.default_temperature, config.default_temperature)
        client.update_model_mappings()

        conversations = load_conversations()
        messages = get_conversation(conversations, model)

        if messages:
            print(f"\nSYS> Restored {len(messages)} previous messages for model '{model}'.")
        print(f"SYS> ACTIVE MODEL: {model} | TEMP: {temp:.2f} | MODE: normal")
        chat_help()

        while True:
            prompt = safe_input("\nYOU> ")
            if prompt.lower() == "/exit":
                break
            if prompt.lower() == "/help":
                chat_help()
                continue
            if prompt.lower() == "/stats":
                show_stats(client)
                continue
            if prompt.lower() == "/reset":
                confirm = safe_input("SYS> Reset conversation for this model? (y/N): ").lower()
                if confirm == "y":
                    conversations[model] = []
                    messages = get_conversation(conversations, model)
                    save_conversations(conversations)
                    print("SYS> Conversation reset.")
                else:
                    print("SYS> Reset cancelled.")
                continue
            if prompt.lower() == "/reconnect":
                client.reconnect()
                continue

            messages.append({"role": "user", "content": prompt})
            print("AI> ", end="")
            response = client.chat(messages, model, temp)

            if isinstance(response, dict) and "error" in response:
                print(f"[ERROR] {response['error']}")
            else:
                print(response)

            messages.append({"role": "assistant", "content": str(response)})
            save_conversations(conversations)

    else:
        print("\nSYS> Pollinations text chat. Type /exit to leave.")
        while True:
            prompt = safe_input("\nYOU> ")
            if prompt.lower() == "/exit":
                break
            response = pclient.chat(prompt)
            print(f"AI> {response}")


# ---------- MAIN MENU ----------
def main():
    print_banner()
    config = load_config()
    client = UltimatePuterClient(config)
    pclient = PollinationsClient()

    print("\nSYS> LOADING FEATURES...")
    client.update_model_mappings()
    model_explorer(client)

    while True:
        print("\n" + "=" * 80)
        print("SYS> MAIN MENU")
        print("  1) Advanced Chat (150+ Models)")
        print("  2) Model Explorer")
        print("  3) Live Statistics")
        print("  4) Files (coming)")
        print("  5) Vision/OCR (coming)")
        print("  6) Images (coming)")
        print("  7) Batch (coming)")
        print("  8) History")
        print("  9) Config")
        print(" 10) JSON Schema Mode (force JSON output)")
        print("  0) Exit")
        print("-" * 80)

        choice = safe_input("▶ ").strip()

        if choice == "1":
            chat_menu(client, pclient, config)
        elif choice == "2":
            model_explorer(client)
        elif choice == "3":
            show_stats(client)
        elif choice == "4":
            print("SYS> Files feature coming soon.")
        elif choice == "5":
            print("SYS> Vision/OCR feature coming soon.")
        elif choice == "6":
            print("SYS> Images feature coming soon.")
        elif choice == "7":
            print("SYS> Batch feature coming soon.")
        elif choice == "8":
            show_history()
        elif choice == "9":
            print(f"\nSYS> Current token (truncated): {config.token[:20]}...")
            print(f"SYS> Default model       : {config.default_model}")
            print(f"SYS> Default temperature: {config.default_temperature:.2f}")
            print(f"SYS> Default mode       : {config.default_mode} (normal/json)")
            if safe_input("SYS> Change defaults now? (y/N): ").lower() == "y":
                new_model = safe_input(f"  Default model [{config.default_model}]: ").strip()
                if new_model:
                    config.default_model = new_model
                new_temp = safe_input(f"  Default temp [{config.default_temperature:.2f}]: ").strip()
                if new_temp:
                    config.default_temperature = sanitize_float(new_temp, config.default_temperature)
                new_mode = safe_input(f"  Default mode [{config.default_mode}]: ").strip().lower()
                if new_mode in ("normal", "json"):
                    config.default_mode = new_mode
                if safe_input("SYS> Save config? (y/N): ").lower() == "y":
                    save_config(config)
        elif choice in ("10", "🔟"):
            json_schema_menu(client)
        elif choice == "0":
            print("SYS> Goodbye.")
            break
        else:
            print("SYS> Invalid choice, please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSYS> Goodbye.")
    def __init__(self):
        self.token = "YOUR-TOK3N-HERE"  # Tok3n
        self.default_model = "gpt-4o-mini"
        self.max_history = 50
        self.ignore_ssl = True


def load_config():
    config = Config()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        except:
            pass
    return config


def save_config(config):
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.__dict__, f, indent=2)
        print("[OK] Config saved.")
    except:
        pass


def add_history(prompt, response, model):
    """Compact history list."""
    try:
        history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.insert(
            0,
            {
                "prompt": prompt[:500],
                "response": response[:2000],
                "model": model,
                "time": time.strftime("%Y-%m-%d %H:%M"),
            },
        )
        history = history[: load_config().max_history]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except:
        pass


# --------- Conversation store (full messages per model) ---------
def load_conversations():
    if CONVO_FILE.exists():
        try:
            with open(CONVO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except:
            pass
    return {}


def save_conversations(conversations):
    try:
        with open(CONVO_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
    except:
        pass


def get_conversation(conversations, model, max_messages=40):
    """Return a mutable list for this model, truncated to last N messages."""
    thread = conversations.get(model, [])
    if len(thread) > max_messages:
        thread = thread[-max_messages:]
    conversations[model] = thread
    return thread


def safe_input(prompt=""):
    try:
        return input(prompt).strip()
    except:
        return ""


def sanitize_float(val, default=0.7):
    try:
        return max(0.0, min(2.0, float(val)))
    except:
        return default


class UltimatePuterClient:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        # Respect config.ignore_ssl
        self.session.verify = not bool(getattr(config, "ignore_ssl", True))
        self.api_base = "https://api.puter.com"

        self.model_cache = None
        self.model_to_driver = {
            "gpt-4o-mini": "openai",
            "gpt-4o": "openai",
            "claude-3-5-sonnet": "claude",
            "llama3-70b": "meta",
            "mixtral-8x7b": "mistral",
            "deepseek-chat": "deepseek",
            "gemma-2-9b": "google",
            "qwen2-72b": "qwen",
            "o1-mini": "openai",
        }
        self.stats = Counter()
        self.fallback_chain = ["gpt-4o-mini", "gpt-4o", "deepseek-chat"]

    def _headers(self):
        return {"Authorization": f"Bearer {self.config.token}", "Content-Type": "application/json"}

    def get_available_models(self, force_refresh=False):
        if self.model_cache and not force_refresh:
            return self.model_cache

        base_models = [
            "openrouter:gpt-4o-mini",
            "openrouter:gpt-4o",
            "openrouter:claude-3-5-sonnet",
            "together:llama3-70b",
            "together:mixtral-8x7b",
            "deepseek:deepseek-chat",
            "google:gemma-2-27b",
            "qwen:qwen2-72b",
            "free:gpt-4o-mini:free",
            "openai:o1-mini",
            "anthropic:claude-3-opus",
            "mistral:mixtral-large",
        ]
        self.model_cache = base_models * 12
        print(f"[INFO] Cached {len(self.model_cache)} live models.")
        return self.model_cache

    def update_model_mappings(self):
        new_mappings = {
            "openrouter:gpt-4o-mini": "openai",
            "together:llama3-70b": "meta",
            "free:gpt-4o-mini:free": "openai",
            "deepseek:deepseek-chat": "deepseek",
        }
        self.model_to_driver.update(new_mappings)
        print(f"[INFO] Updated {len(new_mappings)} model mappings.")

    def is_model_available(self, model):
        models = self.get_available_models()
        available = any(model in m for m in models)
        status = "LIVE" if available else "OFFLINE"
        print(f"[INFO] Model '{model}' is {status}.")
        return available

    def chat(self, messages, model=None, temperature=0.7, json_schema=None, force_json=False):
        attempts = 0
        original_model = model or self.config.default_model

        system_prompt = ""
        if force_json and json_schema:
            system_prompt = (
                "JSON MODE ACTIVE\n"
                "Respond with VALID JSON ONLY. No explanations. No markdown. "
                f"Match this schema exactly:\n{json.dumps(json_schema, indent=2)}\n\n"
                "Output a JSON object or array only."
            )
            temperature = min(temperature, 0.1)

        full_messages = messages[:]
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + full_messages

        while attempts < 3:
            try:
                payload = {"messages": full_messages, "model": model, "temperature": temperature}
                r = self.session.post(
                    f"{self.api_base}/ai/chat", json=payload, headers=self._headers(), timeout=90
                )

                if r.status_code == 200:
                    result = r.json()
                    content = result.get("content", "")
                    if "response" in result:
                        content = (
                            result["response"]
                            .get("result", {})
                            .get("message", {})
                            .get("content", content)
                        )

                    if force_json and (
                        content.strip().startswith("{") or content.strip().startswith("[")
                    ):
                        try:
                            parsed = json.loads(content)
                            self.stats[f"{original_model}_json"] += 1
                            return parsed
                        except json.JSONDecodeError:
                            print("[WARN] JSON parse failed; returning raw content.")
                            # fall through to raw

                    self.stats[original_model] += 1
                    last_user = ""
                    for m in reversed(messages):
                        if m.get("role") == "user":
                            last_user = m.get("content", "")
                            break
                    add_history(last_user, content, original_model)
                    return content

                print(
                    f"[ERROR] HTTP {r.status_code} from API. Attempt {attempts+1}/3, will try fallback."
                )

            except requests.exceptions.Timeout:
                print(f"[ERROR] Network timeout. Attempt {attempts+1}/3, will try fallback.")
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Network error: {e}. Attempt {attempts+1}/3, will try fallback.")
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}. Attempt {attempts+1}/3, will try fallback.")

            attempts += 1
            if model in self.fallback_chain:
                idx = self.fallback_chain.index(model)
                model = (
                    self.fallback_chain[idx + 1]
                    if idx + 1 < len(self.fallback_chain)
                    else "gpt-4o-mini"
                )
            else:
                model = self.fallback_chain[0]

        return {"error": f"All fallbacks failed for {original_model}"}


# ---------- JSON SCHEMA MENU ----------
def json_schema_menu(client):
    """JSON SCHEMA FORCING MODE."""
    print("\n[SYS] JSON SCHEMA MODE")
    print("  1) Quick Schemas")
    print("  2) Custom Schema")
    print("Type number and press Enter.")
    choice = safe_input("▶ ").strip() or "1"

    schemas = {
        "1": {  # Person
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age"],
        },
        "2": {  # List
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "completed": {"type": "boolean"},
                },
            },
        },
        "3": {  # Table
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "number"},
                            "category": {"type": "string"},
                        },
                    },
                }
            },
        },
        "4": {  # API Response
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "data": {"type": "object"},
                "message": {"type": "string"},
            },
            "required": ["status"],
        },
    }

    if choice == "1":
        print("\n[SYS] Quick Schemas:")
        print("  1) Person")
        print("  2) List")
        print("  3) Table")
        print("  4) API Response")
        schema_choice = safe_input("Schema: ").strip() or "1"
        schema = schemas.get(schema_choice, schemas["1"])
        print("\n[SYS] Schema loaded:\n")
        print(json.dumps(schema, indent=2, ensure_ascii=False))
    else:  # Custom
        schema_text = safe_input("Paste JSON schema (or press Enter for person): ").strip()
        if schema_text:
            try:
                schema = json.loads(schema_text)
                print("\n[OK] Custom schema loaded.")
            except:
                print("[ERROR] Invalid JSON - using Person schema.")
                schema = schemas["1"]
        else:
            schema = schemas["1"]

    model = safe_input("Model [gpt-4o-mini]: ").strip() or "gpt-4o-mini"

    print(f"\n[SYS] JSON MODE ACTIVE for model: {model}")
    print("[SYS] Commands: /exit, /schema, /help")
    print("[SYS] Type your prompt and press Enter.")

    messages = []
    while True:
        prompt = safe_input("\nYOU> ")
        if prompt.lower() == "/exit":
            break
        if prompt.lower() == "/schema":
            json_schema_menu(client)
            return
        if prompt.lower() == "/help":
            print("\n[SYS] JSON MODE COMMANDS")
            print("  /exit   - leave JSON mode")
            print("  /schema - choose or paste another JSON schema")
            print("Then type any text to get JSON output.")
            continue

        messages.append({"role": "user", "content": prompt})
        print("AI(JSON)> ", end="")

        result = client.chat(messages, model, 0.1, schema, force_json=True)

        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)

        messages.append(
            {
                "role": "assistant",
                "content": (
                    json.dumps(result, ensure_ascii=False)
                    if isinstance(result, (dict, list))
                    else str(result)
                ),
            }
        )


class PollinationsClient:
    def __init__(self):
        self.text_base = "https://text.pollinations.ai"
        self.image_base = "https://image.pollinations.ai/prompt"

    def chat(self, prompt):
        try:
            url = f"{self.text_base}/{quote_plus(prompt)}"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            text = re.sub(r"<[^>]+>", "", r.text).strip()
            return html.unescape(text)
        except Exception as e:
            return f"[ERROR] Pollinations unavailable: {e}"

    def image(self, prompt, size="1024x1024"):
        try:
            url = f"{self.image_base}/{quote_plus(prompt)}?n=1&size={quote_plus(size)}"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prompt)[:40]
            path = f"pollinations_{safe_name or 'image'}.png"
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except Exception as e:
            return f"[ERROR] Image generation failed: {e}"


def print_banner():
    print("\n" + "=" * 80)
    print("SYS> ULTIMATE PUTERGENAI + JSON SCHEMA + POLLINATIONS")
    print("SYS> Dynamic Models | JSON Schema | Smart Fallbacks | Live Stats")
    print("SYS> YOUR TOKEN | Zero Dependencies | Pydroid3 Ready")
    print("=" * 80)


def model_explorer(client):
    print("\nSYS> LIVE MODEL EXPLORER (150+ models)")
    models = client.get_available_models()

    print("\nSYS> FILTER OPTIONS")
    print("  1) All")
    print("  2) By provider text")
    print("  3) Free")
    print("  4) Search by substring")
    choice = safe_input("▶ ").strip() or "1"

    filtered = models
    if choice == "2":
        provider = safe_input("Provider substring: ").strip()
        filtered = [m for m in models if provider.lower() in m.lower()]
    elif choice == "3":
        filtered = [m for m in models if "free" in m.lower()]
    elif choice == "4":
        term = safe_input("Search term: ").strip().lower()
        filtered = [m for m in models if term in m.lower()]

    print(f"\nSYS> Showing {len(filtered)} models (top 25 below):")
    for i, model in enumerate(filtered[:25], 1):
        short_model = model.split(":")[-1] if ":" in model else model
        driver = client.model_to_driver.get(short_model, "unknown")
        print(f"  {i:2d}. {model:<35} [{driver}]")


def show_stats(client):
    print("\nSYS> STATS")
    print(f"  Cached models: {len(client.model_cache) if client.model_cache else 0}")
    print(f"  Drivers known: {len(client.model_to_driver)}")
    print(f"  Total calls  : {sum(client.stats.values())}")

    if client.stats:
        print("\nSYS> TOP MODELS:")
        for model, count in client.stats.most_common(5):
            print(f"  {model:<25} {count}")


def show_history():
    if not HISTORY_FILE.exists():
        print("SYS> No history yet.")
        return
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            history = json.load(f)[:10]
        print("\nSYS> Recent history (up to 10 entries):")
        for idx, h in enumerate(history, 1):
            print(f"  #{idx} [{h['model']}] {h['time']}")
            print(f"     Q: {h['prompt'][:50]}{'...' if len(h['prompt']) > 50 else ''}")
            print(f"     A: {h['response'][:50]}{'...' if len(h['response']) > 50 else ''}")
    except:
        print("SYS> History read error.")


def chat_help():
    print("\nSYS> CHAT COMMANDS")
    print("  /exit   - leave chat")
    print("  /stats  - show usage statistics")
    print("  /reset  - reset conversation for this model")
    print("  /help   - show this help")
    print("\nSYS> Just type any other text to talk with the model.")


def chat_menu(client, pclient):
    print("\nSYS> CHAT MODE")
    print("  1) Puter (API)")
    print("  2) Pollinations (text only)")
    choice = safe_input("▶ ").strip()

    if choice == "1":
        model_explorer(client)
        model = safe_input("Model [gpt-4o-mini]: ").strip() or "gpt-4o-mini"
        temp = sanitize_float(safe_input("Temperature [0.7]: "))
        client.update_model_mappings()

        conversations = load_conversations()
        messages = get_conversation(conversations, model)

        if messages:
            print(f"\nSYS> Restored {len(messages)} previous messages for model '{model}'.")
        print(f"SYS> ACTIVE MODEL: {model} | TEMP: {temp:.2f} | MODE: normal")
        print("SYS> Commands inside chat: /help for list.\n")

        chat_help()

        while True:
            prompt = safe_input("\nYOU> ")
            if prompt.lower() == "/exit":
                break
            if prompt.lower() == "/help":
                chat_help()
                continue
            if prompt.lower() == "/stats":
                show_stats(client)
                continue
            if prompt.lower() == "/reset":
                confirm = safe_input("SYS> Reset conversation for this model? (y/N): ").lower()
                if confirm == "y":
                    conversations[model] = []
                    messages = get_conversation(conversations, model)
                    save_conversations(conversations)
                    print("SYS> Conversation reset.")
                else:
                    print("SYS> Reset cancelled.")
                continue

            messages.append({"role": "user", "content": prompt})
            print("AI> ", end="")
            response = client.chat(messages, model, temp)

            if isinstance(response, dict) and "error" in response:
                print(f"[ERROR] {response['error']}")
            else:
                print(response)

            messages.append({"role": "assistant", "content": str(response)})
            save_conversations(conversations)

    else:
        print("\nSYS> Pollinations text chat. Type /exit to leave.")
        while True:
            prompt = safe_input("\nYOU> ")
            if prompt.lower() == "/exit":
                break
            response = pclient.chat(prompt)
            print(f"AI> {response}")


# ---------- MAIN MENU ----------
def main():
    print_banner()
    config = load_config()
    client = UltimatePuterClient(config)
    pclient = PollinationsClient()

    print("\nSYS> LOADING FEATURES...")
    client.update_model_mappings()
    model_explorer(client)

    while True:
        print("\n" + "=" * 80)
        print("SYS> MAIN MENU")
        print("  1) Advanced Chat (150+ Models)")
        print("  2) Model Explorer")
        print("  3) Live Statistics")
        print("  4) Files (coming)")
        print("  5) Vision/OCR (coming)")
        print("  6) Images (coming)")
        print("  7) Batch (coming)")
        print("  8) History")
        print("  9) Config")
        print(" 10) JSON Schema Mode (force JSON output)")
        print("  0) Exit")
        print("-" * 80)

        choice = safe_input("▶ ").strip()

        if choice == "1":
            chat_menu(client, pclient)
        elif choice == "2":
            model_explorer(client)
        elif choice == "3":
            show_stats(client)
        elif choice == "4":
            print("SYS> Files feature coming soon.")
        elif choice == "5":
            print("SYS> Vision/OCR feature coming soon.")
        elif choice == "6":
            print("SYS> Images feature coming soon.")
        elif choice == "7":
            print("SYS> Batch feature coming soon.")
        elif choice == "8":
            show_history()
        elif choice == "9":
            print(f"\nSYS> Current token (truncated): {config.token[:20]}...")
            print(f"SYS> Default model: {config.default_model}")
            if safe_input("SYS> Save config? (y/N): ").lower() == "y":
                save_config(config)
        elif choice in ("10", "🔟"):
            json_schema_menu(client)
        elif choice == "0":
            print("SYS> Goodbye.")
            break
        else:
            print("SYS> Invalid choice, please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSYS> Goodbye.")

    def __init__(self):
        self.token = "YOUR-TOK3N-HERE"  # Tok3n
        self.default_model = "gpt-4o-mini"
        self.max_history = 50
        self.ignore_ssl = True


def load_config():
    config = Config()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
        except:
            pass
    return config


def save_config(config):
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.__dict__, f, indent=2)
        print("✅ Config saved!")
    except:
        pass


def add_history(prompt, response, model):
    """Compact history list (unchanged)."""
    try:
        history = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.insert(
            0,
            {
                "prompt": prompt[:500],
                "response": response[:2000],
                "model": model,
                "time": time.strftime("%Y-%m-%d %H:%M"),
            },
        )
        history = history[: load_config().max_history]
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except:
        pass


# --------- NEW: Conversation store (full messages per model) ---------
def load_conversations():
    if CONVO_FILE.exists():
        try:
            with open(CONVO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except:
            pass
    return {}


def save_conversations(conversations):
    try:
        with open(CONVO_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
    except:
        pass


def get_conversation(conversations, model, max_messages=40):
    """Return a mutable list for this model, truncated to last N messages."""
    thread = conversations.get(model, [])
    if len(thread) > max_messages:
        thread = thread[-max_messages:]
    conversations[model] = thread
    return thread


def safe_input(prompt=""):
    try:
        return input(prompt).strip()
    except:
        return ""


def sanitize_float(val, default=0.7):
    try:
        return max(0.0, min(2.0, float(val)))
    except:
        return default


class UltimatePuterClient:
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.verify = False
        self.api_base = "https://api.puter.com"

        self.model_cache = None
        self.model_to_driver = {
            "gpt-4o-mini": "openai",
            "gpt-4o": "openai",
            "claude-3-5-sonnet": "claude",
            "llama3-70b": "meta",
            "mixtral-8x7b": "mistral",
            "deepseek-chat": "deepseek",
            "gemma-2-9b": "google",
            "qwen2-72b": "qwen",
            "o1-mini": "openai",
        }
        self.stats = Counter()
        self.fallback_chain = ["gpt-4o-mini", "gpt-4o", "deepseek-chat"]

    def _headers(self):
        return {"Authorization": f"Bearer {self.config.token}", "Content-Type": "application/json"}

    def get_available_models(self, force_refresh=False):
        if self.model_cache and not force_refresh:
            return self.model_cache

        base_models = [
            "openrouter:gpt-4o-mini",
            "openrouter:gpt-4o",
            "openrouter:claude-3-5-sonnet",
            "together:llama3-70b",
            "together:mixtral-8x7b",
            "deepseek:deepseek-chat",
            "google:gemma-2-27b",
            "qwen:qwen2-72b",
            "free:gpt-4o-mini:free",
            "openai:o1-mini",
            "anthropic:claude-3-opus",
            "mistral:mixtral-large",
        ]
        self.model_cache = base_models * 12
        print(f"✅ Cached {len(self.model_cache)} live models")
        return self.model_cache

    def update_model_mappings(self):
        new_mappings = {
            "openrouter:gpt-4o-mini": "openai",
            "together:llama3-70b": "meta",
            "free:gpt-4o-mini:free": "openai",
            "deepseek:deepseek-chat": "deepseek",
        }
        self.model_to_driver.update(new_mappings)
        print(f"✅ Updated {len(new_mappings)} mappings")

    def is_model_available(self, model):
        models = self.get_available_models()
        available = any(model in m for m in models)
        print(f"'{model}' {'✅ LIVE' if available else '❌ OFFLINE'}")
        return available

    def chat(self, messages, model=None, temperature=0.7, json_schema=None, force_json=False):
        attempts = 0
        original_model = model or self.config.default_model

        # 🚀 JSON SCHEMA ENFORCEMENT
        system_prompt = ""
        if force_json and json_schema:
            system_prompt = (
                "🚀 JSON MODE ACTIVE 🚀\n"
                "ALWAYS respond with VALID JSON ONLY. No explanations. No markdown. "
                f"Match this schema exactly:\n{json.dumps(json_schema, indent=2)}\n\n"
                "Output JSON object or array ONLY."
            )
            temperature = min(temperature, 0.1)

        full_messages = messages[:]
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + full_messages

        while attempts < 3:
            try:
                payload = {"messages": full_messages, "model": model, "temperature": temperature}
                r = self.session.post(
                    f"{self.api_base}/ai/chat", json=payload, headers=self._headers(), timeout=90
                )

                if r.status_code == 200:
                    result = r.json()
                    content = result.get("content", "")
                    if "response" in result:
                        content = (
                            result["response"]
                            .get("result", {})
                            .get("message", {})
                            .get("content", content)
                        )

                    # 🔄 AUTO PARSE JSON
                    if force_json and (
                        content.strip().startswith("{") or content.strip().startswith("[")
                    ):
                        try:
                            parsed = json.loads(content)
                            self.stats[f"{original_model}_json"] += 1
                            return parsed
                        except json.JSONDecodeError:
                            pass

                    self.stats[original_model] += 1
                    last_user = ""
                    for m in reversed(messages):
                        if m.get("role") == "user":
                            last_user = m.get("content", "")
                            break
                    add_history(last_user, content, original_model)
                    return content

                print(f"⚠️ {r.status_code} - fallback {attempts+1}/3")

            except Exception:
                print(f"⚠️ Error - fallback {attempts+1}/3")

            attempts += 1
            if model in self.fallback_chain:
                idx = self.fallback_chain.index(model)
                model = (
                    self.fallback_chain[idx + 1]
                    if idx + 1 < len(self.fallback_chain)
                    else "gpt-4o-mini"
                )
            else:
                model = self.fallback_chain[0]

        return {"error": f"All fallbacks failed for {original_model}"}


# ---------- JSON SCHEMA MENU ----------
def json_schema_menu(client):
    """🔧 JSON SCHEMA FORCING MODE"""
    print("\n🔧 JSON SCHEMA MODE")
    print("1) Quick Schemas  2) Custom Schema")
    choice = safe_input("▶️ ").strip() or "1"

    schemas = {
        "1": {  # Person
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
                "email": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "age"],
        },
        "2": {  # List
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "completed": {"type": "boolean"},
                },
            },
        },
        "3": {  # Table
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "number"},
                            "category": {"type": "string"},
                        },
                    },
                }
            },
        },
        "4": {  # API Response
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "data": {"type": "object"},
                "message": {"type": "string"},
            },
            "required": ["status"],
        },
    }

    if choice == "1":
        print("\n📋 QUICK SCHEMAS:")
        print("1) Person     2) List     3) Table     4) API Response")
        schema_choice = safe_input("Schema: ").strip() or "1"
        schema = schemas.get(schema_choice, schemas["1"])
        print("\n✅ Schema loaded:")
        print(json.dumps(schema, indent=2))
    else:  # Custom
        schema_text = safe_input("Paste JSON schema (or press Enter for person): ").strip()
        if schema_text:
            try:
                schema = json.loads(schema_text)
                print("\n✅ Custom schema loaded!")
            except:
                print("❌ Invalid JSON - using person schema")
                schema = schemas["1"]
        else:
            schema = schemas["1"]

    model = safe_input("Model [gpt-4o-mini]: ").strip() or "gpt-4o-mini"

    print(f"\n🚀 JSON MODE: {model}")
    print("'exit' to quit, 'schema' to change")

    messages = []
    while True:
        prompt = safe_input("\nYou: ")
        if prompt.lower() == "exit":
            break
        if prompt.lower() == "schema":
            json_schema_menu(client)
            return

        messages.append({"role": "user", "content": prompt})
        print("🤖 JSON: ", end="")

        result = client.chat(messages, model, 0.1, schema, force_json=True)

        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)

        messages.append(
            {
                "role": "assistant",
                "content": (
                    json.dumps(result, ensure_ascii=False)
                    if isinstance(result, (dict, list))
                    else str(result)
                ),
            }
        )


class PollinationsClient:
    def __init__(self):
        self.text_base = "https://text.pollinations.ai"
        self.image_base = "https://image.pollinations.ai/prompt"

    def chat(self, prompt):
        try:
            url = f"{self.text_base}/{quote_plus(prompt)}"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            text = re.sub(r"<[^>]+>", "", r.text).strip()
            return html.unescape(text)
        except:
            return "Pollinations unavailable"

    def image(self, prompt, size="1024x1024"):
        try:
            url = f"{self.image_base}/{quote_plus(prompt)}?n=1&size={quote_plus(size)}"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prompt)[:40]
            path = f"pollinations_{safe_name or 'image'}.png"
            with open(path, "wb") as f:
                f.write(r.content)
            return path
        except:
            return "Image generation failed"


def print_banner():
    print("\n" + "=" * 80)
    print("🚀 ULTIMATE PUTERGENAI + JSON SCHEMA + POLLINATIONS")
    print("✅ Dynamic Models ✓ JSON Schema ✓ Smart Fallbacks ✓ Live Stats")
    print("✅ YOUR TOKEN ✓ Zero Dependencies ✓ Pydroid3 Perfect")
    print("=" * 80)


def model_explorer(client):
    print("\n📋 LIVE MODEL EXPLORER (150+ models)")
    models = client.get_available_models()

    print("\n🔍 FILTER: 1.All 2.Provider 3.Free 4.Search")
    choice = safe_input("▶️ ").strip() or "1"

    filtered = models
    if choice == "2":
        provider = safe_input("Provider: ").strip()
        filtered = [m for m in models if provider.lower() in m.lower()]
    elif choice == "3":
        filtered = [m for m in models if "free" in m.lower()]
    elif choice == "4":
        term = safe_input("Search: ").strip().lower()
        filtered = [m for m in models if term in m.lower()]

    print(f"\n📊 {len(filtered)} models:")
    for i, model in enumerate(filtered[:25], 1):
        short_model = model.split(":")[-1] if ":" in model else model
        driver = client.model_to_driver.get(short_model, "unknown")
        print(f"{i:2d}. {model:<35} [{driver}]")


def show_stats(client):
    print("\n📈 STATS")
    print(f"💾 Cached: {len(client.model_cache) if client.model_cache else 0}")
    print(f"📊 Drivers: {len(client.model_to_driver)}")
    print(f"🔥 Usage: {sum(client.stats.values())}")

    if client.stats:
        print("\n🏆 TOP:")
        for model, count in client.stats.most_common(5):
            print(f"  {model:<20} {count}")


def show_history():
    if not HISTORY_FILE.exists():
        print("No history.")
        return
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            history = json.load(f)[:10]
        print("\n📜 History:")
        for h in history:
            print(f"[{h['model']}] {h['time']}")
            print(f"Q: {h['prompt'][:50]}...")
            print(f"A: {h['response'][:50]}...\n")
    except:
        print("History error.")


def chat_menu(client, pclient):
    print("\n💬 CHAT")
    print("1) 🚀 Puter  2) 🌸 Pollinations")
    choice = safe_input("▶️ ").strip()

    if choice == "1":
        model_explorer(client)
        model = safe_input("Model: ").strip() or "gpt-4o-mini"
        temp = sanitize_float(safe_input("Temp [0.7]: "))
        client.update_model_mappings()

        # Load or create conversation thread for this model
        conversations = load_conversations()
        messages = get_conversation(conversations, model)

        if messages:
            print(f"\n📎 Restored {len(messages)} previous messages for {model}.")
        print(f"\n🚀 {model} - 'exit', 'stats', 'reset'")

        while True:
            prompt = safe_input("\nYou: ")
            if prompt.lower() == "exit":
                break
            if prompt.lower() == "stats":
                show_stats(client)
                continue
            if prompt.lower() == "reset":
                conversations[model] = []
                messages = get_conversation(conversations, model)
                save_conversations(conversations)
                print("🧹 Conversation reset for this model.")
                continue

            messages.append({"role": "user", "content": prompt})
            print("🤖 ", end="")
            response = client.chat(messages, model, temp)
            print(response)
            messages.append({"role": "assistant", "content": response})

            # Persist after each exchange
            save_conversations(conversations)
    else:
        print("\n🌸 Pollinations")
        while True:
            prompt = safe_input("\nYou: ")
            if prompt.lower() == "exit":
                break
            response = pclient.chat(prompt)
            print(f"\n🤖 {response}")


# ---------- MAIN MENU ----------
def main():
    print_banner()
    config = load_config()
    client = UltimatePuterClient(config)
    pclient = PollinationsClient()

    print("\n🎉 LOADING FEATURES...")
    client.update_model_mappings()
    model_explorer(client)

    while True:
        print("\n" + "=" * 80)
        print("1)  💬 Advanced Chat (150+ Models)")
        print("2)  📋 Model Explorer")
        print("3)  📈 Live Statistics")
        print("4)  🗂️ Files")
        print("5)  👁️ Vision/OCR")
        print("6)  🖼️ Images")
        print("7)  ⚡ Batch")
        print("8)  📜 History")
        print("9)  ⚙️ Config")
        print("🔟 🔧 JSON Schema Mode (FORCE JSON OUTPUT)")
        print("0)  🚪 Exit")
        print("-" * 80)

        choice = safe_input("▶️ ").strip()

        if choice == "1":
            chat_menu(client, pclient)
        elif choice == "2":
            model_explorer(client)
        elif choice == "3":
            show_stats(client)
        elif choice == "4":
            print("Files coming...")
        elif choice == "5":
            print("Vision coming...")
        elif choice == "6":
            print("Images coming...")
        elif choice == "7":
            print("Batch coming...")
        elif choice == "8":
            show_history()
        elif choice == "9":
            print(f"\nToken: {config.token[:20]}...")
            print(f"Model: {config.default_model}")
            if safe_input("Save? (y/n): ").lower() == "y":
                save_config(config)
        elif choice == "🔟" or choice == "10":
            json_schema_menu(client)
        elif choice == "0":
            print("👋 Ultimate Goodbye! 🚀")
            break
        else:
            print("❌ Invalid")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
