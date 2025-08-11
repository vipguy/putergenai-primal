import requests
import json
from typing import Optional, List, Dict, Union, Generator, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
            "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
        }
        # Model to driver mapping for all models from puter.js documentation
        self.model_to_driver = {
            "gpt-4o-mini": "openai-completion",
            "gpt-4o": "openai-completion",
            "o1": "openai-completion",
            "o1-mini": "openai-completion",
            "o1-pro": "openai-completion",
            "o3": "openai-completion",
            "o3-mini": "openai-completion",
            "o4-mini": "openai-completion",
            "gpt-5": "openai-completion",
            "got-5-mini": "openai-completion",
            "gpt-5-nano": "openai-completion",
            "gpt-5-chat-latest": "openai-completion",
            "gpt-4.1": "openai-completion",
            "gpt-4.1-mini": "openai-completion",
            "gpt-4.1-nano": "openai-completion",
            "gpt-4.5-preview": "openai-completion",
            "claude-sonnet-4": "claude",
            "claude-opus-4": "claude",
            "claude-3-7-sonnet": "claude",
            "claude-3-5-sonnet": "claude",
            "deepseek-chat": "deepseek",
            "deepseek-reasoner": "deepseek",
            "gemini-2.0-flash": "google",
            "gemini-1.5-flash": "google",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "openrouter",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "openrouter",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "openrouter",
            "mistral-large-latest": "mistral",
            "pixtral-large-latest": "mistral",
            "codestral-latest": "mistral",
            "google/gemma-2-27b-it": "google",
            "grok-beta": "xai"
        }
        self.fallback_models = [
            "gpt-4.1-nano", "gpt-4o-mini", "claude-3-5-sonnet", "deepseek-chat", "mistral-large-latest", "grok-beta"
        ]
        self.max_retries = 3

    def login(self, username: str, password: str) -> str:
        payload = {
            "username": username,
            "password": password
        }
        try:
            response = requests.post(self.login_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get("proceed"):
                self.token = data["token"]
                logger.info("Login successful, token acquired")
                return self.token
            else:
                logger.error("Login failed: Invalid credentials")
                raise ValueError("Login failed. Please check your credentials.")
        except requests.RequestException as e:
            logger.error(f"Login error: {e}")
            raise ValueError(f"Login error: {e}")

    def _get_auth_headers(self) -> Dict[str, str]:
        if not self.token:
            logger.error("Authentication error: No token available")
            raise ValueError("Not authenticated. Please login first.")
        return {
            **self.headers,
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def fs_write(self, path: str, content: Union[str, bytes, Any]) -> Dict[str, Any]:
        """
        Write content to a file in Puter FS.
        If content is str or bytes, send directly. If file-like, read it.
        Returns the file info.
        """
        headers = self._get_auth_headers()
        headers.pop("Content-Type")
        if isinstance(content, str):
            content = content.encode('utf-8')
        if not isinstance(content, bytes):
            if hasattr(content, 'read'):
                content = content.read()
            else:
                logger.error("Invalid content type for fs_write")
                raise ValueError("Content must be str, bytes, or file-like object.")
        try:
            response = requests.post(f"{self.api_base}/write", params={"path": path}, data=content, headers=headers)
            response.raise_for_status()
            logger.info(f"File written successfully at {path}")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"fs_write error: {e}")
            raise

    def fs_read(self, path: str) -> bytes:
        """
        Read file content from Puter FS.
        """
        headers = self._get_auth_headers()
        try:
            response = requests.get(f"{self.api_base}/read", params={"path": path}, headers=headers)
            response.raise_for_status()
            logger.info(f"File read successfully from {path}")
            return response.content
        except requests.RequestException as e:
            logger.error(f"fs_read error: {e}")
            raise

    def fs_delete(self, path: str) -> None:
        """
        Delete a file or directory in Puter FS.
        """
        headers = self._get_auth_headers()
        try:
            response = requests.post(f"{self.api_base}/delete", params={"path": path}, headers=headers)
            response.raise_for_status()
            logger.info(f"File deleted successfully at {path}")
        except requests.RequestException as e:
            logger.error(f"fs_delete error: {e}")
            raise

    def ai_chat(
        self,
        prompt: Optional[Union[str, List[Dict[str, Any]]]] = None,
        options: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
        image_url: Optional[Union[str, List[str]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        retry_count: int = 0,
        strict_model: bool = False
    ) -> Union[Dict[str, Any], Generator[tuple[str, str], None, None]]:
        """
        AI chat completion, supporting multiple models with fallback and retries.
        Returns: For stream, a generator of (content, used_model) tuples; for non-stream, a dict with response and used_model.
        """
        if options is None:
            options = {}
        model = options.get("model", self.fallback_models[0])
        driver = self.model_to_driver.get(model, "openai-completion")
        stream = options.get("stream", False)
        temperature = options.get("temperature", 0.7)
        max_tokens = options.get("max_tokens", 1000)

        if messages is None:
            messages = []
            if prompt:
                if isinstance(prompt, str):
                    content = prompt
                else:
                    content = prompt
                if image_url:
                    if not isinstance(image_url, list):
                        image_url = [image_url]
                    content_parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
                    for url in image_url:
                        content_parts.append({"type": "image_url", "image_url": {"url": url}})
                    content = content_parts
                messages.append({"role": "user", "content": content})

        payload = {
            "interface": "puter-chat-completion",
            "driver": driver,
            "method": "complete",
            "args": {
                "messages": messages,
                "model": model  # Explicitly pass the model in args
            },
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tools": options.get("tools"),
            "testMode": test_mode
        }

        headers = self._get_auth_headers()
        try:
            logger.info(f"Sending ai_chat request with model {model}, driver {driver}, stream={stream}, test_mode={test_mode}, retry={retry_count}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            response = requests.post(f"{self.api_base}/drivers/call", json=payload, headers=headers, stream=stream)
            response.raise_for_status()

            def check_used_model(data: Dict[str, Any], requested_model: str, strict: bool) -> str:
                """Extract used model from response metadata and validate."""
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

            if stream:
                def generator():
                    for line in response.iter_lines():
                        if not line:
                            continue
                        logger.debug(f"Raw stream line: {line}")
                        try:
                            # Try parsing as JSON
                            data = json.loads(line)
                            used_model = check_used_model(data, model, strict_model)
                            if data.get("success", False):
                                # Handle claude-style response
                                if "result" in data and "message" in data["result"]:
                                    message_content = data["result"]["message"].get("content", "")
                                    if isinstance(message_content, str):
                                        yield message_content, used_model
                                        logger.info("Processed string content JSON streaming response")
                                    elif isinstance(message_content, list):
                                        content = message_content[0].get("text", "")
                                        if content:
                                            yield content, used_model
                                            logger.info("Processed list content JSON streaming response")
                                        else:
                                            logger.warning(f"No text in list content: {message_content}")
                                    else:
                                        logger.warning(f"Unexpected content type: {type(message_content)}")
                                # Handle OpenAI-style response
                                elif "choices" in data:
                                    content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content:
                                        yield content, used_model
                                        logger.info("Processed OpenAI-style JSON streaming response")
                                    else:
                                        logger.warning(f"No content in OpenAI-style response: {data}")
                                else:
                                    logger.warning(f"Unexpected JSON response format: {data}")
                            else:
                                logger.warning(f"Non-successful JSON response: {data}")
                        except json.JSONDecodeError:
                            # Handle SSE format
                            if line.startswith(b"data: "):
                                try:
                                    data = json.loads(line[6:])
                                    used_model = check_used_model(data, model, strict_model)
                                    content = data.get("text", "")
                                    if content:
                                        yield content, used_model
                                        logger.info("Processed SSE streaming response")
                                    else:
                                        logger.warning(f"No text in SSE data: {data}")
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid SSE data: {line}")
                            else:
                                # Treat as plain text
                                decoded_line = line.decode('utf-8', errors='ignore')
                                if decoded_line.strip():
                                    yield decoded_line, model  # No metadata in plain text
                                    logger.info("Processed plain text streaming response")
                                else:
                                    logger.warning(f"Empty or invalid stream line: {line}")
                logger.info("ai_chat stream request initiated")
                return generator()
            else:
                result = response.json()
                used_model = check_used_model(result, model, strict_model)
                logger.info(f"ai_chat request successful, used model: {used_model}")
                return {"response": result, "used_model": used_model}

        except requests.HTTPError as e:
            error_data = response.json() if 'response' in locals() else {}
            error_code = error_data.get("error", {}).get("code")
            if error_code == "no_implementation_available" and retry_count < self.max_retries:
                # Try with test_mode=True if not already enabled
                if not test_mode:
                    logger.warning(f"Model {model} not available, retrying with test_mode=True, attempt {retry_count + 1}")
                    return self.ai_chat(
                        prompt=prompt,
                        options=options,
                        test_mode=True,
                        image_url=image_url,
                        messages=messages,
                        retry_count=retry_count + 1,
                        strict_model=strict_model
                    )
                # Fallback to another model
                current_model_index = self.fallback_models.index(model) if model in self.fallback_models else -1
                next_model_index = current_model_index + 1 if current_model_index >= 0 else 0
                if next_model_index < len(self.fallback_models):
                    next_model = self.fallback_models[next_model_index]
                    logger.warning(f"Model {model} (driver {driver}) not available, retrying with model {next_model}, attempt {retry_count + 1}")
                    options["model"] = next_model
                    time.sleep(1)
                    return self.ai_chat(
                        prompt=prompt,
                        options=options,
                        test_mode=test_mode,
                        image_url=image_url,
                        messages=messages,
                        retry_count=retry_count + 1,
                        strict_model=strict_model
                    )
                else:
                    logger.error(f"No more fallback models available after {model}. Error: {error_data}")
                    raise ValueError(f"No implementation available for model {model}: {error_data.get('error', {}).get('message', str(e))}")
            logger.error(f"ai_chat error: {e}, response: {error_data}")
            raise ValueError(f"AI chat error: {error_data.get('error', {}).get('message', str(e))}")
        except requests.RequestException as e:
            logger.error(f"ai_chat request failed: {e}")
            raise

    def ai_img2txt(self, image: Union[str, Any], test_mode: bool = False) -> str:
        """
        Image to text (OCR).
        image: URL or file-like.
        """
        headers = self._get_auth_headers()
        try:
            if isinstance(image, str):
                payload = {"image_url": image, "testMode": test_mode}
                response = requests.post(f"{self.api_base}/ai/img2txt", json=payload, headers=headers)
            else:
                files = {"image": image}
                response = requests.post(f"{self.api_base}/ai/img2txt", files=files, data={"testMode": test_mode}, headers=headers)
            response.raise_for_status()
            logger.info("ai_img2txt request successful")
            return response.json().get("text")
        except requests.RequestException as e:
            logger.error(f"ai_img2txt error: {e}")
            raise

    def ai_txt2img(self, prompt: str, test_mode: bool = False) -> str:
        """
        Text to image.
        Returns image data URL.
        """
        payload = {"prompt": prompt, "testMode": test_mode}
        headers = self._get_auth_headers()
        try:
            response = requests.post(f"{self.api_base}/ai/txt2img", json=payload, headers=headers)
            response.raise_for_status()
            logger.info("ai_txt2img request successful")
            return response.json().get("image_url")
        except requests.RequestException as e:
            logger.error(f"ai_txt2img error: {e}")
            raise

    def ai_txt2speech(self, text: str, options: Optional[Dict[str, Any]] = None) -> bytes:
        """
        Text to speech.
        Returns MP3 bytes.
        """
        if options is None:
            options = {}
        payload = {"text": text, "testMode": options.get("testMode", False)}
        headers = self._get_auth_headers()
        try:
            response = requests.post(f"{self.api_base}/ai/txt2speech", json=payload, headers=headers)
            response.raise_for_status()
            logger.info("ai_txt2speech request successful")
            return response.content
        except requests.RequestException as e:
            logger.error(f"ai_txt2speech error: {e}")
            raise