import json
import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional, Union
from urllib.parse import urlparse

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_string(s, allow_empty=False, allow_path=False):
    """
    Sanitize user input for usernames, passwords, and file paths only.
    For chat and prompt text, allow natural language (no sanitization).
    """
    if not isinstance(s, str):
        raise ValueError("Input must be a string.")
    s = s.strip()
    if not allow_empty and not s:
        raise ValueError("Input cannot be empty.")
    if allow_path:
        pattern = r'^[\w\-\.\/]+$'
        if not re.match(pattern, s):
            raise ValueError("Input contains invalid characters for path.")
    return s


def sanitize_url(url):
    if not isinstance(url, str):
        raise ValueError("Invalid URL: not a string.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Invalid URL.")
    return url


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
        # Model to driver mapping for all models from puter.js documentation
        self.model_to_driver = {
            # OpenAI GPT-5.1 models
            "gpt-5.1": "openai-completion",
            "gpt-5.1-codex": "openai-completion",
            "gpt-5.1-codex-mini": "openai-completion",
            "gpt-5.1-chat-latest": "openai-completion",
            # OpenAI GPT-5 models
            "gpt-5-2025-08-07": "openai-completion",
            "gpt-5": "openai-completion",
            "gpt-5-mini-2025-08-07": "openai-completion",
            "gpt-5-mini": "openai-completion",
            "gpt-5-nano-2025-08-07": "openai-completion",
            "gpt-5-nano": "openai-completion",
            "gpt-5-chat-latest": "openai-completion",
            # OpenAI GPT-4o models
            "gpt-4o": "openai-completion",
            "gpt-4o-mini": "openai-completion",
            # OpenAI o-series models
            "o1": "openai-completion",
            "o1-mini": "openai-completion",
            "o1-pro": "openai-completion",
            "o3": "openai-completion",
            "o3-mini": "openai-completion",
            "o4-mini": "openai-completion",
            # OpenAI GPT-4.1 models
            "gpt-4.1": "openai-completion",
            "gpt-4.1-mini": "openai-completion",
            "gpt-4.1-nano": "openai-completion",
            # OpenAI GPT-4.5 models
            "gpt-4.5-preview": "openai-completion",
            # Claude models
            "claude-opus-4-5": "claude",
            "claude-opus-4-5-latest": "claude",
            "claude-opus-4.5": "claude",
            "claude-haiku-4-5-20251001": "claude",
            "claude-haiku-4.5": "claude",
            "claude-haiku-4-5": "claude",
            "claude-sonnet-4-5-20250929": "claude",
            "claude-sonnet-4.5": "claude",
            "claude-sonnet-4-5": "claude",
            "claude-opus-4-1-20250805": "claude",
            "claude-opus-4-1": "claude",
            "claude-opus-4-20250514": "claude",
            "claude-opus-4": "claude",
            "claude-opus-4-latest": "claude",
            "claude-sonnet-4-20250514": "claude",
            "claude-sonnet-4": "claude",
            "claude-sonnet-4-latest": "claude",
            "claude-3-7-sonnet-20250219": "claude",
            "claude-3-7-sonnet-latest": "claude",
            "claude-3-5-sonnet-20241022": "claude",
            "claude-3-5-sonnet-latest": "claude",
            "claude-3-5-sonnet-20240620": "claude",
            "claude-3-haiku-20240307": "claude",
            # Mistral models
            "mistral-large-latest": "mistral",
            "mistral-medium-2508": "mistral",
            "mistral-medium-latest": "mistral",
            "mistral-medium": "mistral",
            "ministral-3b-2410": "mistral",
            "ministral-3b-latest": "mistral",
            "ministral-8b-2410": "mistral",
            "ministral-8b-latest": "mistral",
            "open-mistral-7b": "mistral",
            "mistral-tiny": "mistral",
            "mistral-tiny-2312": "mistral",
            "open-mistral-nemo": "mistral",
            "open-mistral-nemo-2407": "mistral",
            "mistral-tiny-2407": "mistral",
            "mistral-tiny-latest": "mistral",
            "pixtral-large-2411": "mistral",
            "pixtral-large-latest": "mistral",
            "mistral-large-pixtral-2411": "mistral",
            "codestral-2508": "mistral",
            "codestral-latest": "mistral",
            "devstral-small-2507": "mistral",
            "devstral-small-latest": "mistral",
            "pixtral-12b-2409": "mistral",
            "pixtral-12b": "mistral",
            "pixtral-12b-latest": "mistral",
            "mistral-small-2506": "mistral",
            "mistral-small-latest": "mistral",
            "magistral-medium-2509": "mistral",
            "magistral-medium-latest": "mistral",
            "magistral-small-2509": "mistral",
            "magistral-small-latest": "mistral",
            "mistral-moderation-2411": "mistral",
            "mistral-moderation-latest": "mistral",
            "mistral-ocr-2505": "mistral",
            "mistral-ocr-latest": "mistral",
            # xAI/Grok models
            "grok-beta": "xai",
            "grok-vision-beta": "xai",
            "grok-3": "xai",
            "grok-3-fast": "xai",
            "grok-3-mini": "xai",
            "grok-3-mini-fast": "xai",
            "grok-2-vision": "xai",
            "grok-2": "xai",
            # DeepSeek models
            "deepseek-chat": "deepseek",
            "deepseek-reasoner": "deepseek",
            # Google Gemini models
            "gemini-1.5-flash": "google",
            "gemini-2.0-flash": "google",
            "gemini-2.0-flash-lite": "google",
            "gemini-2.5-flash": "google",
            "gemini-2.5-flash-lite": "google",
            "gemini-2.5-pro": "google",
            "gemini-3-pro-preview": "google",
            # TogetherAI models (using together-ai driver)
            "togetherai:togethercomputer/Refuel-Llm-V2": "together-ai",
            "togetherai:cartesia/sonic": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": "together-ai",
            "togetherai:cartesia/sonic-2": "together-ai",
            "togetherai:meta-llama/LlamaGuard-7b": "together-ai",
            "togetherai:togethercomputer/MoA-1": "together-ai",
            "togetherai:meta-llama/LlamaGuard-2-8b": "together-ai",
            "togetherai:meta-llama/Llama-3.3-70B-Instruct-Turbo": "together-ai",
            "togetherai:Qwen/Qwen2.5-72B-Instruct-Turbo": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-dev": "together-ai",
            "togetherai:Alibaba-NLP/gte-modernbert-base": "together-ai",
            "togetherai:mistralai/Mistral-Small-24B-Instruct-2501": "together-ai",
            "togetherai:marin-community/marin-8b-instruct": "together-ai",
            "togetherai:kwaivgI/kling-1.6-standard": "together-ai",
            "togetherai:meta-llama/LlamaGuard-3-11B-Vision-Turbo": "together-ai",
            "togetherai:black-forest-labs/FLUX.2-flex": "together-ai",
            "togetherai:meta-llama/Meta-Llama-Guard-3-8B": "together-ai",
            "togetherai:deepseek-ai/DeepSeek-R1": "together-ai",
            "togetherai:Qwen/Qwen3-Next-80B-A3B-Thinking": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-dev-lora": "together-ai",
            "togetherai:HiDream-ai/HiDream-I1-Full": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-krea-dev": "together-ai",
            "togetherai:togethercomputer/MoA-1-Turbo": "together-ai",
            "togetherai:Lykon/DreamShaper": "together-ai",
            "togetherai:HiDream-ai/HiDream-I1-Dev": "together-ai",
            "togetherai:Qwen/Qwen-Image": "together-ai",
            "togetherai:RunDiffusion/Juggernaut-pro-flux": "together-ai",
            "togetherai:google/imagen-4.0-preview": "together-ai",
            "togetherai:google/imagen-4.0-ultra": "together-ai",
            "togetherai:google/veo-3.0": "together-ai",
            "togetherai:minimax/hailuo-02": "together-ai",
            "togetherai:stabilityai/stable-diffusion-3-medium": "together-ai",
            "togetherai:deepcogito/cogito-v2-preview-llama-405B": "together-ai",
            "togetherai:deepcogito/cogito-v2-preview-llama-70B": "together-ai",
            "togetherai:minimax/video-01-director": "together-ai",
            "togetherai:togethercomputer/m2-bert-80M-32k-retrieval": "together-ai",
            "togetherai:deepseek-ai/DeepSeek-R1-0528-tput": "together-ai",
            "togetherai:google/flash-image-2.5": "together-ai",
            "togetherai:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "together-ai",
            "togetherai:moonshotai/Kimi-K2-Thinking": "together-ai",
            "togetherai:Qwen/Qwen3-Next-80B-A3B-Instruct": "together-ai",
            "togetherai:ServiceNow-AI/Apriel-1.5-15b-Thinker": "together-ai",
            "togetherai:scb10x/scb10x-typhoon-2-1-gemma3-12b": "together-ai",
            "togetherai:meta-llama/Llama-Guard-4-12B": "together-ai",
            "togetherai:HiDream-ai/HiDream-I1-Fast": "together-ai",
            "togetherai:Wan-AI/Wan2.2-T2V-A14B": "together-ai",
            "togetherai:ByteDance/Seedance-1.0-pro": "together-ai",
            "togetherai:google/veo-3.0-fast-audio": "together-ai",
            "togetherai:vidu/vidu-q1": "together-ai",
            "togetherai:intfloat/multilingual-e5-large-instruct": "together-ai",
            "togetherai:meta-llama/Llama-4-Scout-17B-16E-Instruct": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "together-ai",
            "togetherai:ByteDance-Seed/Seedream-3.0": "together-ai",
            "togetherai:ByteDance-Seed/Seedream-4.0": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro": "together-ai",
            "togetherai:meta-llama/Llama-3.2-3B-Instruct-Turbo": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-kontext-max": "together-ai",
            "togetherai:ideogram/ideogram-3.0": "together-ai",
            "togetherai:mixedbread-ai/Mxbai-Rerank-Large-V2": "together-ai",
            "togetherai:Salesforce/Llama-Rank-V1": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-schnell": "together-ai",
            "togetherai:deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "together-ai",
            "togetherai:Qwen/Qwen2.5-VL-72B-Instruct": "together-ai",
            "togetherai:meta-llama/Llama-3-70b-chat-hf": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-kontext-dev": "together-ai",
            "togetherai:zai-org/GLM-4.5-Air-FP8": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-pro": "together-ai",
            "togetherai:black-forest-labs/FLUX.1.1-pro": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3-70B-Instruct-Turbo": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-schnell-Free": "together-ai",
            "togetherai:Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": "together-ai",
            "togetherai:black-forest-labs/FLUX.1-kontext-pro": "together-ai",
            "togetherai:mistralai/Mixtral-8x7B-Instruct-v0.1": "together-ai",
            "togetherai:nvidia/NVIDIA-Nemotron-Nano-9B-v2": "together-ai",
            "togetherai:openai/gpt-oss-120b": "together-ai",
            "togetherai:openai/sora-2": "together-ai",
            "togetherai:kwaivgI/kling-2.1-standard": "together-ai",
            "togetherai:google/imagen-4.0-fast": "together-ai",
            "togetherai:Qwen/Qwen2.5-7B-Instruct-Turbo": "together-ai",
            "togetherai:mistralai/Mistral-7B-Instruct-v0.3": "together-ai",
            "togetherai:google/gemma-3n-E4B-it": "together-ai",
            "togetherai:deepseek-ai/DeepSeek-V3": "together-ai",
            "togetherai:kwaivgI/kling-2.1-master": "together-ai",
            "togetherai:google/veo-3.0-audio": "together-ai",
            "togetherai:Rundiffusion/Juggernaut-Lightning-Flux": "together-ai",
            "togetherai:Wan-AI/Wan2.2-I2V-A14B": "together-ai",
            "togetherai:google/veo-2.0": "together-ai",
            "togetherai:google/veo-3.0-fast": "together-ai",
            "togetherai:kwaivgI/kling-1.6-pro": "together-ai",
            "togetherai:vidu/vidu-2.0": "together-ai",
            "togetherai:kwaivgI/kling-2.1-pro": "together-ai",
            "togetherai:Virtue-AI/VirtueGuard-Text-Lite": "together-ai",
            "togetherai:Qwen/Qwen3-235B-A22B-Instruct-2507-tput": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3-8B-Instruct-Lite": "together-ai",
            "togetherai:Qwen/Qwen3-235B-A22B-fp8-tput": "together-ai",
            "togetherai:togethercomputer/Refuel-Llm-V2-Small": "together-ai",
            "togetherai:arize-ai/qwen-2-1.5b-instruct": "together-ai",
            "togetherai:moonshotai/Kimi-K2-Instruct-0905": "together-ai",
            "togetherai:deepcogito/cogito-v2-preview-llama-109B-MoE": "together-ai",
            "togetherai:hexgrad/Kokoro-82M": "together-ai",
            "togetherai:google/gemini-3-pro-image": "together-ai",
            "togetherai:mercor/cwm": "together-ai",
            "togetherai:keith-aditya/kimi-k2-instruct": "together-ai",
            "togetherai:pangram/mistral-small-2501": "together-ai",
            "togetherai:black-forest-labs/FLUX.2-pro": "together-ai",
            "togetherai:black-forest-labs/FLUX.2-dev": "together-ai",
            "togetherai:zai-org/GLM-4.6": "together-ai",
            "togetherai:openai/whisper-large-v3": "together-ai",
            "togetherai:canopylabs/orpheus-3b-0.1-ft": "together-ai",
            "togetherai:meta-llama/Llama-3.1-405B-Instruct": "together-ai",
            "togetherai:meta-llama/Llama-3-70b-hf": "together-ai",
            "togetherai:Qwen/Qwen3-235B-A22B-Thinking-2507": "together-ai",
            "togetherai:Qwen/Qwen2.5-72B-Instruct": "together-ai",
            "togetherai:openai/gpt-oss-20b": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3.1-70B-Instruct-Reference": "together-ai",
            "togetherai:BAAI/bge-large-en-v1.5": "together-ai",
            "togetherai:meta-llama/Llama-3.2-1B-Instruct": "together-ai",
            "togetherai:Qwen/Qwen2.5-14B-Instruct": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3.1-8B-Instruct-Reference": "together-ai",
            "togetherai:meta-llama/Meta-Llama-3-8B-Instruct": "together-ai",
            "togetherai:BAAI/bge-base-en-v1.5": "together-ai",
            "togetherai:deepseek-ai/DeepSeek-V3.1": "together-ai",
            "togetherai:mistralai/Mistral-7B-Instruct-v0.2": "together-ai",
            # OpenRouter models (using openrouter driver)
            "openrouter:anthropic/claude-opus-4.5": "openrouter",
            "openrouter:openrouter/bert-nebulon-alpha": "openrouter",
            "openrouter:allenai/olmo-3-32b-think": "openrouter",
            "openrouter:allenai/olmo-3-7b-instruct": "openrouter",
            "openrouter:allenai/olmo-3-7b-think": "openrouter",
            "openrouter:google/gemini-3-pro-image-preview": "openrouter",
            "openrouter:x-ai/grok-4.1-fast:free": "openrouter",
            "openrouter:google/gemini-3-pro-preview": "openrouter",
            "openrouter:deepcogito/cogito-v2.1-671b": "openrouter",
            "openrouter:openai/gpt-5.1": "openrouter",
            "openrouter:openai/gpt-5.1-chat": "openrouter",
            "openrouter:openai/gpt-5.1-codex": "openrouter",
            "openrouter:openai/gpt-5.1-codex-mini": "openrouter",
            "openrouter:kwaipilot/kat-coder-pro:free": "openrouter",
            "openrouter:moonshotai/kimi-linear-48b-a3b-instruct": "openrouter",
            "openrouter:moonshotai/kimi-k2-thinking": "openrouter",
            "openrouter:amazon/nova-premier-v1": "openrouter",
            "openrouter:perplexity/sonar-pro-search": "openrouter",
            "openrouter:mistralai/voxtral-small-24b-2507": "openrouter",
            "openrouter:openai/gpt-oss-safeguard-20b": "openrouter",
            "openrouter:nvidia/nemotron-nano-12b-v2-vl:free": "openrouter",
            "openrouter:nvidia/nemotron-nano-12b-v2-vl": "openrouter",
            "openrouter:minimax/minimax-m2": "openrouter",
            "openrouter:liquid/lfm2-8b-a1b": "openrouter",
            "openrouter:liquid/lfm-2.2-6b": "openrouter",
            "openrouter:ibm-granite/granite-4.0-h-micro": "openrouter",
            "openrouter:deepcogito/cogito-v2-preview-llama-405b": "openrouter",
            "openrouter:openai/gpt-5-image-mini": "openrouter",
            "openrouter:anthropic/claude-haiku-4.5": "openrouter",
            "openrouter:qwen/qwen3-vl-8b-thinking": "openrouter",
            "openrouter:qwen/qwen3-vl-8b-instruct": "openrouter",
            "openrouter:openai/gpt-5-image": "openrouter",
            "openrouter:openai/o3-deep-research": "openrouter",
            "openrouter:openai/o4-mini-deep-research": "openrouter",
            "openrouter:nvidia/llama-3.3-nemotron-super-49b-v1.5": "openrouter",
            "openrouter:baidu/ernie-4.5-21b-a3b-thinking": "openrouter",
            "openrouter:google/gemini-2.5-flash-image": "openrouter",
            "openrouter:qwen/qwen3-vl-30b-a3b-thinking": "openrouter",
            "openrouter:qwen/qwen3-vl-30b-a3b-instruct": "openrouter",
            "openrouter:openai/gpt-5-pro": "openrouter",
            "openrouter:z-ai/glm-4.6": "openrouter",
            "openrouter:z-ai/glm-4.6:exacto": "openrouter",
            "openrouter:anthropic/claude-sonnet-4.5": "openrouter",
            "openrouter:deepseek/deepseek-v3.2-exp": "openrouter",
            "openrouter:thedrummer/cydonia-24b-v4.1": "openrouter",
            "openrouter:relace/relace-apply-3": "openrouter",
            "openrouter:google/gemini-2.5-flash-preview-09-2025": "openrouter",
            "openrouter:google/gemini-2.5-flash-lite-preview-09-2025": "openrouter",
            "openrouter:qwen/qwen3-vl-235b-a22b-thinking": "openrouter",
            "openrouter:qwen/qwen3-vl-235b-a22b-instruct": "openrouter",
            "openrouter:qwen/qwen3-max": "openrouter",
            "openrouter:qwen/qwen3-coder-plus": "openrouter",
            "openrouter:openai/gpt-5-codex": "openrouter",
            "openrouter:deepseek/deepseek-v3.1-terminus:exacto": "openrouter",
            "openrouter:deepseek/deepseek-v3.1-terminus": "openrouter",
            "openrouter:x-ai/grok-4-fast": "openrouter",
            "openrouter:alibaba/tongyi-deepresearch-30b-a3b:free": "openrouter",
            "openrouter:alibaba/tongyi-deepresearch-30b-a3b": "openrouter",
            "openrouter:qwen/qwen3-coder-flash": "openrouter",
            "openrouter:opengvlab/internvl3-78b": "openrouter",
            "openrouter:qwen/qwen3-next-80b-a3b-thinking": "openrouter",
            "openrouter:qwen/qwen3-next-80b-a3b-instruct": "openrouter",
            "openrouter:meituan/longcat-flash-chat:free": "openrouter",
            "openrouter:meituan/longcat-flash-chat": "openrouter",
            "openrouter:qwen/qwen-plus-2025-07-28": "openrouter",
            "openrouter:qwen/qwen-plus-2025-07-28:thinking": "openrouter",
            "openrouter:nvidia/nemotron-nano-9b-v2:free": "openrouter",
            "openrouter:nvidia/nemotron-nano-9b-v2": "openrouter",
            "openrouter:moonshotai/kimi-k2-0905": "openrouter",
            "openrouter:moonshotai/kimi-k2-0905:exacto": "openrouter",
            "openrouter:deepcogito/cogito-v2-preview-llama-70b": "openrouter",
            "openrouter:deepcogito/cogito-v2-preview-llama-109b-moe": "openrouter",
            "openrouter:deepcogito/cogito-v2-preview-deepseek-671b": "openrouter",
            "openrouter:stepfun-ai/step3": "openrouter",
            "openrouter:qwen/qwen3-30b-a3b-thinking-2507": "openrouter",
            "openrouter:x-ai/grok-code-fast-1": "openrouter",
            "openrouter:nousresearch/hermes-4-70b": "openrouter",
            "openrouter:nousresearch/hermes-4-405b": "openrouter",
            "openrouter:google/gemini-2.5-flash-image-preview": "openrouter",
            "openrouter:deepseek/deepseek-chat-v3.1": "openrouter",
            "openrouter:openai/gpt-4o-audio-preview": "openrouter",
            "openrouter:mistralai/mistral-medium-3.1": "openrouter",
            "openrouter:baidu/ernie-4.5-21b-a3b": "openrouter",
            "openrouter:baidu/ernie-4.5-vl-28b-a3b": "openrouter",
            "openrouter:z-ai/glm-4.5": "openrouter",
            "openrouter:ai21/jamba-mini-1.7": "openrouter",
            "openrouter:ai21/jamba-large-1.7": "openrouter",
            "openrouter:openai/gpt-5-chat": "openrouter",
            "openrouter:openai/gpt-5": "openrouter",
            "openrouter:openai/gpt-5-mini": "openrouter",
            "openrouter:openai/gpt-5-nano": "openrouter",
            "openrouter:openai/gpt-oss-120b:exacto": "openrouter",
            "openrouter:openai/gpt-oss-120b": "openrouter",
            "openrouter:openai/gpt-oss-20b:free": "openrouter",
            "openrouter:openai/gpt-oss-20b": "openrouter",
            "openrouter:anthropic/claude-opus-4.1": "openrouter",
            "openrouter:anthropic/claude-sonnet-4": "openrouter",
            "openrouter:mistralai/codestral-2508": "openrouter",
            "openrouter:qwen/qwen3-coder-30b-a3b-instruct": "openrouter",
            "openrouter:qwen/qwen3-30b-a3b-instruct-2507": "openrouter",
            "openrouter:z-ai/glm-4.5": "openrouter",
            "openrouter:z-ai/glm-4.5-air:free": "openrouter",
            "openrouter:z-ai/glm-4.5-air": "openrouter",
            "openrouter:qwen/qwen3-235b-a22b-thinking-2507": "openrouter",
            "openrouter:z-ai/glm-4-32b": "openrouter",
            "openrouter:qwen/qwen3-coder:free": "openrouter",
            "openrouter:qwen/qwen3-coder": "openrouter",
            "openrouter:qwen/qwen3-coder:exacto": "openrouter",
            "openrouter:bytedance/ui-tars-1.5-7b": "openrouter",
            "openrouter:google/gemini-2.5-flash-lite": "openrouter",
            "openrouter:qwen/qwen3-235b-a22b-2507": "openrouter",
            "openrouter:switchpoint/router": "openrouter",
            "openrouter:moonshotai/kimi-k2:free": "openrouter",
            "openrouter:moonshotai/kimi-k2": "openrouter",
            "openrouter:thudm/glm-4.1v-9b-thinking": "openrouter",
            "openrouter:mistralai/devstral-medium": "openrouter",
            "openrouter:mistralai/devstral-small": "openrouter",
            "openrouter:cognitivecomputations/dolphin-mistral-24b-venice-edition:free": "openrouter",
            "openrouter:x-ai/grok-4": "openrouter",
            "openrouter:google/gemma-3n-e2b-it:free": "openrouter",
            "openrouter:tencent/hunyuan-a13b-instruct": "openrouter",
            "openrouter:tngtech/deepseek-r1t2-chimera:free": "openrouter",
            "openrouter:tngtech/deepseek-r1t2-chimera": "openrouter",
            "openrouter:morph/morph-v3-large": "openrouter",
            "openrouter:morph/morph-v3-fast": "openrouter",
            "openrouter:baidu/ernie-4.5-vl-424b-a47b": "openrouter",
            "openrouter:baidu/ernie-4.5-300b-a47b": "openrouter",
            "openrouter:thedrummer/anubis-70b-v1.1": "openrouter",
            "openrouter:inception/mercury": "openrouter",
            "openrouter:mistralai/mistral-small-3.2-24b-instruct:free": "openrouter",
            "openrouter:mistralai/mistral-small-3.2-24b-instruct": "openrouter",
            "openrouter:minimax/minimax-m1": "openrouter",
            "openrouter:google/gemini-2.5-flash": "openrouter",
            "openrouter:google/gemini-2.5-pro": "openrouter",
            "openrouter:moonshotai/kimi-dev-72b": "openrouter",
            "openrouter:openai/o3-pro": "openrouter",
            "openrouter:x-ai/grok-3-mini": "openrouter",
            "openrouter:x-ai/grok-3": "openrouter",
            "openrouter:mistralai/magistral-small-2506": "openrouter",
            "openrouter:mistralai/magistral-medium-2506:thinking": "openrouter",
            "openrouter:mistralai/magistral-medium-2506": "openrouter",
            "openrouter:google/gemini-2.5-pro-preview": "openrouter",
            "openrouter:deepseek/deepseek-r1-0528-qwen3-8b:free": "openrouter",
            "openrouter:deepseek/deepseek-r1-0528-qwen3-8b": "openrouter",
            "openrouter:deepseek/deepseek-r1-0528:free": "openrouter",
            "openrouter:deepseek/deepseek-r1-0528": "openrouter",
            "openrouter:minimax/minimax-01": "openrouter",
            "openrouter:mistralai/codestral-2501": "openrouter",
            "openrouter:microsoft/phi-4": "openrouter",
            "openrouter:sao10k/l3.1-70b-hanami-x1": "openrouter",
            "openrouter:deepseek/deepseek-chat": "openrouter",
            "openrouter:sao10k/l3.3-euryale-70b": "openrouter",
            "openrouter:openai/o1": "openrouter",
            "openrouter:cohere/command-a": "openrouter",
            "openrouter:google/gemini-2.0-flash-lite-001": "openrouter",
            "openrouter:anthropic/claude-3.7-sonnet:thinking": "openrouter",
            "openrouter:anthropic/claude-3.7-sonnet": "openrouter",
            "openrouter:mistralai/mistral-saba": "openrouter",
            "openrouter:meta-llama/llama-guard-4-12b": "openrouter",
            "openrouter:openai/o3-mini-high": "openrouter",
            "openrouter:google/gemini-2.0-flash-001": "openrouter",
            "openrouter:qwen/qwen-vl-plus": "openrouter",
            "openrouter:aion-labs/aion-1.0": "openrouter",
            "openrouter:aion-labs/aion-1.0-mini": "openrouter",
            "openrouter:aion-labs/aion-rp-llama-3.1-8b": "openrouter",
            "openrouter:qwen/qwen-vl-max": "openrouter",
            "openrouter:qwen/qwen-turbo": "openrouter",
            "openrouter:qwen/qwen2.5-vl-72b-instruct": "openrouter",
            "openrouter:qwen/qwen-plus": "openrouter",
            "openrouter:qwen/qwen-max": "openrouter",
            "openrouter:openai/o3-mini": "openrouter",
            "openrouter:mistralai/mistral-small-24b-instruct-2501:free": "openrouter",
            "openrouter:mistralai/mistral-small-24b-instruct-2501": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-qwen-32b": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-qwen-14b": "openrouter",
            "openrouter:perplexity/sonar-reasoning": "openrouter",
            "openrouter:perplexity/sonar": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-llama-70b:free": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-llama-70b": "openrouter",
            "openrouter:deepseek/deepseek-r1:free": "openrouter",
            "openrouter:deepseek/deepseek-r1": "openrouter",
            "openrouter:cohere/command-r7b-12-2024": "openrouter",
            "openrouter:cohere/command-r-plus-08-2024": "openrouter",
            "openrouter:sao10k/l3.1-euryale-70b": "openrouter",
            "openrouter:qwen/qwen-vl-plus": "openrouter",
            "openrouter:qwen/qwen-vl-max": "openrouter",
            "openrouter:qwen/qwen-turbo": "openrouter",
            "openrouter:qwen/qwen2.5-vl-72b-instruct": "openrouter",
            "openrouter:qwen/qwen-plus": "openrouter",
            "openrouter:qwen/qwen-max": "openrouter",
            "openrouter:openai/o3-mini": "openrouter",
            "openrouter:mistralai/mistral-small-24b-instruct-2501:free": "openrouter",
            "openrouter:mistralai/mistral-small-24b-instruct-2501": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-qwen-32b": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-qwen-14b": "openrouter",
            "openrouter:perplexity/sonar-reasoning": "openrouter",
            "openrouter:perplexity/sonar": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-llama-70b:free": "openrouter",
            "openrouter:deepseek/deepseek-r1-distill-llama-70b": "openrouter",
            "openrouter:deepseek/deepseek-r1:free": "openrouter",
            "openrouter:deepseek/deepseek-r1": "openrouter",
            "openrouter:cohere/command-r7b-12-2024": "openrouter",
            "openrouter:cohere/command-r-plus-08-2024": "openrouter",
            "openrouter:sao10k/l3.1-euryale-70b": "openrouter",
            "openrouter:qwen/qwen2.5-vl-7b-instruct": "openrouter",
            "openrouter:microsoft/phi-3.5-mini-128k-instruct": "openrouter",
            "openrouter:nousresearch/hermes-3-llama-3.1-70b": "openrouter",
            "openrouter:nousresearch/hermes-3-llama-3.1-405b:free": "openrouter",
            "openrouter:nousresearch/hermes-3-llama-3.1-405b": "openrouter",
            "openrouter:openai/chatgpt-4o-latest": "openrouter",
            "openrouter:sao10k/l3-lunaris-8b": "openrouter",
            "openrouter:openai/gpt-4o-2024-11-20": "openrouter",
            "openrouter:meta-llama/llama-3.1-405b": "openrouter",
            "openrouter:meta-llama/llama-3.1-8b-instruct": "openrouter",
            "openrouter:meta-llama/llama-3.1-405b-instruct": "openrouter",
            "openrouter:meta-llama/llama-3.1-70b-instruct": "openrouter",
            "openrouter:mistralai/mistral-nemo:free": "openrouter",
            "openrouter:mistralai/mistral-nemo": "openrouter",
            "openrouter:openai/gpt-4o-mini-2024-07-18": "openrouter",
            "openrouter:openai/gpt-4o-mini": "openrouter",
            "openrouter:google/gemma-2-27b-it": "openrouter",
            "openrouter:google/gemma-2-9b-it": "openrouter",
            "openrouter:sao10k/l3-euryale-70b": "openrouter",
            "openrouter:nousresearch/hermes-2-pro-llama-3-8b": "openrouter",
            "openrouter:mistralai/mistral-7b-instruct:free": "openrouter",
            "openrouter:mistralai/mistral-7b-instruct": "openrouter",
            "openrouter:mistralai/mistral-7b-instruct-v0.3": "openrouter",
            "openrouter:microsoft/phi-3-mini-128k-instruct": "openrouter",
            "openrouter:microsoft/phi-3-medium-128k-instruct": "openrouter",
            "openrouter:meta-llama/llama-guard-2-8b": "openrouter",
            "openrouter:openai/gpt-4o-2024-05-13": "openrouter",
            "openrouter:openai/gpt-4o": "openrouter",
            "openrouter:openai/gpt-4o:extended": "openrouter",
            "openrouter:meta-llama/llama-3-70b-instruct": "openrouter",
            "openrouter:meta-llama/llama-3-8b-instruct": "openrouter",
            "openrouter:mistralai/mixtral-8x22b-instruct": "openrouter",
            "openrouter:microsoft/wizardlm-2-8x22b": "openrouter",
            "openrouter:openai/gpt-4-turbo": "openrouter",
            "openrouter:anthropic/claude-3-haiku": "openrouter",
            "openrouter:anthropic/claude-3-opus": "openrouter",
            "openrouter:mistralai/mistral-large": "openrouter",
            "openrouter:openai/gpt-3.5-turbo-0613": "openrouter",
            "openrouter:openai/gpt-4-turbo-preview": "openrouter",
            "openrouter:mistralai/mistral-small": "openrouter",
            "openrouter:mistralai/mistral-tiny": "openrouter",
            "openrouter:mistralai/mistral-7b-instruct-v0.2": "openrouter",
            "openrouter:mistralai/mixtral-8x7b-instruct": "openrouter",
            "openrouter:neversleep/noromaid-20b": "openrouter",
            "openrouter:alpindale/goliath-120b": "openrouter",
            "openrouter:openrouter/auto": "openrouter",
            "openrouter:openai/gpt-4-1106-preview": "openrouter",
            "openrouter:openai/gpt-3.5-turbo-instruct": "openrouter",
            "openrouter:openai/gpt-3.5-turbo-16k": "openrouter",
            "openrouter:mancer/weaver": "openrouter",
            "openrouter:undi95/remm-slerp-l2-13b": "openrouter",
            "openrouter:gryphe/mythomax-l2-13b": "openrouter",
            "openrouter:openai/gpt-4-0314": "openrouter",
            "openrouter:openai/gpt-4": "openrouter",
            "openrouter:openai/gpt-3.5-turbo": "openrouter",
            # Special/test models
            "model-fallback-test-1": "openai-completion",
            "costly": "openai-completion",
            "abuse": "openai-completion",
            "fake": "openai-completion",
            # Additional OpenRouter models
            "openrouter:alfredpros/codellama-7b-instruct-solidity": "openrouter",
            "openrouter:allenai/olmo-2-0325-32b-instruct": "openrouter",
            "openrouter:amazon/nova-lite-v1": "openrouter",
            "openrouter:amazon/nova-micro-v1": "openrouter",
            "openrouter:amazon/nova-pro-v1": "openrouter",
            "openrouter:anthracite-org/magnum-v4-72b": "openrouter",
            "openrouter:anthropic/claude-3.5-haiku": "openrouter",
            "openrouter:anthropic/claude-3.5-haiku-20241022": "openrouter",
            "openrouter:anthropic/claude-3.5-sonnet": "openrouter",
            "openrouter:anthropic/claude-opus-4": "openrouter",
            "openrouter:arcee-ai/coder-large": "openrouter",
            "openrouter:arcee-ai/maestro-reasoning": "openrouter",
            "openrouter:arcee-ai/spotlight": "openrouter",
            "openrouter:arcee-ai/virtuoso-large": "openrouter",
            "openrouter:arliai/qwq-32b-arliai-rpr-v1": "openrouter",
            "openrouter:arliai/qwq-32b-arliai-rpr-v1:free": "openrouter",
            "openrouter:cohere/command-r-08-2024": "openrouter",
            "openrouter:deepseek/deepseek-chat-v3-0324": "openrouter",
            "openrouter:deepseek/deepseek-chat-v3-0324:free": "openrouter",
            "openrouter:deepseek/deepseek-prover-v2": "openrouter",
            "openrouter:eleutherai/llemma_7b": "openrouter",
            "openrouter:google/gemini-2.0-flash-exp:free": "openrouter",
            "openrouter:google/gemini-2.5-pro-preview-05-06": "openrouter",
            "openrouter:google/gemma-3-12b-it": "openrouter",
            "openrouter:google/gemma-3-12b-it:free": "openrouter",
            "openrouter:google/gemma-3-27b-it": "openrouter",
            "openrouter:google/gemma-3-27b-it:free": "openrouter",
            "openrouter:google/gemma-3-4b-it": "openrouter",
            "openrouter:google/gemma-3-4b-it:free": "openrouter",
            "openrouter:google/gemma-3n-e4b-it": "openrouter",
            "openrouter:google/gemma-3n-e4b-it:free": "openrouter",
            "openrouter:inception/mercury-coder": "openrouter",
            "openrouter:inflection/inflection-3-pi": "openrouter",
            "openrouter:inflection/inflection-3-productivity": "openrouter",
            "openrouter:meta-llama/llama-3.2-11b-vision-instruct": "openrouter",
            "openrouter:meta-llama/llama-3.2-1b-instruct": "openrouter",
            "openrouter:meta-llama/llama-3.2-3b-instruct": "openrouter",
            "openrouter:meta-llama/llama-3.2-3b-instruct:free": "openrouter",
            "openrouter:meta-llama/llama-3.2-90b-vision-instruct": "openrouter",
            "openrouter:meta-llama/llama-3.3-70b-instruct": "openrouter",
            "openrouter:meta-llama/llama-3.3-70b-instruct:free": "openrouter",
            "openrouter:meta-llama/llama-4-maverick": "openrouter",
            "openrouter:meta-llama/llama-4-scout": "openrouter",
            "openrouter:meta-llama/llama-guard-3-8b": "openrouter",
            "openrouter:microsoft/mai-ds-r1": "openrouter",
            "openrouter:microsoft/mai-ds-r1:free": "openrouter",
            "openrouter:microsoft/phi-4-multimodal-instruct": "openrouter",
            "openrouter:microsoft/phi-4-reasoning-plus": "openrouter",
            "openrouter:mistralai/devstral-small-2505": "openrouter",
            "openrouter:mistralai/ministral-3b": "openrouter",
            "openrouter:mistralai/ministral-8b": "openrouter",
            "openrouter:mistralai/mistral-7b-instruct-v0.1": "openrouter",
            "openrouter:mistralai/mistral-large-2407": "openrouter",
            "openrouter:mistralai/mistral-large-2411": "openrouter",
            "openrouter:mistralai/mistral-medium-3": "openrouter",
            "openrouter:mistralai/mistral-small-3.1-24b-instruct": "openrouter",
            "openrouter:mistralai/mistral-small-3.1-24b-instruct:free": "openrouter",
            "openrouter:mistralai/pixtral-12b": "openrouter",
            "openrouter:mistralai/pixtral-large-2411": "openrouter",
            "openrouter:neversleep/llama-3.1-lumimaid-8b": "openrouter",
            "openrouter:nousresearch/deephermes-3-mistral-24b-preview": "openrouter",
            "openrouter:nvidia/llama-3.1-nemotron-70b-instruct": "openrouter",
            "openrouter:nvidia/llama-3.1-nemotron-ultra-253b-v1": "openrouter",
            "openrouter:openai/codex-mini": "openrouter",
            "openrouter:openai/gpt-4.1": "openrouter",
            "openrouter:openai/gpt-4.1-mini": "openrouter",
            "openrouter:openai/gpt-4.1-nano": "openrouter",
            "openrouter:openai/gpt-4o-2024-08-06": "openrouter",
            "openrouter:openai/gpt-4o-mini-search-preview": "openrouter",
            "openrouter:openai/gpt-4o-search-preview": "openrouter",
            "openrouter:openai/o1-pro": "openrouter",
            "openrouter:openai/o3": "openrouter",
            "openrouter:openai/o4-mini": "openrouter",
            "openrouter:openai/o4-mini-high": "openrouter",
            "openrouter:perplexity/sonar-deep-research": "openrouter",
            "openrouter:perplexity/sonar-pro": "openrouter",
            "openrouter:perplexity/sonar-reasoning-pro": "openrouter",
            "openrouter:qwen/qwen-2.5-72b-instruct": "openrouter",
            "openrouter:qwen/qwen-2.5-72b-instruct:free": "openrouter",
            "openrouter:qwen/qwen-2.5-7b-instruct": "openrouter",
            "openrouter:qwen/qwen-2.5-coder-32b-instruct": "openrouter",
            "openrouter:qwen/qwen-2.5-coder-32b-instruct:free": "openrouter",
            "openrouter:qwen/qwen-2.5-vl-7b-instruct": "openrouter",
            "openrouter:qwen/qwen2.5-coder-7b-instruct": "openrouter",
            "openrouter:qwen/qwen2.5-vl-32b-instruct": "openrouter",
            "openrouter:qwen/qwen2.5-vl-32b-instruct:free": "openrouter",
            "openrouter:qwen/qwen3-14b": "openrouter",
            "openrouter:qwen/qwen3-14b:free": "openrouter",
            "openrouter:qwen/qwen3-235b-a22b": "openrouter",
            "openrouter:qwen/qwen3-235b-a22b:free": "openrouter",
            "openrouter:qwen/qwen3-30b-a3b": "openrouter",
            "openrouter:qwen/qwen3-30b-a3b:free": "openrouter",
            "openrouter:qwen/qwen3-32b": "openrouter",
            "openrouter:qwen/qwen3-4b:free": "openrouter",
            "openrouter:qwen/qwen3-8b": "openrouter",
            "openrouter:qwen/qwq-32b": "openrouter",
            "openrouter:raifle/sorcererlm-8x22b": "openrouter",
            "openrouter:thedrummer/rocinante-12b": "openrouter",
            "openrouter:thedrummer/skyfall-36b-v2": "openrouter",
            "openrouter:thedrummer/unslopnemo-12b": "openrouter",
            "openrouter:tngtech/deepseek-r1t-chimera": "openrouter",
            "openrouter:tngtech/deepseek-r1t-chimera:free": "openrouter",
            "openrouter:x-ai/grok-3-beta": "openrouter",
            "openrouter:x-ai/grok-3-mini-beta": "openrouter",
            "openrouter:z-ai/glm-4.5v": "openrouter",
            # Additional TogetherAI models
            "togetherai:ByteDance/Seedance-1.0-lite": "together-ai",
            "togetherai:Meta-Llama/Llama-Guard-7b": "together-ai",
            "togetherai:kwaivgI/kling-2.0-master": "together-ai",
            "togetherai:meta-llama/Llama-Guard-3-11B-Vision-Turbo": "together-ai",
            "togetherai:openai/sora-2-pro": "together-ai",
            "togetherai:pixverse/pixverse-v5": "together-ai",
            "togetherai:stabilityai/stable-diffusion-xl-base-1.0": "together-ai",
        }
        # For these models, per puter.js docs, temperature must be 1 by default
        # and should be enforced regardless of user-provided options.
        self.force_temperature_1_models = {
            "gpt-5-2025-08-07",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
            "gpt-5.1",
            "gpt-5.1-codex",
            "gpt-5.1-codex-mini",
            "gpt-5.1-chat-latest",
            "o1",
            "o1-mini",
            "o1-pro",
            "o3",
            "o3-mini",
            "o4-mini",
        }
        self.fallback_models = [
            "gpt-5-nano",
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
            response = requests.post(
                self.login_url, headers=self.headers, json=payload
            )
            response.raise_for_status()
            data = response.json()
            if data.get("proceed"):
                self.token = data["token"]
                logger.info("Login successful, token acquired")
                return self.token
            else:
                logger.warning("Login failed: Invalid credentials")
                raise ValueError("Login failed. Please check your credentials.")
        except requests.RequestException as e:
            logger.warning(f"Login error: {e}")
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

    def fs_write(
        self, path: str, content: Union[str, bytes, Any]
    ) -> Dict[str, Any]:
        """
        Write content to a file in Puter FS.
        If content is str or bytes, send directly. If file-like, read it.
        Returns the file info.
        """
        path = sanitize_string(path, allow_path=True)
        headers = self._get_auth_headers()
        headers.pop("Content-Type")
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, bytes):
            if hasattr(content, "read"):
                content = content.read()
            else:
                logger.warning("Invalid content type for fs_write")
                raise ValueError(
                    "Content must be str, bytes, or file-like object."
                )
        try:
            response = requests.post(
                f"{self.api_base}/write",
                params={"path": path},
                data=content,
                headers=headers,
            )
            response.raise_for_status()
            logger.info(f"File written successfully at {path}")
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"fs_write error: {e}")
            raise

    def fs_read(self, path: str) -> bytes:
        """
        Read file content from Puter FS.
        """
        path = sanitize_string(path, allow_path=True)
        headers = self._get_auth_headers()
        try:
            response = requests.get(
                f"{self.api_base}/read",
                params={"path": path},
                headers=headers,
            )
            response.raise_for_status()
            logger.info(f"File read successfully from {path}")
            return response.content
        except requests.RequestException as e:
            logger.warning(f"fs_read error: {e}")
            raise

    def fs_delete(self, path: str) -> None:
        """
        Delete a file or directory in Puter FS.
        """
        path = sanitize_string(path, allow_path=True)
        headers = self._get_auth_headers()
        try:
            response = requests.post(
                f"{self.api_base}/delete",
                params={"path": path},
                headers=headers,
            )
            response.raise_for_status()
            logger.info(f"File deleted successfully at {path}")
        except requests.RequestException as e:
            logger.warning(f"fs_delete error: {e}")
            raise

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
        """
        AI chat completion, supporting multiple models with fallback and retries.
        Returns: For stream, a generator of (content, used_model) tuples; for non-stream, a dict with response and used_model.
        """
        if options is None:
            options = {}
        model = options.get("model", self.fallback_models[0])
        driver = self.model_to_driver.get(model, "openai-completion")
        stream = options.get("stream", False)
        # Default temperature, overridden for specific models below
        temperature = options.get("temperature", 0.7)
        if model in self.force_temperature_1_models:
            if temperature != 1:
                logger.info(
                    f"Overriding temperature to 1 for model '{model}' as per puter.js requirements"
                )
            temperature = 1
        max_tokens = options.get("max_tokens", 1000)

        if messages is None:
            messages = []
            if prompt:
                content = prompt
                if image_url:
                    if not isinstance(image_url, list):
                        image_url = [image_url]
                    content_parts = (
                        [{"type": "text", "text": content}]
                        if isinstance(content, str)
                        else content
                    )
                    for url in image_url:
                        safe_url = sanitize_url(url)
                        content_parts.append(
                            {"type": "image_url", "image_url": {"url": safe_url}}
                        )
                    content = content_parts
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

        headers = self._get_auth_headers()
        try:
            logger.info(
                f"Sending ai_chat request with model {model}, "
                f"driver {driver}, stream={stream}, "
                f"test_mode={test_mode}, retry={retry_count}"
            )
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            response = requests.post(
                f"{self.api_base}/drivers/call",
                json=payload,
                headers=headers,
                stream=stream,
            )
            response.raise_for_status()

            def check_used_model(
                data: Dict[str, Any], requested_model: str, strict: bool
            ) -> str:
                """Extract used model from response metadata and validate."""
                used_model = None
                if (
                    "result" in data
                    and "usage" in data["result"]
                    and data["result"]["usage"]
                ):
                    used_model = data["result"]["usage"][0].get("model", "unknown")
                elif "metadata" in data and "service_used" in data["metadata"]:
                    used_model = data["metadata"].get("service_used", "unknown")
                if used_model and used_model != requested_model:
                    msg = (
                        f"Requested model {requested_model}, "
                        f"but server used {used_model}"
                    )
                    if strict:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
                return used_model or requested_model

            def process_line(
                line: bytes, requested_model: str, strict: bool
            ) -> Optional[tuple[str, str]]:
                if not line:
                    return None
                logger.debug(f"Raw stream line: {line}")
                try:
                    data = json.loads(line)
                    if not data.get("success", True):
                        error_data = data.get("error", {})
                        error_code = error_data.get("code")
                        error_msg = error_data.get("message", "Unknown error")
                        raise ValueError(
                            f"API error: {error_msg} (code: {error_code})"
                        )
                    used_model = check_used_model(data, requested_model, strict)
                    # Handle custom text response format
                    if "type" in data and data["type"] == "text" and "text" in data:
                        content = data["text"]
                        if content:
                            logger.info("Processed custom text JSON streaming response")
                            return content, used_model
                    # Handle Claude-style response
                    if "result" in data and "message" in data["result"]:
                        message_content = data["result"]["message"].get(
                            "content", ""
                        )
                        if isinstance(message_content, str):
                            logger.info(
                                "Processed string content JSON streaming response"
                            )
                            return message_content, used_model
                        elif isinstance(message_content, list):
                            content = message_content[0].get("text", "")
                            if content:
                                logger.info(
                                    "Processed list content JSON streaming response"
                                )
                                return content, used_model
                            else:
                                logger.warning(
                                    f"No text in list content: {message_content}"
                                )
                        else:
                            logger.warning(
                                f"Unexpected content type: {type(message_content)}"
                            )
                    # Handle OpenAI-style response
                    if "choices" in data:
                        content = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            logger.info("Processed OpenAI-style JSON streaming response")
                            return content, used_model
                        else:
                            logger.warning(f"No content in OpenAI-style response: {data}")
                    logger.warning(f"Unexpected JSON response format: {data}")
                except json.JSONDecodeError:
                    # Handle SSE format
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if not data.get("success", True):
                                error_data = data.get("error", {})
                                error_code = error_data.get("code")
                                error_msg = data.get("message", "Unknown error")
                                raise ValueError(
                                    f"API error: {error_msg} (code: {error_code})"
                                )
                            used_model = check_used_model(data, requested_model, strict)
                            content = data.get("text", "")
                            if content:
                                logger.info("Processed SSE streaming response")
                                return content, used_model
                            else:
                                logger.warning(f"No text in SSE data: {data}")
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid SSE data: {line}")
                    else:
                        # Treat as plain text
                        decoded_line = line.decode("utf-8", errors="ignore")
                        if decoded_line.strip():
                            logger.info("Processed plain text streaming response")
                            return decoded_line, requested_model
                        else:
                            logger.warning(f"Empty or invalid stream line: {line}")
                return None

            if stream:
                line_iter = response.iter_lines()
                # Peek at the first line to check for errors
                first_line = next(line_iter, None)
                if first_line:
                    try:
                        # Process first line to check for error
                        processed = process_line(first_line, model, strict_model)
                        if processed is None:
                            # If no content, but no raise, continue
                            pass
                    except ValueError as e:
                        error_msg = str(e)
                        if "API error" in error_msg:
                            # Extract code if possible
                            if "(code: " in error_msg:
                                error_code = error_msg.split("(code: ")[1][:-1]
                            else:
                                error_code = None
                            if error_code in (
                                "no_implementation_available",
                                "forbidden",
                            ) and retry_count < self.max_retries:
                                if strict_model:
                                    raise ValueError(
                                        f"Model {model} not available due to "
                                        "permission issues or implementation. "
                                        "Since strict_model is True, no fallback."
                                    )
                                else:
                                    if not test_mode:
                                        logger.warning(
                                            f"Model {model} not available, "
                                            f"retrying with test_mode=True, "
                                            f"attempt {retry_count + 1}"
                                        )
                                        return self.ai_chat(
                                            prompt=prompt,
                                            options=options,
                                            test_mode=True,
                                            image_url=image_url,
                                            messages=messages,
                                            retry_count=retry_count + 1,
                                            strict_model=strict_model,
                                        )
                                    # Fallback to another model
                                    current_model_index = (
                                        self.fallback_models.index(model)
                                        if model in self.fallback_models
                                        else -1
                                    )
                                    next_model_index = (
                                        current_model_index + 1
                                        if current_model_index >= 0
                                        else 0
                                    )
                                    if next_model_index < len(self.fallback_models):
                                        next_model = self.fallback_models[
                                            next_model_index
                                        ]
                                        logger.warning(
                                            f"Model {model} (driver {driver}) "
                                            f"not available, retrying with "
                                            f"model {next_model}, "
                                            f"attempt {retry_count + 1}"
                                        )
                                        options["model"] = next_model
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
                                    else:
                                        logger.error(
                                            f"No more fallback models available "
                                            f"after {model}."
                                        )
                                        raise
                            else:
                                raise
                        else:
                            raise
                # If no error, create generator that yields processed first and then others
                def generator():
                    if first_line:
                        processed = process_line(first_line, model, strict_model)
                        if processed:
                            yield processed
                    for line in line_iter:
                        processed = process_line(line, model, strict_model)
                        if processed:
                            yield processed

                logger.info("ai_chat stream request initiated")
                return generator()
            else:
                result = response.json()
                if not result.get("success", True):
                    error_data = result.get("error", {})
                    error_code = error_data.get("code")
                    error_msg = error_data.get("message", "Unknown error")
                    if error_code in (
                        "no_implementation_available",
                        "forbidden",
                    ) and retry_count < self.max_retries:
                        if strict_model:
                            raise ValueError(
                                f"Model {model} not available due to "
                                "permission issues or implementation. "
                                "Since strict_model is True, no fallback."
                            )
                        else:
                            if not test_mode:
                                logger.warning(
                                    f"Model {model} not available, "
                                    f"retrying with test_mode=True, "
                                    f"attempt {retry_count + 1}"
                                )
                                return self.ai_chat(
                                    prompt=prompt,
                                    options=options,
                                    test_mode=True,
                                    image_url=image_url,
                                    messages=messages,
                                    retry_count=retry_count + 1,
                                    strict_model=strict_model,
                                )
                            # Fallback to another model
                            current_model_index = (
                                self.fallback_models.index(model)
                                if model in self.fallback_models
                                else -1
                            )
                            next_model_index = (
                                current_model_index + 1
                                if current_model_index >= 0
                                else 0
                            )
                            if next_model_index < len(self.fallback_models):
                                next_model = self.fallback_models[next_model_index]
                                logger.warning(
                                    f"Model {model} (driver {driver}) "
                                    f"not available, retrying with "
                                    f"model {next_model}, "
                                    f"attempt {retry_count + 1}"
                                )
                                options["model"] = next_model
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
                            else:
                                logger.error(
                                    f"No more fallback models available "
                                    f"after {model}."
                                )
                                raise ValueError(
                                    f"No implementation available for "
                                    f"model {model}: {error_msg}"
                                )
                    else:
                        logger.error(f"AI chat error: {error_msg}")
                        raise ValueError(f"AI chat error: {error_msg}")
                used_model = check_used_model(result, model, strict_model)
                logger.info(f"ai_chat request successful, used model: {used_model}")
                return {"response": result, "used_model": used_model}

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
                safe_url = sanitize_url(image)
                payload = {"image_url": safe_url, "testMode": test_mode}
                response = requests.post(
                    f"{self.api_base}/ai/img2txt",
                    json=payload,
                    headers=headers,
                )
            else:
                files = {"image": image}
                response = requests.post(
                    f"{self.api_base}/ai/img2txt",
                    files=files,
                    data={"testMode": test_mode},
                    headers=headers,
                )
            response.raise_for_status()
            logger.info("ai_img2txt request successful")
            return response.json().get("text")
        except requests.RequestException as e:
            logger.warning(f"ai_img2txt error: {e}")
            raise

    def ai_txt2img(self, prompt: str, model: str = "pollinations-image", test_mode: bool = False) -> str:
        """
        Text to image using Puter's driver API.
        Returns image URL.
        """
        payload = {
            "interface": "puter-image-generation",
            "driver": model,  # or other supported image model in puter.js docs
            "method": "generate",
            "args": {
                "prompt": prompt
            },
            "testMode": test_mode
        }

        headers = self._get_auth_headers()
        try:
            response = requests.post(f"{self.api_base}/drivers/call", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if "result" in data and "image_url" in data["result"]:
                return data["result"]["image_url"]
            elif "result" in data and isinstance(data["result"], dict):
                # Some models might return base64 or direct data URL
                return data["result"].get("url") or data["result"].get("data")
            else:
                raise ValueError(f"Unexpected response format: {data}")
        except requests.RequestException as e:
            logger.warning(f"ai_txt2img error: {e}")
            raise

    def ai_txt2speech(
        self, text: str, options: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Text to speech.
        Returns MP3 bytes.
        """
        if options is None:
            options = {}
        # Allow natural language text
        payload = {"text": text, "testMode": options.get("testMode", False)}
        headers = self._get_auth_headers()
        try:
            response = requests.post(
                f"{self.api_base}/ai/txt2speech",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            logger.info("ai_txt2speech request successful")
            return response.content
        except requests.RequestException as e:
            logger.warning(f"ai_txt2speech error: {e}")
            raise
