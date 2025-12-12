import asyncio
import json
import logging
import socket
import ssl
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import certifi
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = "2.1.0"
