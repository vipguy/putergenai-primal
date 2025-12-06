import pytest
import os
import aiohttp
from unittest.mock import AsyncMock, Mock, patch
from dotenv import load_dotenv
from putergenai import PuterClient

# Load environment variables
load_dotenv()


@pytest.fixture
def mock_session():
    """Mock aiohttp ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.closed = False
    return session


@pytest.fixture
def mock_response():
    """Mock aiohttp ClientResponse."""
    response = AsyncMock(spec=aiohttp.ClientResponse)
    response.raise_for_status = Mock()
    response.json = AsyncMock()
    response.read = AsyncMock()
    response.content = AsyncMock()
    response.content.__aiter__ = AsyncMock(return_value=iter([]))
    return response


@pytest.fixture
def client():
    """Create PuterClient instance for testing."""
    return PuterClient(token="test_token")


@pytest.fixture
def client_no_token():
    """Create PuterClient instance without token."""
    return PuterClient()


@pytest.fixture
async def mock_client_session(mock_session):
    """Mock the client's session creation."""
    with patch('putergenai.putergenai.aiohttp.ClientSession', return_value=mock_session):
        with patch('putergenai.putergenai.ssl.create_default_context', return_value=True):
            with patch('putergenai.putergenai.certifi.where', return_value='dummy_cert_file'):
                yield mock_session


@pytest.fixture
def sample_login_response():
    """Sample successful login response."""
    return {
        "proceed": True,
        "token": "sample_token_123"
    }


@pytest.fixture
def sample_models_response():
    """Sample models API response."""
    return {
        "models": [
            "gpt-4o",
            "claude-3-5-sonnet-latest",
            "gpt-5"
        ]
    }


@pytest.fixture
def sample_chat_response():
    """Sample AI chat response."""
    return {
        "success": True,
        "result": {
            "message": {
                "content": "Hello, this is a test response!"
            }
        },
        "used_model": "gpt-4o"
    }


@pytest.fixture
def sample_streaming_data():
    """Sample streaming response data."""
    return [
        b'{"type": "text", "text": "Hello"}',
        b'{"type": "text", "text": " world"}',
        b'{"type": "text", "text": "!"}',
        b''
    ]


@pytest.fixture
def env_credentials():
    """Get credentials from environment variables."""
    username = os.getenv('PUTER_USERNAME')
    password = os.getenv('PUTER_PASSWORD')
    return {"username": username, "password": password}
