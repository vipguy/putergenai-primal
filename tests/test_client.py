
import pytest

from putergenai import PuterClient


class TestPuterClientInit:
    """Test PuterClient initialization and basic properties."""

    def test_client_init_default(self):
        """Test client initialization with default parameters."""
        client = PuterClient()

        assert client.token is None
        assert client.ignore_ssl is False
        assert client.auto_update_models is False
        assert client.api_base == "https://api.puter.com"
        assert client.login_url == "https://puter.com/login"
        assert client._session is None
        assert isinstance(client.model_to_driver, dict)
        assert len(client.model_to_driver) > 0  # Should have predefined models

    def test_client_init_with_token(self):
        """Test client initialization with token."""
        client = PuterClient(token="test_token")

        assert client.token == "test_token"

    def test_client_init_with_ssl_ignore(self):
        """Test client initialization with ignore_ssl=True."""
        client = PuterClient(ignore_ssl=True)

        assert client.ignore_ssl is True

    def test_client_init_with_auto_update(self):
        """Test client initialization with auto_update_models=True."""
        client = PuterClient(auto_update_models=True)

        assert client.auto_update_models is True

    def test_client_init_all_params(self):
        """Test client initialization with all parameters."""
        client = PuterClient(token="test_token", ignore_ssl=True, auto_update_models=True)

        assert client.token == "test_token"
        assert client.ignore_ssl is True
        assert client.auto_update_models is True

    def test_model_mappings_exist(self):
        """Test that model_to_driver mappings are properly initialized."""
        client = PuterClient()

        # Check some known models
        assert "gpt-4o" in client.model_to_driver
        assert "claude-3-5-sonnet-latest" in client.model_to_driver
        assert "mistral-large-latest" in client.model_to_driver
        assert "grok-beta" in client.model_to_driver

    def test_force_temperature_models(self):
        """Test force_temperature_1_models set."""
        client = PuterClient()

        expected_models = {
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

        assert client.force_temperature_1_models == expected_models

    def test_fallback_models(self):
        """Test fallback_models list."""
        client = PuterClient()

        expected_fallbacks = [
            "gpt-5-nano",
            "gpt-4.1-nano",
            "gpt-4o-mini",
            "claude-3-5-sonnet-latest",
            "deepseek-chat",
        ]

        assert client.fallback_models == expected_fallbacks

    def test_max_retries(self):
        """Test max_retries setting."""
        client = PuterClient()

        assert client.max_retries == 3

    def test_cache_settings(self):
        """Test cache initialization."""
        client = PuterClient()

        assert client._models_cache is None
        assert client._cache_timestamp is None
        assert client._cache_ttl.seconds == 3600  # 1 hour

    def test_headers_structure(self):
        """Test headers structure."""
        client = PuterClient()

        required_headers = [
            "Accept",
            "Accept-Language",
            "Connection",
            "Origin",
            "Referer",
            "User-Agent",
        ]

        for header in required_headers:
            assert header in client.headers
            assert client.headers[header] is not None

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, client, mock_client_session):
        """Test async context manager __aenter__."""
        async with client:
            assert client._session is not None

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, client, mock_client_session):
        """Test async context manager __aexit__."""
        async with client:
            pass

        # Session should be closed
        mock_client_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_method(self, client, mock_client_session):
        """Test close method."""
        client._session = mock_client_session
        await client.close()

        mock_client_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_session(self, client):
        """Test close method when no session exists."""
        await client.close()  # Should not raise

    def test_get_auth_headers_with_token(self, client):
        """Test _get_auth_headers with token."""
        headers = client._get_auth_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"
        assert headers["Content-Type"] == "application/json"

    def test_get_auth_headers_no_token(self, client_no_token):
        """Test _get_auth_headers without token raises error."""
        with pytest.raises(ValueError, match="Not authenticated"):
            client_no_token._get_auth_headers()
