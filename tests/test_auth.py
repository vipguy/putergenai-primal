import pytest
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class TestAuthentication:
    """Test authentication functionality."""

    @pytest.mark.asyncio
    async def test_login_success(
        self, client_no_token, mock_client_session, mock_response, sample_login_response
    ):
        """Test successful login."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_login_response

        # Perform login
        token = await client_no_token.login("testuser", "testpass")

        # Assertions
        assert token == "sample_token_123"
        assert client_no_token.token == "sample_token_123"

        # Verify the request was made correctly
        mock_client_session.post.assert_called_once()
        call_args = mock_client_session.post.call_args
        assert call_args[0][0] == "https://puter.com/login"
        assert call_args[1]["json"] == {"username": "testuser", "password": "testpass"}

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(
        self, client_no_token, mock_client_session, mock_response
    ):
        """Test login with invalid credentials."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"proceed": False}

        # Perform login - should raise ValueError
        with pytest.raises(ValueError, match="Login failed"):
            await client_no_token.login("testuser", "wrongpass")

        # Token should not be set
        assert client_no_token.token is None

    @pytest.mark.asyncio
    async def test_login_network_error(self, client_no_token, mock_client_session):
        """Test login with network error."""
        # Setup mock to raise ClientError
        from aiohttp import ClientError

        mock_client_session.post.side_effect = ClientError("Network error")

        # Perform login - should raise ValueError
        with pytest.raises(ValueError, match="Login error: Network error"):
            await client_no_token.login("testuser", "testpass")

        # Token should not be set
        assert client_no_token.token is None

    @pytest.mark.asyncio
    async def test_login_invalid_response_format(
        self, client_no_token, mock_client_session, mock_response
    ):
        """Test login with invalid response format."""
        # Setup mocks with invalid response
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"invalid": "response"}

        # Perform login - should raise ValueError
        with pytest.raises(ValueError, match="Login failed"):
            await client_no_token.login("testuser", "testpass")

    @pytest.mark.asyncio
    async def test_login_empty_username(self, client_no_token):
        """Test login with empty username."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await client_no_token.login("", "testpass")

    @pytest.mark.asyncio
    async def test_login_empty_password(self, client_no_token):
        """Test login with empty password."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await client_no_token.login("testuser", "")

    @pytest.mark.asyncio
    async def test_login_whitespace_username(self, client_no_token):
        """Test login with whitespace-only username."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await client_no_token.login("   ", "testpass")

    @pytest.mark.asyncio
    async def test_login_whitespace_password(self, client_no_token):
        """Test login with whitespace-only password."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            await client_no_token.login("testuser", "   ")

    @pytest.mark.asyncio
    async def test_login_with_env_credentials(
        self,
        client_no_token,
        mock_client_session,
        mock_response,
        sample_login_response,
        env_credentials,
    ):
        """Test login using credentials from .env file."""
        username = env_credentials["username"]
        password = env_credentials["password"]

        # Skip test if credentials not provided
        if not username or not password:
            pytest.skip("PUTER_USERNAME and PUTER_PASSWORD environment variables not set")

        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_login_response

        # Use the credentials from environment
        token = await client_no_token.login(username, password)

        # Assertions
        assert token == "sample_token_123"
        assert client_no_token.token == "sample_token_123"

        # Verify the request was made with correct credentials
        call_args = mock_client_session.post.call_args
        assert call_args[1]["json"] == {"username": username, "password": password}
