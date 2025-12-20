import pytest


class TestAIChat:
    """Test AI chat functionality."""

    @pytest.mark.asyncio
    async def test_ai_chat_non_streaming_success(
        self, client, mock_client_session, mock_response, sample_chat_response
    ):
        """Test successful non-streaming AI chat."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_chat_response

        # Make chat request with explicit model
        result = await client.ai_chat(prompt="Hello", options={"model": "gpt-4o"})

        # Assertions
        assert "response" in result
        assert "used_model" in result
        assert result["used_model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_ai_chat_with_messages(
        self, client, mock_client_session, mock_response, sample_chat_response
    ):
        """Test AI chat with explicit messages."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_chat_response

        messages = [{"role": "user", "content": "Hello"}]
        result = await client.ai_chat(messages=messages)

        # Assertions
        assert result is not None
        # Verify messages were passed correctly
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["args"]["messages"] == messages

    @pytest.mark.asyncio
    async def test_ai_chat_with_options(
        self, client, mock_client_session, mock_response, sample_chat_response
    ):
        """Test AI chat with custom options."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_chat_response

        options = {"model": "gpt-5", "temperature": 1, "max_tokens": 100, "stream": False}

        result = await client.ai_chat(prompt="Hello", options=options)

        # Verify options were passed
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["driver"] == "openai-completion"  # gpt-5 driver
        assert payload["args"]["model"] == "gpt-5"
        assert payload["args"]["temperature"] == 1
        assert payload["args"]["max_tokens"] == 100
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_ai_chat_force_temperature_1(
        self, client, mock_client_session, mock_response, sample_chat_response
    ):
        """Test that certain models force temperature to 1."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_chat_response

        # Use a model that forces temperature 1
        options = {"model": "o1", "temperature": 0.7}

        result = await client.ai_chat(prompt="Hello", options=options)

        # Verify temperature was forced to 1
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["args"]["temperature"] == 1

    @pytest.mark.asyncio
    async def test_ai_chat_with_image_url(
        self, client, mock_client_session, mock_response, sample_chat_response
    ):
        """Test AI chat with image URL."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_chat_response

        image_url = "https://example.com/image.jpg"
        result = await client.ai_chat(prompt="Describe this image", image_url=image_url)

        # Verify image was processed
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        messages = payload["args"]["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # text + image
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_ai_chat_with_multiple_images(
        self, client, mock_client_session, mock_response, sample_chat_response
    ):
        """Test AI chat with multiple image URLs."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_chat_response

        image_urls = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
        result = await client.ai_chat(prompt="Compare these images", image_url=image_urls)

        # Verify images were processed
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        messages = payload["args"]["messages"]
        content = messages[0]["content"]
        assert len(content) == 3  # text + 2 images

    @pytest.mark.asyncio
    async def test_ai_chat_invalid_image_url(self, client):
        """Test AI chat with invalid image URL."""
        with pytest.raises(ValueError):  # URL validation should fail
            await client.ai_chat(prompt="Hello", image_url="not-a-url")

    @pytest.mark.asyncio
    async def test_ai_chat_fallback_on_error(self, client, mock_client_session, mock_response):
        """Test model fallback on error."""
        # Setup mock to return error response
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {
            "success": False,
            "error": {"message": "Model unavailable", "code": "no_implementation_available"},
        }

        # This should trigger fallback to next model
        # Since we don't have a second mock, it should eventually raise
        with pytest.raises(ValueError):
            await client.ai_chat(prompt="Hello", options={"model": "gpt-5-nano"})

    @pytest.mark.asyncio
    async def test_ai_chat_strict_model_error(self, client, mock_client_session, mock_response):
        """Test strict model mode prevents fallback."""
        # Setup mock to return error response
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {
            "success": False,
            "error": {"message": "Model unavailable", "code": "forbidden"},
        }

        # With strict_model=True, should raise immediately
        with pytest.raises(ValueError, match="Model .* unavailable"):
            await client.ai_chat(prompt="Hello", options={"model": "gpt-5-nano"}, strict_model=True)

    @pytest.mark.asyncio
    async def test_ai_chat_with_error_response(self, client, mock_client_session, mock_response):
        """Test AI chat with error response."""
        # Setup mock with error response
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {
            "success": False,
            "error": {"message": "Test error", "code": "test_error"},
        }

        # Should raise ValueError
        with pytest.raises(ValueError, match="AI chat error: Test error"):
            await client.ai_chat(prompt="Hello")

    @pytest.mark.asyncio
    async def test_ai_chat_no_auth(self, client_no_token, mock_client_session):
        """Test that AI chat requires authentication."""
        with pytest.raises(ValueError, match="Not authenticated"):
            await client_no_token.ai_chat(prompt="Hello")

    @pytest.mark.asyncio
    async def test_ai_chat_network_error(self, client, mock_client_session):
        """Test AI chat with network error."""
        from aiohttp import ClientError

        mock_client_session.post.side_effect = ClientError("Network error")

        with pytest.raises(ClientError):
            await client.ai_chat(prompt="Hello")

    @pytest.mark.asyncio
    async def test_ai_chat_invalid_response_format(
        self, client, mock_client_session, mock_response
    ):
        """Test AI chat with invalid response format."""
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"invalid": "response"}

        # This should not raise an error, just return the response
        result = await client.ai_chat(prompt="Hello")
        assert "response" in result
