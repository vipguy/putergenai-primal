from unittest.mock import AsyncMock

import pytest


class TestAIUtilities:
    """Test AI utility functions."""

    @pytest.mark.asyncio
    async def test_ai_img2txt_with_url(self, client, mock_client_session, mock_response):
        """Test ai_img2txt with image URL."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"text": "Extracted text from image"}

        # Call function
        result = await client.ai_img2txt("https://example.com/image.jpg")

        # Assertions
        assert result == "Extracted text from image"

        # Verify request
        mock_client_session.post.assert_called_once()
        call_args = mock_client_session.post.call_args
        assert call_args[0][0] == "https://api.puter.com/ai/img2txt"
        payload = call_args[1]["json"]
        assert str(payload["image_url"]) == "https://example.com/image.jpg"
        assert payload["testMode"] is False

    @pytest.mark.asyncio
    async def test_ai_img2txt_with_test_mode(self, client, mock_client_session, mock_response):
        """Test ai_img2txt with test mode enabled."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"text": "Test mode text"}

        # Call function with test mode
        result = await client.ai_img2txt("https://example.com/image.jpg", test_mode=True)

        # Verify testMode was set
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["testMode"] is True

    @pytest.mark.asyncio
    async def test_ai_img2txt_with_file_object(self, client, mock_client_session, mock_response):
        """Test ai_img2txt with file object."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"text": "File text"}

        # Mock file object
        file_obj = AsyncMock()
        file_obj.read.return_value = b"fake image data"

        # Call function
        result = await client.ai_img2txt(file_obj)

        # Verify it used FormData approach
        call_args = mock_client_session.post.call_args
        assert "data" in call_args[1]
        # Content-Type should be removed from headers
        assert "Content-Type" not in call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_ai_img2txt_invalid_url(self, client):
        """Test ai_img2txt with invalid URL."""
        with pytest.raises(ValueError):  # URL validation should fail
            await client.ai_img2txt("not-a-url")

    @pytest.mark.asyncio
    async def test_ai_img2txt_network_error(self, client, mock_client_session):
        """Test ai_img2txt with network error."""
        from aiohttp import ClientError

        mock_client_session.post.side_effect = ClientError("Network error")

        with pytest.raises(ClientError):
            await client.ai_img2txt("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_ai_img2txt_no_auth(self, client_no_token):
        """Test that ai_img2txt requires authentication."""
        with pytest.raises(ValueError, match="Not authenticated"):
            await client_no_token.ai_img2txt("https://example.com/image.jpg")

    @pytest.mark.asyncio
    async def test_ai_txt2img_success(self, client, mock_client_session, mock_response):
        """Test ai_txt2img successful operation."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {
            "result": {"image_url": "https://example.com/generated.jpg"}
        }

        # Call function
        result = await client.ai_txt2img("A beautiful sunset")

        # Assertions
        assert result == "https://example.com/generated.jpg"

        # Verify request
        mock_client_session.post.assert_called_once()
        call_args = mock_client_session.post.call_args
        assert call_args[0][0] == "https://api.puter.com/drivers/call"
        payload = call_args[1]["json"]
        assert payload["interface"] == "puter-image-generation"
        assert payload["method"] == "generate"
        assert payload["args"]["prompt"] == "A beautiful sunset"
        assert payload["args"]["testMode"] is False

    @pytest.mark.asyncio
    async def test_ai_txt2img_with_custom_model(self, client, mock_client_session, mock_response):
        """Test ai_txt2img with custom model."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"result": {"url": "https://example.com/image.png"}}

        # Call function with custom model
        result = await client.ai_txt2img("A cat", model="custom-model")

        # Verify model was used
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["driver"] == "custom-model"

    @pytest.mark.asyncio
    async def test_ai_txt2img_with_test_mode(self, client, mock_client_session, mock_response):
        """Test ai_txt2img with test mode."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"result": {"data": "base64data"}}

        # Call function with test mode
        result = await client.ai_txt2img("Test prompt", test_mode=True)

        # Verify testMode was set
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["args"]["testMode"] is True

    @pytest.mark.asyncio
    async def test_ai_txt2img_alternative_response_format(
        self, client, mock_client_session, mock_response
    ):
        """Test ai_txt2img with alternative response format."""
        # Setup mocks with url in result
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"result": {"url": "https://example.com/img.jpg"}}

        result = await client.ai_txt2img("Test")

        assert result == "https://example.com/img.jpg"

    @pytest.mark.asyncio
    async def test_ai_txt2img_data_response_format(
        self, client, mock_client_session, mock_response
    ):
        """Test ai_txt2img with data response format."""
        # Setup mocks with data in result
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"result": {"data": "image_data"}}

        result = await client.ai_txt2img("Test")

        assert result == "image_data"

    @pytest.mark.asyncio
    async def test_ai_txt2img_invalid_response(self, client, mock_client_session, mock_response):
        """Test ai_txt2img with invalid response format."""
        # Setup mocks with invalid response
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"invalid": "response"}

        with pytest.raises(ValueError, match="Unexpected response format"):
            await client.ai_txt2img("Test prompt")

    @pytest.mark.asyncio
    async def test_ai_txt2img_no_auth(self, client_no_token):
        """Test that ai_txt2img requires authentication."""
        with pytest.raises(ValueError, match="Not authenticated"):
            await client_no_token.ai_txt2img("Test prompt")

    @pytest.mark.asyncio
    async def test_ai_txt2speech_success(self, client, mock_client_session, mock_response):
        """Test ai_txt2speech successful operation."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.read.return_value = b"fake_audio_data"

        # Call function
        result = await client.ai_txt2speech("Hello world")

        # Assertions
        assert result == b"fake_audio_data"

        # Verify request
        mock_client_session.post.assert_called_once()
        call_args = mock_client_session.post.call_args
        assert call_args[0][0] == "https://api.puter.com/ai/txt2speech"
        payload = call_args[1]["json"]
        assert payload["text"] == "Hello world"
        assert payload["testMode"] is False

    @pytest.mark.asyncio
    async def test_ai_txt2speech_with_options(self, client, mock_client_session, mock_response):
        """Test ai_txt2speech with custom options."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.read.return_value = b"audio_data"

        # Call function with options
        options = {"testMode": True}
        result = await client.ai_txt2speech("Test text", options=options)

        # Verify options were used
        call_args = mock_client_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["testMode"] is True

    @pytest.mark.asyncio
    async def test_ai_txt2speech_no_auth(self, client_no_token):
        """Test that ai_txt2speech requires authentication."""
        with pytest.raises(ValueError, match="Not authenticated"):
            await client_no_token.ai_txt2speech("Test text")

    @pytest.mark.asyncio
    async def test_ai_txt2speech_network_error(self, client, mock_client_session):
        """Test ai_txt2speech with network error."""
        from aiohttp import ClientError

        mock_client_session.post.side_effect = ClientError("Network error")

        with pytest.raises(ClientError):
            await client.ai_txt2speech("Test text")

    # Test update_model_mappings method
    @pytest.mark.asyncio
    async def test_update_model_mappings(self, client, mock_client_session, mock_response):
        """Test update_model_mappings functionality."""
        # Setup mocks for get_available_models with simple string models
        simple_models_response = {"models": ["gpt-4o", "claude-3-sonnet", "mistral-large"]}
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = simple_models_response

        # Call update
        await client.update_model_mappings()

        # Verify mappings were updated
        assert "gpt-4o" in client.model_to_driver
        assert client.model_to_driver["gpt-4o"] == "openai-completion"
        assert "claude-3-sonnet" in client.model_to_driver
        assert client.model_to_driver["claude-3-sonnet"] == "claude"

    @pytest.mark.asyncio
    async def test_update_model_mappings_with_provider(
        self, client, mock_client_session, mock_response
    ):
        """Test update_model_mappings with provider inference."""
        models_data = {"models": ["test-claude-model"]}

        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = models_data

        # Call update
        await client.update_model_mappings()

        # Verify model was added to mapping
        assert "test-claude-model" in client.model_to_driver

    @pytest.mark.asyncio
    async def test_update_model_mappings_heuristic_fallback(
        self, client, mock_client_session, mock_response
    ):
        """Test update_model_mappings heuristic driver detection."""
        models_data = {
            "models": ["openrouter:gpt-4", "togetherai:llama-2", "claude-3-sonnet", "unknown-model"]
        }

        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = models_data

        # Call update
        await client.update_model_mappings()

        # Verify heuristics worked
        assert client.model_to_driver["openrouter:gpt-4"] == "openrouter"
        assert client.model_to_driver["togetherai:llama-2"] == "together-ai"
        assert client.model_to_driver["claude-3-sonnet"] == "claude"
        assert client.model_to_driver["unknown-model"] == "openai-completion"  # default
