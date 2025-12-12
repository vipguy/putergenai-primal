import pytest


class TestModelsAPI:
    """Test models API functionality."""

    @pytest.mark.asyncio
    async def test_get_available_models_success(
        self, client, mock_client_session, mock_response, sample_models_response
    ):
        """Test successful retrieval of available models."""
        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_models_response

        # Get models
        result = await client.get_available_models()

        # Assertions
        assert "models" in result
        assert len(result["models"]) == 3
        assert result["models"][0] == "gpt-4o"
        assert result["models"][1] == "claude-3-5-sonnet-latest"
        assert result["models"][2] == "gpt-5"

        # Verify cache is set
        assert client._models_cache == result
        assert client._cache_timestamp is not None

    @pytest.mark.asyncio
    async def test_get_available_models_force_refresh(
        self, client, mock_client_session, mock_response, sample_models_response
    ):
        """Test get_available_models with force_refresh."""
        # Setup initial cache
        client._models_cache = {"models": ["cached_model"]}

        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_models_response

        # Get models with force refresh
        result = await client.get_available_models(force_refresh=True)

        # Should fetch new data and update cache
        assert result == sample_models_response
        assert client._models_cache == sample_models_response

    @pytest.mark.asyncio
    async def test_get_available_models_from_cache(
        self, client, mock_client_session, sample_models_response
    ):
        """Test get_available_models returns cached data."""
        # Setup cache
        client._models_cache = sample_models_response
        from datetime import datetime, timedelta

        client._cache_timestamp = datetime.now() - timedelta(minutes=30)  # Recent cache

        # Get models - should not make HTTP request
        result = await client.get_available_models()

        # Should return cached data
        assert result == sample_models_response
        mock_client_session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_available_models_expired_cache(
        self, client, mock_client_session, mock_response, sample_models_response
    ):
        """Test get_available_models refreshes expired cache."""
        # Setup expired cache
        client._models_cache = {"models": ["old_cached_model"]}
        from datetime import datetime, timedelta

        client._cache_timestamp = datetime.now() - timedelta(hours=2)  # Expired cache

        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_models_response

        # Get models - should refresh cache
        result = await client.get_available_models()

        # Should return new data
        assert result == sample_models_response
        mock_client_session.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_available_models_invalid_response(
        self, client, mock_client_session, mock_response
    ):
        """Test get_available_models with invalid response format."""
        # Setup mocks with invalid response
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = "invalid_response"

        # Should raise ValueError
        with pytest.raises(ValueError, match="Unexpected API response format"):
            await client.get_available_models()

    @pytest.mark.asyncio
    async def test_get_available_models_missing_models_key(
        self, client, mock_client_session, mock_response
    ):
        """Test get_available_models with missing models key."""
        # Setup mocks with response missing models key
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"other_key": "value"}

        # Should raise ValueError
        with pytest.raises(ValueError, match="API response is missing the 'models' key"):
            await client.get_available_models()

    @pytest.mark.asyncio
    async def test_get_available_models_non_list_models(
        self, client, mock_client_session, mock_response
    ):
        """Test get_available_models with non-list models value."""
        # Setup mocks with non-list models
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"models": "not_a_list"}

        # Should raise ValueError
        with pytest.raises(ValueError, match="The 'models' field must be a list"):
            await client.get_available_models()

    @pytest.mark.asyncio
    async def test_get_available_models_network_error_with_cache(
        self, client, mock_client_session, sample_models_response
    ):
        """Test get_available_models falls back to cache on network error."""
        # Setup cache
        client._models_cache = sample_models_response

        # Setup mock to raise error
        from aiohttp import ClientError

        mock_client_session.get.side_effect = ClientError("Network error")

        # Get models - should return cache
        result = await client.get_available_models()

        assert result == sample_models_response

    @pytest.mark.asyncio
    async def test_get_available_models_network_error_no_cache(self, client, mock_client_session):
        """Test get_available_models raises error when no cache and network fails."""
        # Setup mock to raise error
        from aiohttp import ClientError

        mock_client_session.get.side_effect = ClientError("Network error")

        # Should raise the original error
        with pytest.raises(ClientError):
            await client.get_available_models()

    @pytest.mark.asyncio
    async def test_is_model_available_string_model(
        self, client, mock_client_session, mock_response, sample_models_response
    ):
        """Test is_model_available with string model."""
        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_models_response

        result = await client.is_model_available("gpt-4o")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_model_available_dict_model(self, client, mock_client_session, mock_response):
        """Test is_model_available with dict model."""
        # Setup mocks with dict model
        models_response = {"models": [{"id": "gpt-5", "name": "GPT-5", "provider": "openai"}]}
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = models_response

        result = await client.is_model_available("gpt-5")

        assert result is True

    @pytest.mark.asyncio
    async def test_is_model_available_with_aliases(
        self, client, mock_client_session, mock_response
    ):
        """Test is_model_available with model aliases."""
        models_response = {"models": [{"id": "test-model", "aliases": ["alias1", "alias2"]}]}

        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = models_response

        result1 = await client.is_model_available("alias1")
        result2 = await client.is_model_available("alias2")
        result3 = await client.is_model_available("test-model")

        assert result1 is True
        assert result2 is True
        assert result3 is True

    @pytest.mark.asyncio
    async def test_is_model_available_not_found(
        self, client, mock_client_session, mock_response, sample_models_response
    ):
        """Test is_model_available with non-existent model."""
        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = sample_models_response

        result = await client.is_model_available("non-existent-model")

        assert result is False
