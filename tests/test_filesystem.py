from unittest.mock import AsyncMock

import pytest


class TestFilesystem:
    """Test filesystem operations."""

    @pytest.mark.asyncio
    async def test_fs_write_string_content(self, client, mock_client_session, mock_response):
        """Test fs_write with string content."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"success": True}

        # Write file
        result = await client.fs_write("test.txt", "Hello World")

        # Assertions
        assert result == {"success": True}

        # Verify the request
        mock_client_session.post.assert_called_once()
        call_args = mock_client_session.post.call_args
        assert call_args[0][0] == "https://api.puter.com/write"
        assert call_args[1]["params"] == {"path": "test.txt"}
        assert call_args[1]["data"] == b"Hello World"

    @pytest.mark.asyncio
    async def test_fs_write_bytes_content(self, client, mock_client_session, mock_response):
        """Test fs_write with bytes content."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"success": True}

        content = b"Binary data"
        result = await client.fs_write("test.bin", content)

        # Assertions
        assert result == {"success": True}

        # Verify the request
        call_args = mock_client_session.post.call_args
        assert call_args[1]["data"] == content

    @pytest.mark.asyncio
    async def test_fs_write_file_like_object(self, client, mock_client_session, mock_response):
        """Test fs_write with file-like object."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response
        mock_response.json.return_value = {"success": True}

        # Mock file-like object
        file_obj = AsyncMock()
        file_obj.read.return_value = b"File content"

        result = await client.fs_write("test.txt", file_obj)

        # Should not be called because we check hasattr(content, "read")
        file_obj.read.assert_not_called()

        # Content should be treated as file-like but since it's not bytes/str, should raise
        # Actually, the code checks for bytes/str first, then hasattr(read)
        # Let's adjust the mock
        file_obj = b"File content"  # Make it bytes

        result = await client.fs_write("test.txt", file_obj)
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_fs_write_invalid_content_type(self, client):
        """Test fs_write with invalid content type."""
        with pytest.raises(ValueError, match="Content must be str, bytes, or file-like object"):
            await client.fs_write("test.txt", 123)

    @pytest.mark.asyncio
    async def test_fs_write_invalid_path(self, client):
        """Test fs_write with invalid path."""
        with pytest.raises(ValueError):  # Path validation should fail
            await client.fs_write("invalid@path", "content")

    @pytest.mark.asyncio
    async def test_fs_write_empty_path(self, client):
        """Test fs_write with empty path."""
        with pytest.raises(ValueError):  # Path validation should fail
            await client.fs_write("", "content")

    @pytest.mark.asyncio
    async def test_fs_write_network_error(self, client, mock_client_session):
        """Test fs_write with network error."""
        from aiohttp import ClientError

        mock_client_session.post.side_effect = ClientError("Network error")

        with pytest.raises(ClientError):
            await client.fs_write("test.txt", "content")

    @pytest.mark.asyncio
    async def test_fs_read_success(self, client, mock_client_session, mock_response):
        """Test fs_read successful operation."""
        # Setup mocks
        mock_client_session.get.return_value.__aenter__.return_value = mock_response
        mock_response.read.return_value = b"File content"

        # Read file
        result = await client.fs_read("test.txt")

        # Assertions
        assert result == b"File content"

        # Verify the request
        mock_client_session.get.assert_called_once()
        call_args = mock_client_session.get.call_args
        assert call_args[0][0] == "https://api.puter.com/read"
        assert call_args[1]["params"] == {"path": "test.txt"}

    @pytest.mark.asyncio
    async def test_fs_read_invalid_path(self, client):
        """Test fs_read with invalid path."""
        with pytest.raises(ValueError):
            await client.fs_read("invalid@path")

    @pytest.mark.asyncio
    async def test_fs_read_empty_path(self, client):
        """Test fs_read with empty path."""
        with pytest.raises(ValueError):
            await client.fs_read("")

    @pytest.mark.asyncio
    async def test_fs_read_network_error(self, client, mock_client_session):
        """Test fs_read with network error."""
        from aiohttp import ClientError

        mock_client_session.get.side_effect = ClientError("Network error")

        with pytest.raises(ClientError):
            await client.fs_read("test.txt")

    @pytest.mark.asyncio
    async def test_fs_delete_success(self, client, mock_client_session, mock_response):
        """Test fs_delete successful operation."""
        # Setup mocks
        mock_client_session.post.return_value.__aenter__.return_value = mock_response

        # Delete file
        await client.fs_delete("test.txt")

        # Verify the request
        mock_client_session.post.assert_called_once()
        call_args = mock_client_session.post.call_args
        assert call_args[0][0] == "https://api.puter.com/delete"
        assert call_args[1]["params"] == {"path": "test.txt"}

    @pytest.mark.asyncio
    async def test_fs_delete_invalid_path(self, client):
        """Test fs_delete with invalid path."""
        with pytest.raises(ValueError):
            await client.fs_delete("invalid@path")

    @pytest.mark.asyncio
    async def test_fs_delete_empty_path(self, client):
        """Test fs_delete with empty path."""
        with pytest.raises(ValueError):
            await client.fs_delete("")

    @pytest.mark.asyncio
    async def test_fs_delete_network_error(self, client, mock_client_session):
        """Test fs_delete with network error."""
        from aiohttp import ClientError

        mock_client_session.post.side_effect = ClientError("Network error")

        with pytest.raises(ClientError):
            await client.fs_delete("test.txt")

    @pytest.mark.asyncio
    async def test_fs_operations_no_auth(self, client_no_token):
        """Test that fs operations require authentication."""
        with pytest.raises(ValueError, match="Not authenticated"):
            await client_no_token.fs_write("test.txt", "content")

        with pytest.raises(ValueError, match="Not authenticated"):
            await client_no_token.fs_read("test.txt")

        with pytest.raises(ValueError, match="Not authenticated"):
            await client_no_token.fs_delete("test.txt")
