"""Tests for WebSocket connection management."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from model_garden.api.websocket import ConnectionManager, get_connection_manager


class TestConnectionManager:
    """Tests for the ConnectionManager class."""

    def test_init(self):
        """Test ConnectionManager initialization."""
        manager = ConnectionManager()
        assert manager.active_connections == {}

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connecting a WebSocket."""
        manager = ConnectionManager()

        # Create mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()

        await manager.connect(mock_websocket, "job-123")

        # Verify accept was called
        mock_websocket.accept.assert_awaited_once()

        # Verify connection was stored
        assert "job-123" in manager.active_connections
        assert mock_websocket in manager.active_connections["job-123"]

    @pytest.mark.asyncio
    async def test_connect_multiple_to_same_job(self):
        """Test connecting multiple WebSockets to the same job."""
        manager = ConnectionManager()

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        await manager.connect(mock_ws1, "job-123")
        await manager.connect(mock_ws2, "job-123")

        assert len(manager.active_connections["job-123"]) == 2
        assert mock_ws1 in manager.active_connections["job-123"]
        assert mock_ws2 in manager.active_connections["job-123"]

    @pytest.mark.asyncio
    async def test_connect_to_different_jobs(self):
        """Test connecting WebSockets to different jobs."""
        manager = ConnectionManager()

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        await manager.connect(mock_ws1, "job-1")
        await manager.connect(mock_ws2, "job-2")

        assert "job-1" in manager.active_connections
        assert "job-2" in manager.active_connections
        assert len(manager.active_connections["job-1"]) == 1
        assert len(manager.active_connections["job-2"]) == 1

    def test_disconnect(self):
        """Test disconnecting a WebSocket."""
        manager = ConnectionManager()

        mock_websocket = MagicMock()
        manager.active_connections["job-123"] = [mock_websocket]

        manager.disconnect(mock_websocket, "job-123")

        # Job ID should be removed when no connections remain
        assert "job-123" not in manager.active_connections

    def test_disconnect_one_of_multiple(self):
        """Test disconnecting one WebSocket when multiple exist."""
        manager = ConnectionManager()

        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        manager.active_connections["job-123"] = [mock_ws1, mock_ws2]

        manager.disconnect(mock_ws1, "job-123")

        assert "job-123" in manager.active_connections
        assert len(manager.active_connections["job-123"]) == 1
        assert mock_ws2 in manager.active_connections["job-123"]

    def test_disconnect_nonexistent_job(self):
        """Test disconnecting from a non-existent job doesn't crash."""
        manager = ConnectionManager()
        mock_websocket = MagicMock()

        # Should not raise
        manager.disconnect(mock_websocket, "nonexistent-job")

    def test_disconnect_nonexistent_websocket(self):
        """Test disconnecting a WebSocket that's not in the list."""
        manager = ConnectionManager()

        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        manager.active_connections["job-123"] = [mock_ws1]

        # Disconnecting ws2 which is not in the list should not crash
        manager.disconnect(mock_ws2, "job-123")

        # Original connection should remain
        assert mock_ws1 in manager.active_connections["job-123"]

    @pytest.mark.asyncio
    async def test_send_update(self):
        """Test sending update to all connections for a job."""
        manager = ConnectionManager()

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        manager.active_connections["job-123"] = [mock_ws1, mock_ws2]

        message = {"type": "progress", "step": 10, "loss": 0.5}
        await manager.send_update("job-123", message)

        mock_ws1.send_json.assert_awaited_once_with(message)
        mock_ws2.send_json.assert_awaited_once_with(message)

    @pytest.mark.asyncio
    async def test_send_update_nonexistent_job(self):
        """Test sending update to non-existent job doesn't crash."""
        manager = ConnectionManager()

        # Should not raise
        await manager.send_update("nonexistent", {"message": "test"})

    @pytest.mark.asyncio
    async def test_send_update_removes_failed_connections(self):
        """Test that failed connections are removed on send."""
        manager = ConnectionManager()

        # One working connection, one failing
        mock_ws_good = AsyncMock()
        mock_ws_bad = AsyncMock()
        mock_ws_bad.send_json.side_effect = Exception("Connection closed")

        manager.active_connections["job-123"] = [mock_ws_good, mock_ws_bad]

        await manager.send_update("job-123", {"test": "message"})

        # Good connection should receive message
        mock_ws_good.send_json.assert_awaited_once()

        # Bad connection should be removed
        assert mock_ws_bad not in manager.active_connections.get("job-123", [])

    @pytest.mark.asyncio
    async def test_broadcast_system_update(self):
        """Test broadcasting to all jobs."""
        manager = ConnectionManager()

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()

        manager.active_connections["job-1"] = [mock_ws1]
        manager.active_connections["job-2"] = [mock_ws2, mock_ws3]

        message = {"type": "system", "message": "Server restarting"}
        await manager.broadcast_system_update(message)

        mock_ws1.send_json.assert_awaited_once_with(message)
        mock_ws2.send_json.assert_awaited_once_with(message)
        mock_ws3.send_json.assert_awaited_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_empty_connections(self):
        """Test broadcasting with no connections doesn't crash."""
        manager = ConnectionManager()

        # Should not raise
        await manager.broadcast_system_update({"message": "test"})


class TestGetConnectionManager:
    """Tests for the get_connection_manager function."""

    def test_returns_manager(self):
        """Test that get_connection_manager returns a ConnectionManager."""
        manager = get_connection_manager()
        assert isinstance(manager, ConnectionManager)

    def test_returns_same_instance(self):
        """Test that get_connection_manager returns the same instance."""
        manager1 = get_connection_manager()
        manager2 = get_connection_manager()
        assert manager1 is manager2
