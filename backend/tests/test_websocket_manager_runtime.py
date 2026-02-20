import asyncio

from fastapi.websockets import WebSocketState

from backend.services.websocket_manager import ConnectionManager


class DummyWebSocket:
    def __init__(self):
        self.client_state = WebSocketState.CONNECTED
        self.sent = []

    async def accept(self):
        return None

    async def send_json(self, message):
        self.sent.append(message)



def test_connection_stats_include_non_analysis_connections():
    manager = ConnectionManager()
    ws = DummyWebSocket()

    asyncio.run(manager.connect(ws, "user-1"))
    stats = manager.get_connection_stats()

    assert stats["total_connections"] == 1
    assert stats["total_users"] == 1
    assert stats["active_analyses"] == 0
