import asyncio
import importlib.util
import sys
import types
from pathlib import Path


# Provide lightweight fastapi stubs so this module can be tested without fastapi installed.
fastapi_stub = types.ModuleType("fastapi")
websockets_stub = types.ModuleType("fastapi.websockets")


class WebSocketStateStub:
    CONNECTED = "CONNECTED"


class WebSocketStub:
    pass


class WebSocketDisconnectStub(Exception):
    pass


fastapi_stub.WebSocket = WebSocketStub
fastapi_stub.WebSocketDisconnect = WebSocketDisconnectStub
websockets_stub.WebSocketState = WebSocketStateStub

sys.modules.setdefault("fastapi", fastapi_stub)
sys.modules.setdefault("fastapi.websockets", websockets_stub)

MODULE_PATH = Path(__file__).resolve().parents[1] / "services" / "websocket_manager.py"
spec = importlib.util.spec_from_file_location("websocket_manager_module", MODULE_PATH)
websocket_manager = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(websocket_manager)


class FakeWebSocket:
    def __init__(self):
        self.client_state = websocket_manager.WebSocketState.CONNECTED
        self.sent_messages = []

    async def accept(self):
        return None

    async def send_json(self, message):
        self.sent_messages.append(message)


def test_subscribe_moves_connection_between_analyses_without_stale_membership(monkeypatch):
    manager = websocket_manager.ConnectionManager()
    monkeypatch.setattr(websocket_manager, "manager", manager)

    ws = FakeWebSocket()
    asyncio.run(manager.connect(ws, user_id="u1", analysis_id="analysis-a"))

    assert ws in manager.active_connections["analysis-a"]

    asyncio.run(
        websocket_manager.handle_websocket_message(
            ws, {"type": "subscribe_analysis", "analysis_id": "analysis-b"}
        )
    )

    assert "analysis-a" not in manager.active_connections
    assert ws in manager.active_connections["analysis-b"]
    assert manager.connection_metadata[ws]["analysis_id"] == "analysis-b"


def test_unsubscribe_clears_active_subscription_and_metadata(monkeypatch):
    manager = websocket_manager.ConnectionManager()
    monkeypatch.setattr(websocket_manager, "manager", manager)

    ws = FakeWebSocket()
    asyncio.run(manager.connect(ws, user_id="u1", analysis_id="analysis-a"))

    asyncio.run(
        websocket_manager.handle_websocket_message(
            ws, {"type": "unsubscribe_analysis", "analysis_id": "analysis-a"}
        )
    )

    assert "analysis-a" not in manager.active_connections
    assert manager.connection_metadata[ws]["analysis_id"] is None
