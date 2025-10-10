#!/usr/bin/env python3
"""
Mock Backend Server for Testing WebSocket Connection
Minimal implementation to test the frontend without full dependencies
"""

import asyncio
import json
from datetime import datetime
from typing import Set
import sys

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("ERROR: Missing dependencies. Please install:")
    print("  pip3 install fastapi uvicorn websockets")
    sys.exit(1)

app = FastAPI(title="Mock Brain Tumor Detector API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection management
active_connections: Set[WebSocket] = set()

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "message": "Mock backend running"}

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.add(websocket)
    print(f"✅ WebSocket connected: {user_id}")
    
    # Send welcome message
    await websocket.send_json({
        "type": "connection_established",
        "connection_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "message": "Mock WebSocket connection established"
    })
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            print(f"📨 Received: {data}")
            
            # Send pong response to ping
            if data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        print(f"🔴 WebSocket disconnected: {user_id}")
        active_connections.discard(websocket)

if __name__ == "__main__":
    print("🚀 Starting Mock Backend Server...")
    print("   WebSocket endpoint: ws://localhost:8000/ws/{user_id}")
    print("   Health check: http://localhost:8000/api/v1/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
