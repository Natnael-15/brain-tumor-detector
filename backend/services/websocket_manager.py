# WebSocket Manager for Real-time Updates
# Phase 3 Step 2: Real-time Communication

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        # Store active connections by analysis_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store user connections by user_id
        self.user_connections: Dict[str, Set[WebSocket]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str, analysis_id: Optional[str] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        # Store connection metadata
        connection_id = str(uuid.uuid4())
        self.connection_metadata[websocket] = {
            "connection_id": connection_id,
            "user_id": user_id,
            "analysis_id": analysis_id,
            "connected_at": datetime.now(),
            "last_ping": datetime.now()
        }
        
        # Add to user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)
        
        # Add to analysis connections if specified
        if analysis_id:
            if analysis_id not in self.active_connections:
                self.active_connections[analysis_id] = set()
            self.active_connections[analysis_id].add(websocket)
        
        logger.info(f"WebSocket connected: user={user_id}, analysis={analysis_id}, connection={connection_id}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Real-time connection established"
        }, websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket not in self.connection_metadata:
            return
            
        metadata = self.connection_metadata[websocket]
        user_id = metadata["user_id"]
        analysis_id = metadata.get("analysis_id")
        connection_id = metadata["connection_id"]
        
        # Remove from user connections
        if user_id in self.user_connections:
            self.user_connections[user_id].discard(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove from analysis connections
        if analysis_id and analysis_id in self.active_connections:
            self.active_connections[analysis_id].discard(websocket)
            if not self.active_connections[analysis_id]:
                del self.active_connections[analysis_id]
        
        # Remove metadata
        del self.connection_metadata[websocket]
        
        logger.info(f"WebSocket disconnected: user={user_id}, analysis={analysis_id}, connection={connection_id}")
    
    def update_analysis_subscription(self, websocket: WebSocket, analysis_id: str):
        """Move a websocket to the requested analysis subscription."""
        if websocket not in self.connection_metadata:
            return

        metadata = self.connection_metadata[websocket]
        previous_analysis_id = metadata.get("analysis_id")

        if previous_analysis_id and previous_analysis_id in self.active_connections:
            self.active_connections[previous_analysis_id].discard(websocket)
            if not self.active_connections[previous_analysis_id]:
                del self.active_connections[previous_analysis_id]

        if analysis_id not in self.active_connections:
            self.active_connections[analysis_id] = set()
        self.active_connections[analysis_id].add(websocket)
        metadata["analysis_id"] = analysis_id

    def remove_analysis_subscription(self, websocket: WebSocket, analysis_id: Optional[str] = None):
        """Remove a websocket from an analysis subscription."""
        metadata = self.connection_metadata.get(websocket)
        target_analysis_id = analysis_id or (metadata.get("analysis_id") if metadata else None)

        if target_analysis_id and target_analysis_id in self.active_connections:
            self.active_connections[target_analysis_id].discard(websocket)
            if not self.active_connections[target_analysis_id]:
                del self.active_connections[target_analysis_id]

        if metadata and metadata.get("analysis_id") == target_analysis_id:
            metadata["analysis_id"] = None

    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to a specific WebSocket connection"""
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending personal message: {e}")
                self.disconnect(websocket)
    
    async def send_to_analysis(self, message: Dict, analysis_id: str):
        """Send message to all connections watching a specific analysis"""
        if analysis_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[analysis_id].copy():
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(message)
                    else:
                        disconnected.append(websocket)
                except Exception as e:
                    logger.error(f"Error sending to analysis {analysis_id}: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws)
    
    async def send_to_user(self, message: Dict, user_id: str):
        """Send message to all connections for a specific user"""
        if user_id in self.user_connections:
            disconnected = []
            for websocket in self.user_connections[user_id].copy():
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(message)
                    else:
                        disconnected.append(websocket)
                except Exception as e:
                    logger.error(f"Error sending to user {user_id}: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws)
    
    async def broadcast(self, message: Dict):
        """Send message to all active connections"""
        all_connections = set()
        for connections in self.active_connections.values():
            all_connections.update(connections)
        for connections in self.user_connections.values():
            all_connections.update(connections)
        
        disconnected = []
        for websocket in all_connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
                else:
                    disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws)
    
    async def send_analysis_update(self, analysis_id: str, status: str, progress: int, data: Optional[Dict] = None):
        """Send analysis progress update to all relevant connections"""
        message = {
            "type": "analysis_update",
            "analysis_id": analysis_id,
            "stage": status,  # Frontend expects 'stage'
            "progress": progress,
            "message": data.get("message", f"Analysis {status}") if data else f"Analysis {status}",
            "timestamp": datetime.now().isoformat(),
            "model": data.get("model") if data else None,
            "file_name": data.get("file_name") if data else None,
            "data": data or {}
        }
        
        # Send to analysis-specific connections
        await self.send_to_analysis(message, analysis_id)
        
        # Also send to all user connections (for compatibility with frontend)
        await self.broadcast(message)
        
        logger.info(f"Analysis update sent: {analysis_id} - {status} ({progress}%)")
    
    async def send_analysis_result(self, analysis_id: str, result: Dict):
        """Send completed analysis result to all relevant connections"""
        
        # Extract model name properly
        model_name = result.get("model_name") or result.get("model_id") or result.get("analysis_metadata", {}).get("model")
        
        message = {
            "type": "analysis_update",  # Use same type as updates for consistency
            "analysis_id": analysis_id,
            "stage": "completed",
            "progress": 100,
            "message": "Analysis completed successfully",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "file_name": result.get("analysis_metadata", {}).get("file_names", [None])[0],
            "results": result,
            "model_id": result.get("model_id"),
            "model_type": result.get("model_type")
        }
        
        logger.info(f"📤 Sending analysis result message: {json.dumps(message, indent=2, default=str)}")
        
        # Send to analysis-specific connections
        await self.send_to_analysis(message, analysis_id)
        
        # Also send to all user connections (for compatibility with frontend)
        await self.broadcast(message)
        
        logger.info(f"Analysis result sent: {analysis_id}")
    
    async def send_analysis_error(self, analysis_id: str, error: str):
        """Send analysis error notification to all relevant connections"""
        message = {
            "type": "analysis_error",
            "analysis_id": analysis_id,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to analysis-specific connections
        await self.send_to_analysis(message, analysis_id)
        
        # Also send to all user connections (for compatibility with frontend)
        await self.broadcast(message)
        
        logger.info(f"Analysis error sent: {analysis_id} - {error}")
    
    async def send_system_notification(self, notification: Dict, user_id: Optional[str] = None):
        """Send system notification"""
        message = {
            "type": "system_notification",
            "notification": notification,
            "timestamp": datetime.now().isoformat()
        }
        
        if user_id:
            await self.send_to_user(message, user_id)
        else:
            await self.broadcast(message)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        total_connections = len(self.connection_metadata)
        total_users = len(self.user_connections)
        active_analyses = len(self.active_connections)
        
        return {
            "total_connections": total_connections,
            "total_users": total_users,
            "active_analyses": active_analyses,
            "connections_by_analysis": {
                analysis_id: len(connections) 
                for analysis_id, connections in self.active_connections.items()
            }
        }
    
    async def ping_all_connections(self):
        """Ping all connections to check if they're alive"""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.now().isoformat()
        }
        await self.broadcast(ping_message)
    
    async def handle_pong(self, websocket: WebSocket):
        """Handle pong response from client"""
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["last_ping"] = datetime.now()


# Global connection manager instance
manager = ConnectionManager()


# WebSocket event handlers
async def handle_websocket_message(websocket: WebSocket, data: Dict):
    """Handle incoming WebSocket messages"""
    message_type = data.get("type")
    
    if message_type == "pong":
        await manager.handle_pong(websocket)
    elif message_type == "ping":
        # Respond to ping with pong
        pong_message = {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        }
        await manager.send_personal_message(pong_message, websocket)
        logger.debug("Responded to ping with pong")
    elif message_type == "health_check":
        # Handle health check from client - respond immediately
        await manager.send_personal_message({
            "type": "health_check",
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "status": "healthy"
        }, websocket)
    elif message_type == "subscribe_analysis":
        analysis_id = data.get("analysis_id")
        if analysis_id and websocket in manager.connection_metadata:
            manager.update_analysis_subscription(websocket, analysis_id)

            # Send confirmation
            await manager.send_personal_message({
                "type": "subscription_confirmed",
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat()
            }, websocket)
    elif message_type == "unsubscribe_analysis":
        analysis_id = data.get("analysis_id")
        if websocket in manager.connection_metadata:
            manager.remove_analysis_subscription(websocket, analysis_id)
    else:
        logger.warning(f"Unknown WebSocket message type: {message_type}")


# Startup task to periodically ping connections
async def connection_health_check():
    """Periodically check connection health"""
    while True:
        await asyncio.sleep(30)  # Ping every 30 seconds
        await manager.ping_all_connections()
        
        # Log connection stats
        stats = manager.get_connection_stats()
        logger.info(f"WebSocket connections: {stats}")