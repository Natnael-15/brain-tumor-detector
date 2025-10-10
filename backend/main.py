# FastAPI Backend for Brain MRI Tumor Detector
# Modern Python API with async support - Phase 3 Integration

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import logging
import shutil
import json

# Import services
from services.model_service import model_service
from services.websocket_manager import manager, handle_websocket_message, connection_health_check

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Brain MRI Tumor Detector API",
    description="Advanced AI-powered brain tumor detection and analysis API with Phase 3 Integration",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS - Allow multiple frontend ports for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "ws://localhost:8000",
        "ws://127.0.0.1:8000",
        "ws://0.0.0.0:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global variables
analysis_results = {}
analysis_status = {}

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    logger.info("Starting Brain MRI Tumor Detector API v3.0...")
    logger.info("Phase 3: Advanced AI Models Integration + Real-time WebSockets")
    
    # Initialize model service
    await model_service.initialize()
    
    # Start WebSocket health check task (temporarily disabled for debugging)
    # asyncio.create_task(connection_health_check())
    
    # Get available models
    models = await model_service.get_available_models()
    logger.info(f"Loaded {len(models)} AI models:")
    for model in models:
        logger.info(f"  - {model['name']} ({model['type']}) - {model['model_type']}")
    
    logger.info("API startup completed successfully - Ready for Phase 3 with Real-time Updates!")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Mock authentication"""
    if credentials.credentials == "mock-token":
        return {"id": "1", "email": "demo@example.com", "name": "Demo User"}
    raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    models = await model_service.get_available_models()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "phase": "Phase 3 - Advanced AI Integration",
        "models_loaded": len([m for m in models if m.get("loaded", False)]),
        "ai_backend": "Phase 1 Models Integrated"
    }

@app.get("/api/v1/websocket/test")
async def websocket_test():
    """Test WebSocket configuration and availability"""
    return {
        "websocket_available": True,
        "endpoints": {
            "user_connection": "/ws/{user_id}",
            "analysis_connection": "/ws/{user_id}/analysis/{analysis_id}"
        },
        "available_hosts": ["localhost:8000", "127.0.0.1:8000"],
        "server_binding": "0.0.0.0:8000",
        "cors_configuration": {
            "http_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "ws_origins": ["ws://localhost:8000", "ws://127.0.0.1:8000", "ws://0.0.0.0:8000"]
        },
        "connection_status": {
            "total_connections": len(manager.user_connections) if hasattr(manager, 'user_connections') else 0,
            "active_users": len(manager.user_connections.keys()) if hasattr(manager, 'user_connections') else 0
        }
    }

@app.get("/api/v1/models")
async def get_available_models():
    """Get list of available AI models with Phase 1 integration"""
    models = await model_service.get_available_models()
    return {
        "models": models,
        "total": len(models),
        "phase": "Phase 3 - Real AI Models",
        "capabilities": [
            "Real-time tumor detection",
            "Advanced segmentation",
            "Ensemble predictions", 
            "Uncertainty quantification",
            "Multi-modal analysis"
        ]
    }

@app.post("/api/v1/analysis/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    model: str = "ensemble",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: dict = Depends(get_current_user)
):
    """Upload medical images and start analysis with Phase 3 AI models"""
    
    # Validate model selection
    available_models = await model_service.get_available_models()
    model_ids = [m["id"] for m in available_models]
    
    if model not in model_ids:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available. Available: {model_ids}")
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Create upload directory for this analysis
    analysis_dir = UPLOAD_DIR / f"analysis_{analysis_id}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    uploaded_files = []
    for file in files:
        if file.size and file.size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="File too large")
        
        # Ensure filename is not None
        filename = file.filename or f"uploaded_file_{uuid.uuid4().hex[:8]}"
        file_path = analysis_dir / filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        uploaded_files.append(str(file_path))
    
    # Initialize analysis status
    analysis_status[analysis_id] = {
        "id": analysis_id,
        "status": "queued",
        "progress": 0,
        "model": model,
        "files": uploaded_files,
        "created_at": datetime.now().isoformat(),
        "user_id": current_user["id"],
        "phase": "Phase 3 - Real AI Analysis"
    }
    
    # Start background analysis with real AI models
    background_tasks.add_task(run_analysis, analysis_id, uploaded_files, model)
    
    return {
        "analysis_id": analysis_id,
        "status": "queued",
        "message": "Phase 3 AI analysis started successfully",
        "model_info": next((m for m in available_models if m["id"] == model), None)
    }

async def run_analysis(analysis_id: str, file_paths: List[str], model: str):
    """Run analysis in background using Phase 3 AI models with real-time updates"""
    try:
        # Update status and send WebSocket update
        analysis_status[analysis_id]["status"] = "preprocessing"
        analysis_status[analysis_id]["progress"] = 10
        await manager.send_analysis_update(analysis_id, "preprocessing", 10, {
            "message": "Starting preprocessing...",
            "model": model,
            "file_name": Path(file_paths[0]).name if file_paths else "Unknown",
            "files": len(file_paths)
        })
        
        # Preprocessing phase
        logger.info(f"Starting Phase 3 analysis {analysis_id} with model {model}")
        await asyncio.sleep(1)
        analysis_status[analysis_id]["progress"] = 25
        await manager.send_analysis_update(analysis_id, "preprocessing", 25, {
            "message": "Preprocessing medical images...",
            "model": model,
            "file_name": Path(file_paths[0]).name if file_paths else "Unknown"
        })
        
        # Model inference using real AI models
        analysis_status[analysis_id]["status"] = "analyzing"
        analysis_status[analysis_id]["progress"] = 40
        await manager.send_analysis_update(analysis_id, "analyzing", 40, {
            "message": f"Running {model} model inference...",
            "model": model,
            "file_name": Path(file_paths[0]).name if file_paths else "Unknown"
        })
        
        # Use the model service for real prediction
        primary_file = file_paths[0] if file_paths else None
        if primary_file:
            results = await model_service.predict(model, primary_file, analysis_id)
        else:
            raise ValueError("No files provided for analysis")
        
        analysis_status[analysis_id]["progress"] = 80
        await manager.send_analysis_update(analysis_id, "analyzing", 80, {
            "message": "AI analysis complete, processing results...",
            "file_name": Path(file_paths[0]).name if file_paths else "Unknown",
            "model": model
        })
        
        # Post-processing
        analysis_status[analysis_id]["status"] = "finalizing"
        await asyncio.sleep(0.5)
        analysis_status[analysis_id]["progress"] = 95
        await manager.send_analysis_update(analysis_id, "finalizing", 95, {
            "message": "Generating clinical report...",
            "file_name": Path(file_paths[0]).name if file_paths else "Unknown",
            "model": model
        })
        
        # Enhance results with additional metadata
        enhanced_results = {
            **results,
            "analysis_metadata": {
                "files_processed": len(file_paths),
                "file_names": [Path(fp).name for fp in file_paths],
                "processing_completed": datetime.now().isoformat(),
                "phase": "Phase 3 - Advanced AI"
            },
            "visualization": {
                "segmentation_available": True,
                "report_url": f"/api/v1/analysis/{analysis_id}/report"
            },
            "clinical_notes": [
                "AI-powered analysis completed",
                "Phase 3 model integration successful",
                "Results ready for clinical review",
                "Confidence-based recommendations provided"
            ]
        }
        
        # Store results
        analysis_results[analysis_id] = enhanced_results
        analysis_status[analysis_id]["status"] = "completed"
        analysis_status[analysis_id]["progress"] = 100
        analysis_status[analysis_id]["completed_at"] = datetime.now().isoformat()
        
        # Send final WebSocket update with results
        await manager.send_analysis_result(analysis_id, enhanced_results)
        
        logger.info(f"Phase 3 analysis {analysis_id} completed successfully with {model}")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        analysis_status[analysis_id]["status"] = "failed"
        analysis_status[analysis_id]["error"] = str(e)
        analysis_status[analysis_id]["completed_at"] = datetime.now().isoformat()
        
        # Send error WebSocket update
        await manager.send_analysis_error(analysis_id, str(e))

@app.get("/api/v1/analysis/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get analysis status and progress"""
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status = analysis_status[analysis_id]
    
    # Check user permission
    if status["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return status

@app.get("/api/v1/analysis/{analysis_id}/results")
async def get_analysis_results(
    analysis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get analysis results"""
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    status = analysis_status[analysis_id]
    
    # Check user permission
    if status["user_id"] != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return analysis_results[analysis_id]

@app.post("/api/v1/auth/login")
async def login(credentials: dict):
    """Login endpoint"""
    email = credentials.get("email")
    password = credentials.get("password")
    
    # Mock authentication
    if email and password:
        return {
            "access_token": "mock-token",
            "token_type": "bearer",
            "user": {
                "id": "1",
                "email": email,
                "name": "Demo User",
                "role": "radiologist"
            }
        }
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

# WebSocket Endpoints for Real-time Updates
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time updates"""
    logger.info(f"🔗 WebSocket connection request from user: {user_id}")
    logger.info(f"🔗 WebSocket headers: {websocket.headers}")
    logger.info(f"🔗 WebSocket client: {websocket.client}")
    
    try:
        await manager.connect(websocket, user_id)
        logger.info(f"✅ WebSocket connected successfully: {user_id}")
        
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            logger.info(f"📨 WebSocket message from {user_id}: {data}")
            await handle_websocket_message(websocket, data)
            
    except WebSocketDisconnect:
        logger.info(f"🔴 WebSocket disconnected: {user_id}")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error for user {user_id}: {e}")
        logger.error(f"❌ WebSocket error details: {type(e).__name__}: {str(e)}")
        manager.disconnect(websocket)

@app.websocket("/ws/{user_id}/analysis/{analysis_id}")
async def websocket_analysis_endpoint(websocket: WebSocket, user_id: str, analysis_id: str):
    """WebSocket endpoint for specific analysis updates"""
    await manager.connect(websocket, user_id, analysis_id)
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            await handle_websocket_message(websocket, data)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}, analysis {analysis_id}: {e}")
        manager.disconnect(websocket)

@app.get("/api/v1/websocket/stats")
async def get_websocket_stats(current_user: dict = Depends(get_current_user)):
    """Get WebSocket connection statistics"""
    return manager.get_connection_stats()

@app.post("/api/v1/notifications/send")
async def send_notification(
    notification: dict,
    user_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Send system notification via WebSocket"""
    await manager.send_system_notification(notification, user_id)
    return {"status": "notification_sent", "user_id": user_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")