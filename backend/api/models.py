from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    ENSEMBLE = "ensemble"

class AnalysisStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# User models
class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: str
    role: str
    created_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True

# Authentication models
class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class TokenData(BaseModel):
    email: Optional[str] = None

# Analysis models
class ModelInfo(BaseModel):
    id: str
    name: str
    type: ModelType
    description: str
    loaded: bool
    version: Optional[str] = None

class AnalysisCreate(BaseModel):
    model: str
    description: Optional[str] = None

class PredictionResult(BaseModel):
    tumor_detected: bool
    tumor_type: Optional[str] = None
    confidence: float
    tumor_volume_ml: Optional[float] = None
    location: Optional[str] = None

class SegmentationResult(BaseModel):
    tumor_mask: str
    volume_rendering: str
    slice_count: int

class AnalysisMetrics(BaseModel):
    dice_score: Optional[float] = None
    hausdorff_distance: Optional[float] = None
    processing_time: float
    model_inference_time: float

class AnalysisResult(BaseModel):
    analysis_id: str
    model_used: str
    predictions: PredictionResult
    segmentation: SegmentationResult
    metrics: AnalysisMetrics
    clinical_notes: List[str]
    completed_at: datetime

class AnalysisStatusModel(BaseModel):
    id: str
    status: AnalysisStatus
    progress: int
    model: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    user_id: str

class AnalysisList(BaseModel):
    analyses: List[AnalysisStatusModel]
    total: int
    limit: int
    offset: int

# File upload models
class FileInfo(BaseModel):
    filename: str
    size: int
    content_type: str
    upload_time: datetime

class UploadResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    files: List[FileInfo]

# Health check models
class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    version: str
    models_loaded: int
    system_info: Optional[Dict[str, Any]] = None

# Error models
class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime

class ValidationError(BaseModel):
    field: str
    message: str
    value: Any

class ValidationErrorResponse(BaseModel):
    detail: str
    errors: List[ValidationError]
    timestamp: datetime