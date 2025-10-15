"""FastAPI backend for Model Garden."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model_garden.training import ModelTrainer


# Pydantic models for API
class ModelInfo(BaseModel):
    """Model information response."""
    id: str
    name: str
    base_model: str
    status: str
    created_at: str
    updated_at: str
    size_bytes: Optional[int] = None
    path: str
    training_job_id: Optional[str] = None
    config: Optional[Dict] = None
    metrics: Optional[Dict] = None


class TrainingJobRequest(BaseModel):
    """Request to create a training job."""
    name: str
    base_model: str
    dataset_path: str
    output_dir: str
    hyperparameters: Optional[Dict] = None
    lora_config: Optional[Dict] = None
    from_hub: bool = False


class TrainingJobInfo(BaseModel):
    """Training job information."""
    id: str
    name: str
    status: str
    base_model: str
    dataset_path: str
    output_dir: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict] = None
    error_message: Optional[str] = None


class APIResponse(BaseModel):
    """Standard API response format."""
    success: bool
    data: Optional[Dict] = None
    message: str


class PaginatedResponse(BaseModel):
    """Paginated response format."""
    items: List[Dict]
    total: int
    page: int
    page_size: int
    pages: int


# Global variables for managing state
training_jobs: Dict[str, Dict] = {}
models_storage: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("🌱 Model Garden API starting up...")
    
    # Initialize storage directories
    storage_root = Path("./storage")
    models_dir = storage_root / "models"
    datasets_dir = storage_root / "datasets"
    logs_dir = storage_root / "logs"
    
    for directory in [models_dir, datasets_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Scan for existing models
    scan_existing_models()
    
    print(f"✓ Found {len(models_storage)} existing models")
    print("🚀 Model Garden API ready!")
    
    yield
    
    # Shutdown
    print("🌱 Model Garden API shutting down...")


def scan_existing_models():
    """Scan for existing models in the models directory."""
    models_dir = Path("./models")
    if not models_dir.exists():
        return
    
    for model_path in models_dir.iterdir():
        if model_path.is_dir():
            # Check if it's a valid model directory
            if (model_path / "config.json").exists() or (model_path / "adapter_config.json").exists():
                model_id = model_path.name
                models_storage[model_id] = {
                    "id": model_id,
                    "name": model_id,
                    "base_model": "unknown",
                    "status": "available",
                    "created_at": "unknown",
                    "updated_at": "unknown",
                    "path": str(model_path),
                    "size_bytes": calculate_dir_size(model_path),
                }


def calculate_dir_size(path: Path) -> int:
    """Calculate the total size of a directory."""
    total_size = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


# Create FastAPI app
app = FastAPI(
    title="Model Garden API",
    description="Fine-tune and serve LLMs with carbon footprint tracking",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoints (must be defined before static files to take precedence)
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Model Garden API is running"}


# Models endpoints
@app.get("/api/v1/models", response_model=PaginatedResponse)
async def list_models(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    base_model: Optional[str] = None,
):
    """List all available models."""
    # Filter models
    filtered_models = list(models_storage.values())
    
    if status:
        filtered_models = [m for m in filtered_models if m["status"] == status]
    
    if base_model:
        filtered_models = [m for m in filtered_models if m["base_model"] == base_model]
    
    # Pagination
    total = len(filtered_models)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    items = filtered_models[start_idx:end_idx]
    
    pages = (total + page_size - 1) // page_size
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@app.get("/api/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """Get details for a specific model."""
    if model_id not in models_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    model_data = models_storage[model_id]
    return ModelInfo(**model_data)


@app.delete("/api/v1/models/{model_id}", response_model=APIResponse)
async def delete_model(model_id: str):
    """Delete a model."""
    if model_id not in models_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Remove from storage (in production, also delete files)
    model_path = Path(models_storage[model_id]["path"])
    if model_path.exists():
        import shutil
        shutil.rmtree(model_path)
    
    del models_storage[model_id]
    
    return APIResponse(
        success=True,
        message=f"Model {model_id} deleted successfully"
    )


# Training endpoints
@app.get("/api/v1/training/jobs", response_model=PaginatedResponse)
async def list_training_jobs(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
):
    """List all training jobs."""
    # Filter jobs
    filtered_jobs = list(training_jobs.values())
    
    if status:
        filtered_jobs = [j for j in filtered_jobs if j["status"] == status]
    
    # Pagination
    total = len(filtered_jobs)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    items = filtered_jobs[start_idx:end_idx]
    
    pages = (total + page_size - 1) // page_size
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


@app.post("/api/v1/training/jobs", response_model=APIResponse)
async def create_training_job(job_request: TrainingJobRequest):
    """Create a new training job."""
    import uuid
    from datetime import datetime
    
    job_id = str(uuid.uuid4())
    
    # Create job record
    job_info = {
        "id": job_id,
        "name": job_request.name,
        "status": "queued",
        "base_model": job_request.base_model,
        "dataset_path": job_request.dataset_path,
        "output_dir": job_request.output_dir,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
        "progress": {"current_step": 0, "total_steps": 0, "epoch": 0},
        "error_message": None,
        "hyperparameters": job_request.hyperparameters or {},
        "lora_config": job_request.lora_config or {},
        "from_hub": job_request.from_hub,
    }
    
    training_jobs[job_id] = job_info
    
    # TODO: Add job to queue for processing
    
    return APIResponse(
        success=True,
        data={"job_id": job_id},
        message=f"Training job {job_id} created successfully"
    )


@app.get("/api/v1/training/jobs/{job_id}", response_model=TrainingJobInfo)
async def get_training_job(job_id: str):
    """Get details for a specific training job."""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    job_data = training_jobs[job_id]
    return TrainingJobInfo(**job_data)


@app.delete("/api/v1/training/jobs/{job_id}", response_model=APIResponse)
async def cancel_training_job(job_id: str):
    """Cancel a training job."""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    job = training_jobs[job_id]
    
    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in {job['status']} state"
        )
    
    # Update job status
    job["status"] = "cancelled"
    
    return APIResponse(
        success=True,
        message=f"Training job {job_id} cancelled successfully"
    )


# System endpoints
@app.get("/api/v1/system/status")
async def system_status():
    """Get system status information."""
    import psutil
    import torch
    
    # GPU information
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
        }
    else:
        gpu_info = {"available": False}
    
    return {
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
            },
        },
        "gpu": gpu_info,
        "storage": {
            "models_count": len(models_storage),
            "training_jobs_count": len(training_jobs),
            "active_jobs": len([j for j in training_jobs.values() if j["status"] in ["running", "queued"]]),
        },
    }


# Mount static files for the frontend (must be last to not override API routes)
frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
if frontend_build_path.exists():
    # Serve static assets
    app.mount("/_app", StaticFiles(directory=str(frontend_build_path / "_app")), name="static-assets")
    
    # Catch-all route for SvelteKit client-side routing
    from fastapi.responses import FileResponse
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SvelteKit SPA for all non-API routes."""
        # Try to serve specific HTML files first (e.g., models.html)
        html_file = frontend_build_path / f"{full_path}.html"
        if html_file.exists():
            return FileResponse(html_file)
        
        # Check if it's a directory with an index.html
        dir_path = frontend_build_path / full_path
        if dir_path.is_dir():
            index_file = dir_path / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
        
        # Fallback to main index.html for client-side routing
        return FileResponse(frontend_build_path / "index.html")
else:
    print("⚠️  Frontend build not found. Run 'cd frontend && npm run build' to build the frontend.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)