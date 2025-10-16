"""FastAPI backend for Model Garden."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
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
    is_vision: bool = False  # Flag for vision-language models
    model_type: Optional[str] = None  # 'text' or 'vision'


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


def run_training_job(job_id: str):
    """Execute a training job in the background."""
    try:
        job = training_jobs[job_id]
        
        # Update job status to running
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat() + "Z"
        
        print(f"🚀 Starting training job {job_id}: {job['name']}")
        
        # Check if this is a vision-language model
        is_vision = job.get("is_vision", False)
        from_hub = job.get("from_hub", False)
        
        if is_vision:
            # Use VisionLanguageTrainer for vision models
            from model_garden.vision_training import VisionLanguageTrainer
            
            print(f"🎨 Using VisionLanguageTrainer for {job['base_model']}")
            
            trainer = VisionLanguageTrainer(
                base_model=job["base_model"],
                max_seq_length=job["hyperparameters"].get("max_seq_length", 2048),
                load_in_4bit=True,
            )
            
            # Load model
            trainer.load_model()
            
            # Prepare for training with LoRA
            lora_config = job["lora_config"]
            trainer.prepare_for_training(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 16),
            )
            
            # Load and format dataset (handles both file and Hub, base64 and file paths)
            train_dataset = trainer.load_dataset(
                dataset_path=job["dataset_path"],
                from_hub=from_hub,
            )
            formatted_dataset = trainer.format_dataset(train_dataset)
            
            # Train
            hyperparams = job["hyperparameters"]
            trainer.train(
                dataset=formatted_dataset,
                output_dir=job["output_dir"],
                num_train_epochs=hyperparams.get("num_epochs", 3),
                per_device_train_batch_size=hyperparams.get("batch_size", 1),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 8),
                learning_rate=hyperparams.get("learning_rate", 2e-5),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
            )
            
            # Save model
            trainer.save_model(job["output_dir"], save_method="lora")
        else:
            # Use standard ModelTrainer for text-only models
            trainer = ModelTrainer(
                base_model=job["base_model"],
                max_seq_length=job["hyperparameters"].get("max_seq_length", 2048),
                load_in_4bit=True,
            )
            
            # Load model
            trainer.load_model()
            
            # Prepare for training with LoRA
            lora_config = job["lora_config"]
            trainer.prepare_for_training(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 16),
            )
            
            # Load dataset
            if from_hub:
                train_dataset = trainer.load_dataset_from_hub(job["dataset_path"])
            else:
                train_dataset = trainer.load_dataset_from_file(job["dataset_path"])
            
            # Format dataset
            train_dataset = trainer.format_dataset(
                train_dataset,
                instruction_field=job["hyperparameters"].get("instruction_field", "instruction"),
                input_field=job["hyperparameters"].get("input_field", "input"),
                output_field=job["hyperparameters"].get("output_field", "output"),
            )
            
            # Train
            hyperparams = job["hyperparameters"]
            trainer.train(
                dataset=train_dataset,
                output_dir=job["output_dir"],
                num_train_epochs=hyperparams.get("num_epochs", 3),
                per_device_train_batch_size=hyperparams.get("batch_size", 2),
                gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
                learning_rate=hyperparams.get("learning_rate", 2e-4),
                max_steps=hyperparams.get("max_steps", -1),
                logging_steps=hyperparams.get("logging_steps", 10),
                save_steps=hyperparams.get("save_steps", 100),
            )
            
            # Save final model
            save_method = hyperparams.get("save_method", "merged_16bit")
            if save_method != "lora":
                trainer.save_model(job["output_dir"], save_method=save_method)
        
        # Update job status to completed
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job["progress"] = {"current_step": 100, "total_steps": 100, "epoch": hyperparams.get("num_epochs", 3)}
        
        # Add model to storage
        model_id = Path(job["output_dir"]).name
        models_storage[model_id] = {
            "id": model_id,
            "name": job["name"],
            "base_model": job["base_model"],
            "status": "available",
            "created_at": job["created_at"],
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "path": job["output_dir"],
            "training_job_id": job_id,
            "size_bytes": calculate_dir_size(Path(job["output_dir"])),
        }
        
        print(f"✅ Training job {job_id} completed successfully!")
        
    except Exception as e:
        # Update job status to failed
        job["status"] = "failed"
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job["error_message"] = str(e)
        print(f"❌ Training job {job_id} failed: {e}")


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
    
    # Auto-load inference model if specified
    autoload_model = os.getenv("MODEL_GARDEN_AUTOLOAD_MODEL")
    if autoload_model:
        print(f"🔄 Auto-loading inference model: {autoload_model}")
        try:
            from model_garden.inference import InferenceService, set_inference_service
            
            inference_service = InferenceService(model_path=autoload_model)
            await inference_service.load_model()
            set_inference_service(inference_service)  # Register globally
            print(f"✅ Inference model loaded: {autoload_model}")
        except Exception as e:
            print(f"❌ Failed to auto-load model: {e}")
            import traceback
            traceback.print_exc()
    
    print("🚀 Model Garden API ready!")
    
    yield
    
    # Shutdown
    print("🌱 Model Garden API shutting down...")
    
    # Cleanup inference service if loaded
    from model_garden.inference import get_inference_service, set_inference_service
    inference_service = get_inference_service()
    if inference_service is not None:
        try:
            await inference_service.close()
            set_inference_service(None)
        except Exception as e:
            print(f"Warning: Error closing inference service: {e}")


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


# Add validation error handler to debug 422 errors
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Log and return validation errors."""
    print(f"🔴 Validation Error on {request.method} {request.url.path}")
    
    # Get error details but truncate any large input values
    errors = exc.errors()
    truncated_errors = []
    for error in errors:
        error_copy = error.copy()
        # Truncate large input values in the error
        if 'input' in error_copy and isinstance(error_copy['input'], str) and len(error_copy['input']) > 200:
            error_copy['input'] = f"{error_copy['input'][:200]}... [truncated {len(error_copy['input'])} chars]"
        truncated_errors.append(error_copy)
    
    print(f"   Error count: {len(truncated_errors)}")
    for i, error in enumerate(truncated_errors[:3]):  # Only show first 3 errors
        print(f"   Error {i+1}: {error}")
    if len(truncated_errors) > 3:
        print(f"   ... and {len(truncated_errors) - 3} more errors")
    
    # Don't include the full body or large inputs in the response
    return JSONResponse(
        status_code=422,
        content={"detail": truncated_errors, "error_count": len(truncated_errors)}
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
async def create_training_job(job_request: TrainingJobRequest, background_tasks: BackgroundTasks):
    """Create a new training job."""
    import uuid
    
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
        "is_vision": job_request.is_vision or False,
        "model_type": job_request.model_type or ("vision" if job_request.is_vision else "text"),
    }
    
    training_jobs[job_id] = job_info
    
    # Start training job in background
    background_tasks.add_task(run_training_job, job_id)
    
    return APIResponse(
        success=True,
        data={"job_id": job_id},
        message=f"Training job {job_id} created and queued for execution"
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


# ============================================================================
# Inference Endpoints
# ============================================================================

class InferenceRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False


class ChatMessage(BaseModel):
    """Chat message with support for both text and multimodal content."""
    role: str  # system, user, assistant
    content: Union[str, List[Dict], Dict]  # Can be string or structured content for vision
    name: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow extra fields like 'image' for compatibility


class ResponseFormat(BaseModel):
    """OpenAI-compatible response format for structured outputs."""
    type: str  # "json_object" or "json_schema" or "text"
    json_schema: Optional[Dict] = None  # JSON schema for structured output


def convert_response_format_to_structured_outputs(response_format: Optional[ResponseFormat]) -> Optional[Dict]:
    """Convert OpenAI response_format to vLLM structured_outputs parameters.
    
    Args:
        response_format: OpenAI-style response format specification
        
    Returns:
        Dictionary with structured output parameters for vLLM, or None if not applicable
    """
    if not response_format:
        return None
    
    if response_format.type == "text":
        # No structured output needed
        return None
    
    elif response_format.type == "json_object":
        # Generic JSON object - use a permissive JSON schema
        return {
            "json": {
                "type": "object",
                "properties": {},
                "additionalProperties": True
            }
        }
    
    elif response_format.type == "json_schema":
        # Specific JSON schema provided
        if not response_format.json_schema:
            raise ValueError("json_schema must be provided when type is 'json_schema'")
        
        # Extract the schema from the OpenAI format
        # OpenAI format: {"name": "schema_name", "schema": {...}, "strict": true}
        schema = response_format.json_schema
        if isinstance(schema, dict):
            # If it has a 'schema' key, extract it
            if "schema" in schema:
                actual_schema = schema["schema"]
            else:
                actual_schema = schema
            
            return {"json": actual_schema}
        
        return {"json": schema}
    
    else:
        # Unknown type, return None
        return None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: Optional[str] = None  # Model name (optional, we use the loaded model)
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None  # None = auto-determine based on response_format
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = -1
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = 1  # Number of completions to generate
    best_of: Optional[int] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    user: Optional[str] = None  # User identifier
    response_format: Optional[ResponseFormat] = None  # Structured output format
    
    class Config:
        extra = "allow"  # Allow extra fields for compatibility


class LoadModelRequest(BaseModel):
    """Request to load a model for inference."""
    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    quantization: Optional[str] = None


@app.post("/api/v1/inference/load", response_model=APIResponse)
async def load_inference_model(request: LoadModelRequest):
    """Load a model for inference."""
    from model_garden.inference import InferenceService, get_inference_service, set_inference_service
    
    # Check if a model is already loaded
    current_service = get_inference_service()
    if current_service and current_service.is_loaded:
        return APIResponse(
            success=False,
            message=f"Model already loaded: {current_service.model_path}. Unload it first."
        )
    
    try:
        # Create new inference service
        service = InferenceService(
            model_path=request.model_path,
            tensor_parallel_size=request.tensor_parallel_size,
            gpu_memory_utilization=request.gpu_memory_utilization,
            max_model_len=request.max_model_len,
            dtype=request.dtype,
            quantization=request.quantization,
        )
        
        # Load the model
        await service.load_model()
        
        # Set as global service
        set_inference_service(service)
        
        return APIResponse(
            success=True,
            data=service.get_model_info(),
            message=f"Model loaded successfully: {request.model_path}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@app.post("/api/v1/inference/unload", response_model=APIResponse)
async def unload_inference_model():
    """Unload the currently loaded model."""
    from model_garden.inference import get_inference_service, set_inference_service
    
    service = get_inference_service()
    if not service or not service.is_loaded:
        return APIResponse(
            success=False,
            message="No model currently loaded"
        )
    
    try:
        await service.unload_model()
        set_inference_service(None)
        
        return APIResponse(
            success=True,
            message="Model unloaded successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )


@app.get("/api/v1/inference/status")
async def get_inference_status():
    """Get inference service status."""
    from model_garden.inference import get_inference_service
    
    service = get_inference_service()
    if not service:
        return {
            "loaded": False,
            "model_info": None,
        }
    
    return {
        "loaded": service.is_loaded,
        "model_info": service.get_model_info() if service.is_loaded else None,
    }


@app.post("/api/v1/inference/generate")
async def generate_text(request: InferenceRequest):
    """Generate text from a prompt."""
    from model_garden.inference import get_inference_service
    from fastapi.responses import StreamingResponse
    import json
    
    service = get_inference_service()
    if not service or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded. Load a model first using /api/v1/inference/load"
        )
    
    try:
        if request.stream:
            # Streaming response
            async def generate_stream():
                stream = await service.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    frequency_penalty=request.frequency_penalty,
                    presence_penalty=request.presence_penalty,
                    stop=request.stop,
                    stream=True,
                )
                
                async for chunk in stream:
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Complete response
            text = await service.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                stop=request.stop,
                stream=False,
            )
            
            return {
                "text": text,
                "model": service.model_path,
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@app.post("/api/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with vision support."""
    from model_garden.inference import get_inference_service
    from fastapi.responses import StreamingResponse
    import json
    import base64
    import re
    
    # Log the incoming request for debugging
    print(f"📨 Received chat completion request: model={request.model}, messages={len(request.messages)}, stream={request.stream}")
    
    service = get_inference_service()
    if not service or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded. Load a model first using /api/v1/inference/load"
        )
    
    try:
        # Process messages and extract multimodal content
        processed_messages = []
        image_data = None
        
        for msg in request.messages:
            msg_dict = msg.dict()
            content = msg_dict.get("content", "")
            
            # Handle multimodal content (OpenAI format)
            if isinstance(content, list):
                # Content is an array of content parts (text + images)
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            # Extract image URL or base64 data
                            image_url = part.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url
                            
                            # Check if it's a base64 data URL
                            if url.startswith("data:image/"):
                                # Extract base64 data
                                match = re.match(r"data:image/[^;]+;base64,(.+)", url)
                                if match:
                                    image_data = match.group(1)
                                    print(f"🖼️  Extracted base64 image data ({len(image_data)} chars)")
                            else:
                                # It's a regular URL
                                image_data = url
                                print(f"🖼️  Using image URL: {url[:100]}...")
                    else:
                        text_parts.append(str(part))
                
                # Combine text parts
                msg_dict["content"] = " ".join(text_parts)
            elif isinstance(content, dict):
                # Single content part
                if content.get("type") == "text":
                    msg_dict["content"] = content.get("text", "")
                elif content.get("type") == "image_url":
                    image_url = content.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url
                    
                    if url.startswith("data:image/"):
                        match = re.match(r"data:image/[^;]+;base64,(.+)", url)
                        if match:
                            image_data = match.group(1)
                    else:
                        image_data = url
            # else: content is already a string, keep as-is
            
            # Check for 'image' field in message (custom format)
            if "image" in msg_dict and msg_dict["image"]:
                image_data = msg_dict["image"]
                print(f"🖼️  Found image in custom 'image' field")
            
            processed_messages.append(msg_dict)
        
        # Prepare generation parameters
        gen_params = {
            "messages": processed_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream or False,
        }
        
        # Add image if found
        if image_data:
            gen_params["image"] = image_data
            print(f"✅ Added image to generation parameters")
        
        # Add stop sequences if provided
        if request.stop:
            gen_params["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]
        
        # Add structured output parameters if response_format is provided
        if request.response_format:
            structured_outputs = convert_response_format_to_structured_outputs(request.response_format)
            if structured_outputs:
                gen_params["structured_outputs"] = structured_outputs
                print(f"✅ Added structured output parameters: {list(structured_outputs.keys())}")
                
                # Auto-increase max_tokens for structured outputs if not specified
                if request.max_tokens is None:
                    # Estimate needed tokens based on schema complexity
                    if request.response_format.type == "json_schema" and request.response_format.json_schema:
                        # For schemas, use 2048 tokens as a safe default
                        gen_params["max_tokens"] = 2048
                        print(f"⚙️  Auto-set max_tokens=2048 for JSON schema output")
                    else:
                        # For generic json_object, use 1024 tokens
                        gen_params["max_tokens"] = 1024
                        print(f"⚙️  Auto-set max_tokens=1024 for generic JSON output")
        
        # If still no max_tokens set, use reasonable default
        if gen_params["max_tokens"] is None:
            gen_params["max_tokens"] = 512  # Higher than old 256 default
            print(f"⚙️  Using default max_tokens=512")
        
        if request.stream:
            # Streaming response
            async def generate_stream():
                stream = await service.chat_completion(**gen_params)
                
                async for chunk in stream:
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Complete response
            response = await service.chat_completion(**gen_params)
            
            return response
            
    except Exception as e:
        import traceback
        print(f"❌ Chat completion error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}"
        )


# OpenAI-compatible endpoint (without /api prefix for compatibility)
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint (standard path)."""
    # Forward to the main handler
    return await chat_completions(request)


# ============================================================================
# System endpoints
# ============================================================================
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