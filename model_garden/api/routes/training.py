# Training job routes
"""
Routes for training job management:
- GET /api/v1/training/jobs - List training jobs
- POST /api/v1/training/jobs - Create a training job
- GET /api/v1/training/jobs/{job_id} - Get job details
- DELETE /api/v1/training/jobs/{job_id} - Cancel/delete a job
- POST /api/v1/training/jobs/{job_id}/stop - Request early stopping
- POST /api/v1/training/jobs/{job_id}/rerun - Rerun a job
- GET /api/v1/training/queue - Get queue status
- WebSocket /ws/training/{job_id} - Real-time updates
"""

import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)

from ..models import APIResponse, PaginatedResponse, TrainingJobInfo, TrainingJobRequest
from ..storage import get_storage_manager
from ..websocket import get_connection_manager


def utc_now() -> datetime:
    """Get the current UTC time."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Get the current UTC time as ISO 8601 string with Z suffix."""
    # Use replace to remove timezone info, then add Z suffix
    # This produces: 2025-12-03T15:16:07.239014Z (valid ISO 8601)
    return datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"


router = APIRouter(tags=["training"])


def resolve_path(path: str) -> str:
    """Resolve a path to an absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p.resolve())


def resolve_model_path(path: str) -> str:
    """Resolve a model path, handling simple names.

    If path is a simple name (no slashes), place it in the models directory.
    This ensures trained models are saved to ./models/<name> by default.
    """
    if "/" not in path and "\\" not in path:
        # Simple name - put in models directory
        models_dir = Path.cwd() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return str((models_dir / path).resolve())
    return resolve_path(path)


def create_training_job_record(
    job_id: str,
    job_request: TrainingJobRequest,
    dataset_path: str,
    validation_dataset_path: str | None,
    output_dir: str,
) -> TrainingJobInfo:
    """Create a TrainingJobInfo record from a request."""
    # Use nested hyperparameters and lora_config dicts from request
    hyperparams = job_request.hyperparameters or {}
    lora_cfg = job_request.lora_config or {}

    return TrainingJobInfo(
        id=job_id,
        name=job_request.name,
        status="queued",
        base_model=job_request.base_model,
        dataset_path=dataset_path,
        validation_dataset_path=validation_dataset_path,
        output_dir=output_dir,
        created_at=utc_now_iso(),
        hyperparameters=hyperparams,
        lora_config=lora_cfg,
        from_hub=job_request.from_hub,
        validation_from_hub=job_request.validation_from_hub,
        is_vision=job_request.is_vision,
        model_type="vision" if job_request.is_vision else "text",
        save_method=job_request.save_method,
        selective_loss=job_request.selective_loss,
        selective_loss_level=job_request.selective_loss_level,
        selective_loss_schema_keys=job_request.selective_loss_schema_keys,
        selective_loss_masking_strategy=job_request.selective_loss_masking_strategy,
        selective_loss_masking_start_epoch=job_request.selective_loss_masking_start_epoch,
        selective_loss_mask_every_n_steps=job_request.selective_loss_mask_every_n_steps,
        selective_loss_mask_for_n_steps=job_request.selective_loss_mask_for_n_steps,
        selective_loss_structural_weight=job_request.selective_loss_structural_weight,
        selective_loss_verbose=job_request.selective_loss_verbose,
        quality_mode=job_request.quality_mode,
        load_in_16bit=job_request.load_in_16bit,
        load_in_8bit=job_request.load_in_8bit,
        early_stopping_enabled=job_request.early_stopping_enabled,
        early_stopping_patience=job_request.early_stopping_patience,
        early_stopping_threshold=job_request.early_stopping_threshold,
    )


@router.get("/api/v1/training/jobs", response_model=PaginatedResponse)
async def list_training_jobs(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
):
    """List all training jobs."""
    storage = get_storage_manager()
    training_jobs = storage.load_training_jobs()

    # Filter jobs
    filtered_jobs = list(training_jobs.values())

    if status_filter:
        filtered_jobs = [j for j in filtered_jobs if j["status"] == status_filter]

    # Sort by created_at in descending order (newest first)
    filtered_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

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


@router.post("/api/v1/training/jobs", response_model=APIResponse)
async def create_training_job(job_request: TrainingJobRequest, background_tasks: BackgroundTasks):
    """Create a new training job."""
    from model_garden.queue import get_job_queue

    storage = get_storage_manager()
    training_jobs = storage.load_training_jobs()

    job_id = str(uuid.uuid4())

    # Only resolve paths for local files, not HuggingFace Hub datasets
    dataset_path = (
        job_request.dataset_path if job_request.from_hub else resolve_path(job_request.dataset_path)
    )

    # Resolve output directory for models
    output_dir = resolve_model_path(job_request.output_dir)

    # Handle validation dataset path
    validation_dataset_path = None
    if job_request.validation_dataset_path:
        validation_dataset_path = (
            job_request.validation_dataset_path
            if job_request.validation_from_hub
            else resolve_path(job_request.validation_dataset_path)
        )

    # Create job record
    job_info_model = create_training_job_record(
        job_id=job_id,
        job_request=job_request,
        dataset_path=dataset_path,
        validation_dataset_path=validation_dataset_path,
        output_dir=output_dir,
    )

    # Convert to dict for storage
    job_info = job_info_model.model_dump(mode="json", exclude_none=False)

    training_jobs[job_id] = job_info
    storage.save_training_jobs(training_jobs)

    # Add to job queue
    queue = get_job_queue()
    await queue.add_job(job_id=job_id, job_type="training", job_config=job_info, priority=0)

    # Get queue position
    position = await queue.get_queue_position(job_id)
    position_msg = f" (position in queue: {position})" if position and position > 1 else ""

    return APIResponse(
        success=True,
        data={"job_id": job_id, "queue_position": position},
        message=f"Training job {job_id} created and queued for execution{position_msg}",
    )


@router.get("/api/v1/training/jobs/{job_id}", response_model=TrainingJobInfo)
async def get_training_job(job_id: str):
    """Get details for a specific training job."""
    from model_garden.queue import get_job_queue

    storage = get_storage_manager()
    training_jobs = storage.load_training_jobs()

    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found"
        )

    job_data = training_jobs[job_id].copy()

    # Add queue information if job is queued
    if job_data["status"] == "queued":
        queue = get_job_queue()
        position = await queue.get_queue_position(job_id)
        if position:
            job_data["queue_position"] = position

    return TrainingJobInfo(**job_data)


@router.delete("/api/v1/training/jobs/{job_id}", response_model=APIResponse)
async def delete_or_cancel_training_job(job_id: str):
    """Delete or cancel a training job."""
    from model_garden.queue import get_job_queue

    from .. import cancellation_events

    storage = get_storage_manager()
    manager = get_connection_manager()
    training_jobs = storage.load_training_jobs()

    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found"
        )

    job = training_jobs[job_id]

    # If the job is running, signal the training thread to stop
    if job["status"] == "running":
        event = cancellation_events.get(job_id)
        if event:
            event.set()

    # If job is finished, delete it from the list
    if job["status"] in ["completed", "failed", "cancelled"]:
        del training_jobs[job_id]
        storage.save_training_jobs(training_jobs)

        # Notify WebSocket clients
        await manager.send_update(
            job_id,
            {
                "type": "job_deleted",
                "job_id": job_id,
                "timestamp": utc_now_iso(),
            },
        )

        return APIResponse(success=True, message=f"Training job {job_id} deleted successfully")

    # Try to cancel in queue
    queue = get_job_queue()
    cancelled_in_queue = await queue.cancel_job(job_id)

    # Mark as cancelled
    job["status"] = "cancelled"
    job["completed_at"] = utc_now_iso()
    storage.save_training_jobs(training_jobs)

    # Notify WebSocket clients
    await manager.send_update(
        job_id,
        {
            "type": "status_update",
            "job_id": job_id,
            "status": "cancelled",
            "completed_at": job["completed_at"],
            "timestamp": utc_now_iso(),
        },
    )

    status_msg = "cancelled from queue" if cancelled_in_queue else "cancellation requested"

    return APIResponse(success=True, message=f"Training job {job_id} {status_msg}")


@router.post("/api/v1/training/jobs/{job_id}/stop", response_model=APIResponse)
async def request_early_stop(job_id: str):
    """Request early stopping for a running training job."""
    storage = get_storage_manager()
    manager = get_connection_manager()
    training_jobs = storage.load_training_jobs()

    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found"
        )

    job = training_jobs[job_id]

    # Only allow early stopping for running jobs
    if job["status"] != "running":
        return APIResponse(
            success=False,
            message=f"Cannot request early stopping for job with status: {job['status']}",
        )

    # Set the early stop flag
    try:
        # Import global early stop requests map from api module
        import model_garden.api as api_module

        early_stop_map = getattr(api_module, "early_stop_requests", {})
        if not hasattr(api_module, "early_stop_requests"):
            api_module.early_stop_requests = {}
            early_stop_map = api_module.early_stop_requests
        early_stop_map[job_id] = True
        print(f"🛑 Early stopping requested for job {job_id}")

        # Notify WebSocket clients
        await manager.send_update(
            job_id,
            {
                "type": "early_stop_requested",
                "job_id": job_id,
                "message": "Early stopping requested - will stop at next evaluation",
                "timestamp": utc_now_iso(),
            },
        )

        return APIResponse(
            success=True, message="Early stopping requested - training will stop at next evaluation"
        )
    except Exception as e:
        return APIResponse(success=False, message=f"Failed to request early stopping: {str(e)}")


@router.post("/api/v1/training/jobs/{job_id}/rerun", response_model=APIResponse)
async def rerun_training_job(job_id: str, background_tasks: BackgroundTasks):
    """Rerun a past training job with the same configuration."""
    from model_garden.queue import get_job_queue

    storage = get_storage_manager()
    training_jobs = storage.load_training_jobs()

    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job {job_id} not found"
        )

    original_job = training_jobs[job_id]

    # Only allow rerunning completed, failed, or cancelled jobs
    if original_job["status"] in ["running", "queued"]:
        return APIResponse(
            success=False,
            message=f"Cannot rerun a job that is currently {original_job['status']}. Cancel it first.",
        )

    new_job_id = str(uuid.uuid4())

    # Generate new name with timestamp
    timestamp_suffix = utc_now().strftime("%Y%m%d_%H%M%S")
    new_job_name = f"{original_job['name']}_rerun_{timestamp_suffix}"

    # Resolve output directory with new name
    original_output = Path(original_job["output_dir"])
    new_output_dir = str(original_output.parent / new_job_name)

    # Clone job configuration
    new_job = {
        "id": new_job_id,
        "name": new_job_name,
        "status": "queued",
        "base_model": original_job["base_model"],
        "dataset_path": original_job["dataset_path"],
        "validation_dataset_path": original_job.get("validation_dataset_path"),
        "output_dir": new_output_dir,
        "created_at": utc_now_iso(),
        "started_at": None,
        "completed_at": None,
        "progress": {"current_step": 0, "total_steps": 0, "epoch": 0},
        "error_message": None,
        "hyperparameters": original_job.get("hyperparameters", {}).copy(),
        "lora_config": original_job.get("lora_config", {}).copy(),
        "from_hub": original_job.get("from_hub", False),
        "validation_from_hub": original_job.get("validation_from_hub", False),
        "is_vision": original_job.get("is_vision", False),
        "model_type": original_job.get("model_type", "text"),
        "save_method": original_job.get("save_method", "merged_16bit"),
        "metrics": {"training": [], "validation": []},
        # Clone selective loss settings
        "selective_loss": original_job.get("selective_loss", False),
        "selective_loss_level": original_job.get("selective_loss_level", "conservative"),
        "selective_loss_schema_keys": original_job.get("selective_loss_schema_keys"),
        "selective_loss_masking_strategy": original_job.get(
            "selective_loss_masking_strategy", "epoch_based"
        ),
        "selective_loss_masking_start_epoch": original_job.get(
            "selective_loss_masking_start_epoch", 0.0
        ),
        "selective_loss_mask_every_n_steps": original_job.get(
            "selective_loss_mask_every_n_steps", 100
        ),
        "selective_loss_mask_for_n_steps": original_job.get("selective_loss_mask_for_n_steps", 50),
        "selective_loss_structural_weight": original_job.get(
            "selective_loss_structural_weight", 0.1
        ),
        "selective_loss_verbose": original_job.get("selective_loss_verbose", False),
        # Clone quality settings
        "quality_mode": original_job.get("quality_mode", False),
        "load_in_16bit": original_job.get("load_in_16bit", False),
        "load_in_8bit": original_job.get("load_in_8bit", False),
        # Clone early stopping settings
        "early_stopping_enabled": original_job.get("early_stopping_enabled", False),
        "early_stopping_patience": original_job.get("early_stopping_patience", 3),
        "early_stopping_threshold": original_job.get("early_stopping_threshold", 0.0),
        # Clone backend setting
        "backend": original_job.get("backend", "unsloth"),
        # Metadata
        "rerun_from": job_id,
        "rerun_from_name": original_job["name"],
    }

    training_jobs[new_job_id] = new_job
    storage.save_training_jobs(training_jobs)

    # Add to job queue
    queue = get_job_queue()
    await queue.add_job(job_id=new_job_id, job_type="training", job_config=new_job, priority=0)

    position = await queue.get_queue_position(new_job_id)
    position_msg = f" (position in queue: {position})" if position and position > 1 else ""

    return APIResponse(
        success=True,
        data={
            "job_id": new_job_id,
            "original_job_id": job_id,
            "queue_position": position,
            "name": new_job_name,
        },
        message=f"Training job rerun created and queued for execution{position_msg}",
    )


@router.get("/api/v1/training/queue", response_model=APIResponse)
async def get_training_queue():
    """Get current training job queue status."""
    from model_garden.queue import JobStatus, get_job_queue

    queue = get_job_queue()

    # Get queued and running jobs
    queued_jobs = await queue.list_jobs(status=JobStatus.QUEUED, job_type="training")
    running_jobs = await queue.list_jobs(status=JobStatus.RUNNING, job_type="training")

    return APIResponse(
        success=True,
        data={
            "queued": len(queued_jobs),
            "running": len(running_jobs),
            "queued_jobs": [
                {
                    "job_id": j["job_id"],
                    "name": j["job_config"].get("name", "Unnamed"),
                    "position": i + 1,
                    "queued_at": j["queued_at"],
                    "priority": j["priority"],
                }
                for i, j in enumerate(queued_jobs)
            ],
            "running_jobs": [
                {
                    "job_id": j["job_id"],
                    "name": j["job_config"].get("name", "Unnamed"),
                    "started_at": j["started_at"],
                    "status_message": j["status_message"],
                }
                for j in running_jobs
            ],
        },
        message=f"{len(queued_jobs)} jobs queued, {len(running_jobs)} running",
    )


@router.websocket("/ws/training/{job_id}")
async def websocket_training_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training job updates."""
    manager = get_connection_manager()
    storage = get_storage_manager()
    training_jobs = storage.load_training_jobs()

    await manager.connect(websocket, job_id)

    try:
        # Send initial job status
        if job_id in training_jobs:
            await websocket.send_json(
                {
                    "type": "initial_state",
                    "job": training_jobs[job_id],
                    "timestamp": utc_now_iso(),
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Training job {job_id} not found",
                    "timestamp": utc_now_iso(),
                }
            )
            await websocket.close()
            return

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": utc_now_iso()})
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break

    finally:
        manager.disconnect(websocket, job_id)
