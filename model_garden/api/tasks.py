# Background tasks for API operations
"""
Background task functions for:
- run_training_job: Execute training jobs
- run_model_loading: Load models for inference

These tasks are run in the background by FastAPI's BackgroundTasks
or by the job queue worker.

Training jobs now use TrainingService for consistency with CLI,
eliminating duplicated training logic.
"""

import asyncio
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_garden.services import TrainingRequest, TrainingService

from .storage import get_storage_manager
from .websocket import get_connection_manager


def utc_now() -> datetime:
    """Get the current UTC time."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Get the current UTC time as ISO 8601 string with Z suffix."""
    # Use replace to remove timezone info, then add Z suffix
    # This produces: 2025-12-03T15:16:07.239014Z (valid ISO 8601)
    return datetime.now(UTC).replace(tzinfo=None).isoformat() + "Z"


def _run_async_in_thread(coro):
    """Helper to run async code in a sync context (thread or main).

    When run_training_job is called from a thread (by the queue worker),
    we need to create a new event loop for that thread.
    """
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def calculate_dir_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    if path.exists():
        for file in path.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
    return total


async def run_model_loading(
    job_id: str,
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int,
    enforce_eager: bool,
    limit_mm_per_prompt: dict[str, int] | None,
    dtype: str,
    quantization: str | None,
):
    """Execute model loading in the background."""
    from model_garden.inference import InferenceService, set_inference_service
    from model_garden.queue import get_job_queue

    queue = get_job_queue()

    try:
        await queue.start_job(job_id)

        print(f"🔄 Loading model: {model_path}")

        service = InferenceService(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            dtype=dtype,
            quantization=quantization,
        )

        await service.load_model()
        set_inference_service(service)

        # Initialize carbon tracking
        try:
            from model_garden.carbon import init_inference_tracker

            init_inference_tracker(model_path)
            print(f"✅ Carbon tracking initialized for {model_path}")
        except Exception as e:
            print(f"⚠️  Failed to initialize carbon tracking: {e}")

        model_info = service.get_model_info()
        await queue.complete_job(job_id, result=model_info)

        print(f"✅ Model loaded successfully: {model_path}")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Model loading failed: {error_msg}")
        await queue.fail_job(job_id, error=error_msg)
        raise


# Global maps for job management (accessed by callbacks)
cancellation_events: dict[str, threading.Event] = {}
early_stop_requests: dict[str, bool] = {}


def create_progress_callback(job_id: str, manager: Any):
    """Create a progress callback for training.

    This callback:
    - Updates job progress in storage
    - Sends WebSocket updates
    - Handles cancellation and early stopping
    """
    from transformers import TrainerCallback

    storage = get_storage_manager()

    class ProgressCallback(TrainerCallback):
        def __init__(self, job_id: str, manager: Any):
            self.job_id = job_id
            self.manager = manager
            self.training_metrics: list[dict] = []
            self.validation_metrics: list[dict] = []
            self.cancellation_event: threading.Event | None = None

            # ETA calculation state
            self.start_time = None
            self.last_step_time = None
            self.steps_per_second_ema = 0.0
            self.ema_alpha = 0.1

        def on_train_begin(self, args, state, control, **kwargs):
            """Called at the beginning of training."""
            # Initialize metrics
            training_jobs = storage.load_training_jobs()
            if self.job_id in training_jobs:
                training_jobs[self.job_id]["metrics"] = {"training": [], "validation": []}
                storage.save_training_jobs(training_jobs)

            # Initialize timing
            self.start_time = time.time()
            self.last_step_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            """Called at the end of each training step."""
            # Check for cancellation
            if self.cancellation_event is not None and self.cancellation_event.is_set():
                print(f"✋ Cancellation requested for job {self.job_id} - stopping training")
                raise KeyboardInterrupt()

            # Update progress
            current_step = state.global_step
            total_steps = (
                state.max_steps
                if state.max_steps > 0
                else args.num_train_epochs * state.num_train_epochs
            )
            epoch = state.epoch or 0

            # Calculate ETA
            current_time = time.time()
            # Initialize last_step_time if not set (e.g. if on_train_begin wasn't called or resumed)
            if self.last_step_time is None:
                self.last_step_time = current_time

            step_time = current_time - self.last_step_time
            self.last_step_time = current_time

            # Avoid division by zero
            if step_time > 0:
                current_steps_per_sec = 1.0 / step_time
                if self.steps_per_second_ema == 0:
                    self.steps_per_second_ema = current_steps_per_sec
                else:
                    self.steps_per_second_ema = (
                        self.ema_alpha * current_steps_per_sec
                        + (1 - self.ema_alpha) * self.steps_per_second_ema
                    )

            remaining_steps = total_steps - current_step
            eta_seconds = (
                remaining_steps / self.steps_per_second_ema if self.steps_per_second_ema > 0 else 0
            )

            training_jobs = storage.load_training_jobs()
            if self.job_id in training_jobs:
                training_jobs[self.job_id]["progress"] = {
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "epoch": epoch,
                    "percentage": (current_step / total_steps * 100) if total_steps > 0 else 0,
                    "eta_seconds": eta_seconds,
                    "steps_per_second": self.steps_per_second_ema,
                }
                training_jobs[self.job_id]["current_step"] = current_step
                training_jobs[self.job_id]["total_steps"] = total_steps
                training_jobs[self.job_id]["current_epoch"] = epoch
                storage.save_training_jobs(training_jobs)

            # Send progress update via WebSocket
            _run_async_in_thread(
                self.manager.send_update(
                    self.job_id,
                    {
                        "type": "progress",
                        "job_id": self.job_id,
                        "current_step": current_step,
                        "total_steps": total_steps,
                        "epoch": epoch,
                        "percentage": (current_step / total_steps * 100) if total_steps > 0 else 0,
                        "eta_seconds": eta_seconds,
                        "steps_per_second": self.steps_per_second_ema,
                        "timestamp": utc_now_iso(),
                    },
                )
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called when logging occurs."""
            # Check for cancellation
            if self.cancellation_event is not None and self.cancellation_event.is_set():
                print(f"✋ Cancellation requested for job {self.job_id} during logging")
                raise KeyboardInterrupt()

            # Check for early stopping request
            if early_stop_requests.get(self.job_id, False):
                print(f"🛑 Early stopping requested for job {self.job_id}")
                control.should_training_stop = True
                del early_stop_requests[self.job_id]

            if logs:
                current_step = state.global_step
                timestamp = utc_now_iso()

                is_eval = any(k.startswith("eval_") for k in logs.keys())

                training_jobs = storage.load_training_jobs()

                if is_eval:
                    # Validation metrics
                    eval_loss = logs.get("eval_loss")
                    metric_point = {
                        "step": current_step,
                        "loss": eval_loss,
                        "timestamp": timestamp,
                    }

                    for key, value in logs.items():
                        if key.startswith("eval_") and key != "eval_loss":
                            metric_point[key.replace("eval_", "")] = value

                    self.validation_metrics.append(metric_point)

                    if self.job_id in training_jobs:
                        if "metrics" not in training_jobs[self.job_id]:
                            training_jobs[self.job_id]["metrics"] = {}
                        training_jobs[self.job_id]["metrics"]["validation"] = (
                            self.validation_metrics
                        )
                        storage.save_training_jobs(training_jobs)

                    _run_async_in_thread(
                        self.manager.send_update(
                            self.job_id,
                            {
                                "type": "validation_metrics",
                                "job_id": self.job_id,
                                "metrics": metric_point,
                                "timestamp": timestamp,
                            },
                        )
                    )
                else:
                    # Training metrics
                    train_loss = logs.get("loss")
                    learning_rate = logs.get("learning_rate")

                    if train_loss is not None:
                        metric_point = {
                            "step": current_step,
                            "loss": train_loss,
                            "learning_rate": learning_rate,
                            "timestamp": timestamp,
                        }

                        for key, value in logs.items():
                            if key not in ["loss", "learning_rate", "epoch"]:
                                metric_point[key] = value

                        self.training_metrics.append(metric_point)

                        if self.job_id in training_jobs:
                            if "metrics" not in training_jobs[self.job_id]:
                                training_jobs[self.job_id]["metrics"] = {}
                            training_jobs[self.job_id]["metrics"]["training"] = (
                                self.training_metrics
                            )
                            storage.save_training_jobs(training_jobs)

                        _run_async_in_thread(
                            self.manager.send_update(
                                self.job_id,
                                {
                                    "type": "training_metrics",
                                    "job_id": self.job_id,
                                    "metrics": metric_point,
                                    "timestamp": timestamp,
                                },
                            )
                        )

                # Send formatted log message
                log_message = " | ".join(
                    [
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in logs.items()
                        if k != "epoch"
                    ]
                )

                _run_async_in_thread(
                    self.manager.send_update(
                        self.job_id,
                        {
                            "type": "log",
                            "job_id": self.job_id,
                            "message": log_message,
                            "timestamp": timestamp,
                        },
                    )
                )

    return ProgressCallback(job_id, manager)


def run_training_job(job_id: str):
    """Execute a training job in the background.

    This is the main training entry point called by the job queue worker.
    Now uses TrainingService for consistent behavior with CLI.
    """
    from model_garden.queue import get_job_queue

    storage = get_storage_manager()
    manager = get_connection_manager()

    try:
        queue = get_job_queue()
        _run_async_in_thread(queue.start_job(job_id))

        # Register cancellation event
        cancellation_events[job_id] = threading.Event()

        training_jobs = storage.load_training_jobs()

        if job_id not in training_jobs:
            print(f"⚠️  Job {job_id} not found in training_jobs")
            _run_async_in_thread(queue.fail_job(job_id, "Job configuration not found"))
            return

        job = training_jobs[job_id]

        # Update status to running
        job["status"] = "running"
        job["started_at"] = utc_now_iso()
        storage.save_training_jobs(training_jobs)

        # Notify WebSocket clients
        _run_async_in_thread(
            manager.send_update(
                job_id,
                {
                    "type": "status_update",
                    "job_id": job_id,
                    "status": "running",
                    "started_at": job["started_at"],
                    "timestamp": utc_now_iso(),
                },
            )
        )

        print(f"🚀 Starting training job {job_id}: {job['name']}")

        # Build TrainingRequest from job config - single source of truth
        request = TrainingRequest.from_dict(job)
        request.job_id = job_id

        # Set up warning callback for WebSocket notifications
        def send_warning_to_ui(message: str):
            _run_async_in_thread(
                manager.send_update(
                    job_id,
                    {
                        "type": "warning",
                        "job_id": job_id,
                        "message": message,
                        "timestamp": utc_now_iso(),
                    },
                )
            )

        request.warning_callback = send_warning_to_ui

        # Build progress callback for WebSocket updates
        progress_callback = create_progress_callback(job_id, manager)
        progress_callback.cancellation_event = cancellation_events.get(job_id)

        # Execute training through unified service
        service = TrainingService()
        result = service.train(request, callbacks=[progress_callback])

        if not result.success:
            raise RuntimeError(result.error or "Training failed")

        # Update status to completed
        training_jobs = storage.load_training_jobs()
        job = training_jobs[job_id]
        job["status"] = "completed"
        job["completed_at"] = utc_now_iso()

        # Update job with dataset stats from result metrics
        if result.metrics:
            if "dataset_size" in result.metrics and result.metrics["dataset_size"]:
                job["dataset_size"] = result.metrics["dataset_size"]
            if "dataset_num_samples" in result.metrics and result.metrics["dataset_num_samples"]:
                job["dataset_num_samples"] = result.metrics["dataset_num_samples"]

        storage.save_training_jobs(training_jobs)

        # Notify completion
        _run_async_in_thread(
            manager.send_update(
                job_id,
                {
                    "type": "status_update",
                    "job_id": job_id,
                    "status": "completed",
                    "completed_at": job["completed_at"],
                    "timestamp": utc_now_iso(),
                },
            )
        )

        # Register model
        models_storage = storage.load_models()
        model_id = Path(job["output_dir"]).name
        models_storage[model_id] = {
            "id": model_id,
            "name": job["name"],
            "base_model": job["base_model"],
            "status": "available",
            "created_at": job["created_at"],
            "updated_at": utc_now_iso(),
            "path": job["output_dir"],
            "training_job_id": job_id,
            "size_bytes": calculate_dir_size(Path(job["output_dir"])),
        }
        storage.save_models(models_storage)

        # Complete in queue
        _run_async_in_thread(
            queue.complete_job(
                job_id, result={"model_id": model_id, "output_dir": job["output_dir"]}
            )
        )

        print(f"✅ Training job {job_id} completed successfully!")

    except KeyboardInterrupt:
        _handle_job_cancellation(job_id, storage, manager)

    except Exception as e:
        _handle_job_failure(job_id, str(e), storage, manager)
        import traceback

        traceback.print_exc()

    finally:
        # Clean up cancellation event
        if job_id in cancellation_events:
            del cancellation_events[job_id]


def _handle_job_cancellation(job_id: str, storage, manager):
    """Handle job cancellation."""
    from model_garden.queue import get_job_queue

    print(f"✋ Training job {job_id} cancelled by user")

    training_jobs = storage.load_training_jobs()
    if job_id in training_jobs:
        training_jobs[job_id]["status"] = "cancelled"
        training_jobs[job_id]["completed_at"] = utc_now_iso()
        training_jobs[job_id]["error_message"] = "Training cancelled by user"
        storage.save_training_jobs(training_jobs)

        queue = get_job_queue()
        _run_async_in_thread(queue.cancel_job(job_id))

        _run_async_in_thread(
            manager.send_update(
                job_id,
                {
                    "type": "status_update",
                    "job_id": job_id,
                    "status": "cancelled",
                    "timestamp": utc_now_iso(),
                },
            )
        )


def _handle_job_failure(job_id: str, error_message: str, storage, manager):
    """Handle job failure."""
    from model_garden.queue import get_job_queue

    print(f"❌ Training job {job_id} failed: {error_message}")

    training_jobs = storage.load_training_jobs()
    if job_id in training_jobs:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = utc_now_iso()
        training_jobs[job_id]["error_message"] = error_message
        storage.save_training_jobs(training_jobs)

        queue = get_job_queue()
        _run_async_in_thread(queue.fail_job(job_id, error_message))

        _run_async_in_thread(
            manager.send_update(
                job_id,
                {
                    "type": "status_update",
                    "job_id": job_id,
                    "status": "failed",
                    "error_message": error_message,
                    "timestamp": utc_now_iso(),
                },
            )
        )
