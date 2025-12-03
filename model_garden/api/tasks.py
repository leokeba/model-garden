# Background tasks for API operations
"""
Background task functions for:
- run_training_job: Execute training jobs
- run_model_loading: Load models for inference

These tasks are run in the background by FastAPI's BackgroundTasks
or by the job queue worker.
"""

import asyncio
import gc
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

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


def cleanup_training_resources(*objects):
    """Aggressively free memory after training completes.

    This function attempts to:
    1. Delete Python objects (model, tokenizer, trainer, datasets)
    2. Force garbage collection
    3. Clear CUDA cache and synchronize
    4. Run multiple GC passes to break circular references

    Args:
        *objects: Variable arguments of objects to delete/cleanup
    """

    print("🧹 Cleaning up training resources...")

    # First pass: delete and dereference objects
    for obj in objects:
        try:
            if obj is not None:
                # Try to move model to CPU first to free GPU memory
                if hasattr(obj, "to"):
                    try:
                        obj.to("cpu")
                    except Exception:
                        pass

                # Delete references held by trainer
                if hasattr(obj, "model"):
                    try:
                        obj.model = None
                    except Exception:
                        pass
                if hasattr(obj, "tokenizer"):
                    try:
                        obj.tokenizer = None
                    except Exception:
                        pass
                if hasattr(obj, "optimizer"):
                    try:
                        obj.optimizer = None
                    except Exception:
                        pass
                if hasattr(obj, "lr_scheduler"):
                    try:
                        obj.lr_scheduler = None
                    except Exception:
                        pass

                del obj
        except Exception as e:
            print(f"⚠️  Warning during cleanup: {e}")

    # Multiple GC passes to break circular references
    for _ in range(5):
        gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        # Move all tensors to CPU to free GPU memory
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Additional synchronization
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # Final GC passes
    for _ in range(3):
        gc.collect()

    # Report memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(
            f"🧹 Cleanup complete. GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
    else:
        print("🧹 Cleanup complete.")


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

        def on_train_begin(self, args, state, control, **kwargs):
            """Called at the beginning of training."""
            # Initialize metrics
            training_jobs = storage.load_training_jobs()
            if self.job_id in training_jobs:
                training_jobs[self.job_id]["metrics"] = {"training": [], "validation": []}
                storage.save_training_jobs(training_jobs)

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

            training_jobs = storage.load_training_jobs()
            if self.job_id in training_jobs:
                training_jobs[self.job_id]["progress"] = {
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "epoch": epoch,
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

        # Training configuration
        is_vision = job.get("is_vision", False)
        from_hub = job.get("from_hub", False)
        validation_from_hub = job.get("validation_from_hub", False)
        validation_dataset_path = job.get("validation_dataset_path")

        # Quality mode settings
        quality_mode = job.get("quality_mode", False)
        load_in_16bit = job.get("load_in_16bit", False)
        load_in_8bit = job.get("load_in_8bit", False)

        if quality_mode:
            print("🎯 Quality mode enabled - using higher precision settings")
            load_in_16bit = True
            load_in_8bit = False
            lora_config = job.get("lora_config", {})
            if lora_config.get("use_gradient_checkpointing") == "unsloth":
                lora_config["use_gradient_checkpointing"] = True
            hyperparams = job.get("hyperparameters", {})
            if hyperparams.get("optim") == "adamw_8bit":
                hyperparams["optim"] = "adamw_torch"
            if lora_config.get("r", 16) >= 32 and not lora_config.get("use_rslora"):
                lora_config["use_rslora"] = True
            job["lora_config"] = lora_config
            job["hyperparameters"] = hyperparams

        load_in_4bit = not (load_in_16bit or load_in_8bit)
        backend = job.get("backend", "unsloth")

        # Execute training based on model type
        if is_vision:
            _run_vision_training(
                job,
                job_id,
                load_in_4bit,
                load_in_8bit,
                backend,
                from_hub,
                validation_from_hub,
                validation_dataset_path,
                storage,
                manager,
            )
        else:
            _run_text_training(
                job,
                job_id,
                load_in_4bit,
                load_in_8bit,
                backend,
                from_hub,
                validation_from_hub,
                validation_dataset_path,
                storage,
                manager,
            )

        # Update status to completed
        training_jobs = storage.load_training_jobs()
        job = training_jobs[job_id]
        job["status"] = "completed"
        job["completed_at"] = utc_now_iso()
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


def _run_vision_training(
    job,
    job_id,
    load_in_4bit,
    load_in_8bit,
    backend,
    from_hub,
    validation_from_hub,
    validation_dataset_path,
    storage,
    manager,
):
    """Execute vision model training."""
    from model_garden.training import create_vision_trainer

    print(f"🎨 Using VisionLanguageTrainer for {job['base_model']}")

    trainer = create_vision_trainer(
        base_model=job["base_model"],
        max_seq_length=job["hyperparameters"].get("max_seq_length", 16384),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        backend=backend,
    )

    # Set up warning callback to send warnings to WebSocket/UI
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

    trainer.warning_callback = send_warning_to_ui

    trainer.load_model()

    lora_config = job["lora_config"]
    trainer.prepare_for_training(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.0),
        lora_bias=lora_config.get("lora_bias", "none"),
        use_rslora=lora_config.get("use_rslora", False),
        use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_config.get("random_state", 42),
        loftq_config=lora_config.get("loftq_config"),
        finetune_vision_layers=lora_config.get("finetune_vision_layers", True),
        finetune_language_layers=lora_config.get("finetune_language_layers", True),
        finetune_attention_modules=lora_config.get("finetune_attention_modules", True),
        finetune_mlp_modules=lora_config.get("finetune_mlp_modules", True),
    )

    # Load datasets
    train_dataset = trainer.load_dataset(
        dataset_path=job["dataset_path"],
        from_hub=from_hub,
        split="train",
    )
    formatted_train_dataset = trainer.format_dataset(train_dataset)

    formatted_val_dataset = None
    if validation_dataset_path:
        val_dataset = trainer.load_dataset(
            dataset_path=validation_dataset_path,
            from_hub=validation_from_hub,
            split="validation",
        )
        formatted_val_dataset = trainer.format_dataset(val_dataset)

    # Build callbacks
    progress_callback = create_progress_callback(job_id, manager)
    progress_callback.cancellation_event = cancellation_events.get(job_id)

    callbacks = [progress_callback]

    if job.get("early_stopping_enabled", False):
        from model_garden.training import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                patience=job.get("early_stopping_patience", 3),
                threshold=job.get("early_stopping_threshold", 0.0),
            )
        )

    # Train
    hyperparams = job["hyperparameters"]
    trainer.train(
        dataset=formatted_train_dataset,
        eval_dataset=formatted_val_dataset,
        eval_steps=hyperparams.get("eval_steps"),
        output_dir=job["output_dir"],
        num_train_epochs=hyperparams.get("num_epochs", 3),
        per_device_train_batch_size=hyperparams.get("batch_size", 1),
        gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 8),
        learning_rate=hyperparams.get("learning_rate", 2e-5),
        warmup_steps=hyperparams.get("warmup_steps", 10),
        max_steps=hyperparams.get("max_steps", -1),
        logging_steps=hyperparams.get("logging_steps", 10),
        save_steps=hyperparams.get("save_steps", 100),
        optim=hyperparams.get("optim", "adamw_8bit"),
        weight_decay=hyperparams.get("weight_decay", 0.01),
        lr_scheduler_type=hyperparams.get("lr_scheduler_type", "cosine"),
        callbacks=callbacks,
        selective_loss=job.get("selective_loss", False),
        selective_loss_level=job.get("selective_loss_level", "conservative"),
    )

    # Save
    save_method = job.get("save_method", "merged_16bit")
    trainer.save_model(job["output_dir"], save_method=save_method)

    # Cleanup
    cleanup_training_resources(
        trainer.model,
        trainer.tokenizer,
        trainer.processor,
        trainer,
        formatted_train_dataset,
        formatted_val_dataset,
        train_dataset,
        progress_callback,
    )


def _run_text_training(
    job,
    job_id,
    load_in_4bit,
    load_in_8bit,
    backend,
    from_hub,
    validation_from_hub,
    validation_dataset_path,
    storage,
    manager,
):
    """Execute text model training."""
    from model_garden.training import create_text_trainer

    trainer = create_text_trainer(
        base_model=job["base_model"],
        max_seq_length=job["hyperparameters"].get("max_seq_length", 2048),
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        backend=backend,
    )

    trainer.load_model()

    lora_config = job["lora_config"]
    trainer.prepare_for_training(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 16),
        lora_dropout=lora_config.get("lora_dropout", 0.0),
        lora_bias=lora_config.get("lora_bias", "none"),
        use_rslora=lora_config.get("use_rslora", False),
        use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_config.get("random_state", 42),
        loftq_config=lora_config.get("loftq_config"),
    )

    # Load datasets
    if from_hub:
        train_dataset = trainer.load_dataset_from_hub(job["dataset_path"], split="train")
    else:
        train_dataset = trainer.load_dataset_from_file(job["dataset_path"])

    train_dataset = trainer.format_dataset(
        train_dataset,
        instruction_field=job["hyperparameters"].get("instruction_field", "instruction"),
        input_field=job["hyperparameters"].get("input_field", "input"),
        output_field=job["hyperparameters"].get("output_field", "output"),
    )

    val_dataset = None
    if validation_dataset_path:
        if validation_from_hub:
            val_dataset = trainer.load_dataset_from_hub(validation_dataset_path, split="validation")
        else:
            val_dataset = trainer.load_dataset_from_file(validation_dataset_path)
        val_dataset = trainer.format_dataset(
            val_dataset,
            instruction_field=job["hyperparameters"].get("instruction_field", "instruction"),
            input_field=job["hyperparameters"].get("input_field", "input"),
            output_field=job["hyperparameters"].get("output_field", "output"),
        )

    # Build callbacks
    progress_callback = create_progress_callback(job_id, manager)
    progress_callback.cancellation_event = cancellation_events.get(job_id)

    callbacks = [progress_callback]

    if job.get("early_stopping_enabled", False):
        from model_garden.training import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                patience=job.get("early_stopping_patience", 3),
                threshold=job.get("early_stopping_threshold", 0.0),
            )
        )

    # Train
    hyperparams = job["hyperparameters"]
    trainer.train(
        dataset=train_dataset,
        eval_dataset=val_dataset,
        eval_steps=hyperparams.get("eval_steps"),
        output_dir=job["output_dir"],
        num_train_epochs=hyperparams.get("num_epochs", 3),
        per_device_train_batch_size=hyperparams.get("batch_size", 2),
        gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
        learning_rate=hyperparams.get("learning_rate", 2e-4),
        warmup_steps=hyperparams.get("warmup_steps", 10),
        max_steps=hyperparams.get("max_steps", -1),
        logging_steps=hyperparams.get("logging_steps", 10),
        save_steps=hyperparams.get("save_steps", 100),
        optim=hyperparams.get("optim", "adamw_8bit"),
        weight_decay=hyperparams.get("weight_decay", 0.01),
        lr_scheduler_type=hyperparams.get("lr_scheduler_type", "linear"),
        callbacks=callbacks,
    )

    # Save
    save_method = job["hyperparameters"].get("save_method", "merged_16bit")
    if save_method != "lora":
        trainer.save_model(job["output_dir"], save_method=save_method)

    # Cleanup
    cleanup_training_resources(
        trainer.model, trainer.tokenizer, trainer, train_dataset, val_dataset, progress_callback
    )


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
