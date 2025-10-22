"""FastAPI backend for Model Garden."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from model_garden.job_queue import get_job_queue, JobStatus

# Load environment variables from .env file FIRST
load_dotenv()

# Configure PyTorch to use fewer compile workers to reduce memory usage
# Default is CPU count (often 24-32), which uses ~16GB RAM for workers alone!
# We set to 4 workers which is enough for most workloads and uses ~2GB RAM
if "TORCH_COMPILE_MAX_WORKERS" not in os.environ:
    os.environ["TORCH_COMPILE_MAX_WORKERS"] = "4"

# Also set worker timeout to clean up idle workers faster
if "TORCH_COMPILE_WORKER_TIMEOUT" not in os.environ:
    os.environ["TORCH_COMPILE_WORKER_TIMEOUT"] = "300"  # 5 minutes
# Also set worker timeout to clean up idle workers faster
if "TORCH_COMPILE_WORKER_TIMEOUT" not in os.environ:
    os.environ["TORCH_COMPILE_WORKER_TIMEOUT"] = "300"  # 5 minutes

from fastapi import BackgroundTasks, FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
import threading
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# IMPORTANT: Don't import transformers here - it initializes tokenizers which causes
# fork warnings when using background tasks. Import lazily in functions that need it.
# from transformers import TrainerCallback  # REMOVED - now lazy-loaded

# Memory management utilities
from model_garden.memory_management import cleanup_training_resources

# Don't import training modules at module level - they import unsloth which spawns workers!
# Import them lazily in the functions that need them (run_training_job)
# from model_garden.training import ModelTrainer  # REMOVED - now lazy-loaded


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
    validation_dataset_path: Optional[str] = None  # Optional validation dataset
    output_dir: str
    hyperparameters: Optional[Dict] = None
    lora_config: Optional[Dict] = None
    from_hub: bool = False
    validation_from_hub: bool = False  # Separate flag for validation dataset
    is_vision: bool = False  # Flag for vision-language models
    model_type: Optional[str] = None  # 'text' or 'vision'
    save_method: str = "merged_16bit"  # How to save: 'lora', 'merged_16bit', 'merged_4bit'
    selective_loss: bool = False  # Enable selective loss for structured outputs
    selective_loss_level: str = "conservative"  # Level: conservative, moderate, aggressive
    selective_loss_schema_keys: Optional[List[str]] = None  # Schema keys to mask
    selective_loss_masking_start_step: int = 0  # Delay masking until this step (0=immediate, legacy)
    selective_loss_masking_start_epoch: float = 0.0  # Delay masking until this epoch (0.0=immediate, more robust)
    selective_loss_verbose: bool = False  # Print masking statistics
    # Early stopping
    early_stopping_enabled: bool = False  # Enable early stopping
    early_stopping_patience: int = 3  # Number of evals with no improvement before stopping
    early_stopping_threshold: float = 0.0  # Minimum improvement to count
    # Quality settings
    quality_mode: bool = False  # Enable quality-optimized settings (16-bit, better optimizer, etc.)
    load_in_16bit: bool = False  # Load model in 16-bit precision (better quality, 4x more memory)
    load_in_8bit: bool = False  # Load model in 8-bit precision (balanced quality/memory)


class TrainingJobInfo(BaseModel):
    """Training job information."""
    id: str
    name: str
    status: str
    base_model: str
    dataset_path: str
    validation_dataset_path: Optional[str] = None
    output_dir: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict] = None
    error_message: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    lora_config: Optional[Dict] = None
    from_hub: Optional[bool] = False
    validation_from_hub: Optional[bool] = False
    is_vision: Optional[bool] = False
    model_type: Optional[str] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    current_epoch: Optional[int] = None
    save_method: Optional[str] = "merged_16bit"
    metrics: Optional[Dict] = None  # Training and validation metrics history


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


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Configure storage directories from environment variables
HF_HOME = os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface'))
MODELS_DIR = os.getenv('MODELS_DIR', str(PROJECT_ROOT / 'models'))

# Set HuggingFace cache environment variables
# These must be set before importing any HF libraries
os.environ['HF_HOME'] = HF_HOME
os.environ['TRANSFORMERS_CACHE'] = str(Path(HF_HOME) / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path(HF_HOME) / 'datasets')

# Ensure directories exist
Path(HF_HOME).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

print(f"📁 HuggingFace cache: {HF_HOME}")
print(f"📁 Models directory: {MODELS_DIR}")

# Storage manager for persistent data
class StorageManager:
    """Manages persistent storage of training jobs and models."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_file = storage_dir / "training_jobs.json"
        self.models_file = storage_dir / "models.json"
    
    def load_training_jobs(self) -> Dict[str, Dict]:
        """Load training jobs from disk."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, 'r') as f:
                    data = json.load(f)
                    print(f"✓ Loaded {len(data)} training jobs from {self.jobs_file}")
                    return data
            except Exception as e:
                print(f"⚠️  Error loading training jobs: {e}")
                import traceback
                traceback.print_exc()
                return {}
        else:
            print(f"ℹ️  No training jobs file found at {self.jobs_file}")
        return {}
    
    def save_training_jobs(self, jobs: Dict[str, Dict]) -> None:
        """Save training jobs to disk."""
        try:
            with open(self.jobs_file, 'w') as f:
                json.dump(jobs, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving training jobs: {e}")
    
    def load_models(self) -> Dict[str, Dict]:
        """Load models from disk."""
        if self.models_file.exists():
            try:
                with open(self.models_file, 'r') as f:
                    data = json.load(f)
                    print(f"✓ Loaded {len(data)} models from {self.models_file}")
                    return data
            except Exception as e:
                print(f"⚠️  Error loading models: {e}")
                import traceback
                traceback.print_exc()
                return {}
        else:
            print(f"ℹ️  No models file found at {self.models_file}")
        return {}
    
    def save_models(self, models: Dict[str, Dict]) -> None:
        """Save models to disk."""
        try:
            with open(self.models_file, 'w') as f:
                json.dump(models, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving models: {e}")

# Initialize storage manager
storage_manager = StorageManager(PROJECT_ROOT / "storage")

# Global variables for managing state (loaded from disk)
training_jobs: Dict[str, Dict] = storage_manager.load_training_jobs()
models_storage: Dict[str, Dict] = storage_manager.load_models()

print(f"📊 Loaded {len(training_jobs)} training jobs from storage")
print(f"📊 Loaded {len(models_storage)} models from storage")


def resolve_path(path_str: str, is_model_dir: bool = False) -> str:
    """Resolve a path relative to the project root if it's not absolute.
    
    Args:
        path_str: Path string (can be relative or absolute)
        is_model_dir: If True, resolve relative to MODELS_DIR instead of PROJECT_ROOT
        
    Returns:
        Absolute path string
    """
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    else:
        # Resolve relative to MODELS_DIR for model outputs, otherwise PROJECT_ROOT
        base_dir = Path(MODELS_DIR) if is_model_dir else PROJECT_ROOT
        resolved = (base_dir / path).resolve()
        return str(resolved)


def resolve_model_path(path_str: str) -> str:
    """Resolve a model path, handling both simple names and paths.
    
    This function handles:
    - Simple model names: "my-model" -> MODELS_DIR/my-model
    - Paths with ./models/ prefix: "./models/my-model" -> MODELS_DIR/my-model
    - Absolute paths: "/path/to/model" -> /path/to/model
    - HuggingFace IDs: "org/model" -> org/model (unchanged)
    
    Args:
        path_str: Model path or name
        
    Returns:
        Resolved path string
    """
    # If it's an absolute path, return as-is
    if Path(path_str).is_absolute():
        return path_str
    
    # If it looks like a HuggingFace model ID (contains / but not at start), return as-is
    if '/' in path_str and not path_str.startswith(('./', '../', '/')):
        return path_str
    
    # Strip ./models/ or models/ prefix if present
    cleaned_path = path_str
    if cleaned_path.startswith('./models/'):
        cleaned_path = cleaned_path[len('./models/'):]
    elif cleaned_path.startswith('models/'):
        cleaned_path = cleaned_path[len('models/'):]
    
    # Resolve relative to MODELS_DIR
    return str((Path(MODELS_DIR) / cleaned_path).resolve())


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        print(f"✓ WebSocket connected for job {job_id} (total: {len(self.active_connections[job_id])})")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Remove a WebSocket connection."""
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        print(f"✓ WebSocket disconnected for job {job_id}")
    
    async def send_update(self, job_id: str, message: dict):
        """Send an update to all connections for a specific job."""
        if job_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error sending to WebSocket: {e}")
                    disconnected.append(connection)
            
            # Remove disconnected clients
            for connection in disconnected:
                self.disconnect(connection, job_id)
    
    async def broadcast_system_update(self, message: dict):
        """Send a system-wide update to all connections."""
        for job_id in list(self.active_connections.keys()):
            await self.send_update(job_id, message)

manager = ConnectionManager()


def create_progress_callback(job_id: str, manager: ConnectionManager):
    """Factory function to create a ProgressCallback.
    
    This is defined as a factory to avoid importing TrainerCallback at module level,
    which would initialize tokenizers and cause fork warnings.
    
    Args:
        job_id: Training job ID
        manager: WebSocket connection manager
        
    Returns:
        ProgressCallback instance
    """
    from transformers import TrainerCallback
    
    class ProgressCallback(TrainerCallback):
        """Custom callback to send training progress via WebSocket."""
        
        def __init__(self, job_id: str, manager: ConnectionManager):
            self.job_id = job_id
            self.manager = manager
            self.training_metrics = []  # Store metrics history
            self.validation_metrics = []
            # Optional threading.Event that can be set to request cancellation
            self.cancellation_event = None
        
        def on_step_end(self, args, state, control, **kwargs):
            """Called at the end of each training step."""
            # If cancellation_event is set, stop training
            try:
                if hasattr(self, "cancellation_event") and self.cancellation_event is not None:
                    if self.cancellation_event.is_set():
                        print(f"✋ Cancellation requested for job {self.job_id} - stopping training")
                        raise KeyboardInterrupt()
            except Exception:
                pass
            if state.global_step > 0:
                # Calculate progress
                if state.max_steps > 0:
                    total_steps = state.max_steps
                else:
                    # Estimate total steps from num_train_epochs
                    # Use getattr to safely access train_dataloader if available
                    train_dataloader = getattr(state, 'train_dataloader', None)
                    if train_dataloader and hasattr(train_dataloader, '__len__'):
                        total_steps = len(train_dataloader) * args.num_train_epochs
                    else:
                        # Fallback: use a reasonable estimate
                        total_steps = 100 * args.num_train_epochs
                
                current_epoch = state.epoch if hasattr(state, 'epoch') else 0
                
                # Update job progress
                if self.job_id in training_jobs:
                    training_jobs[self.job_id]["progress"] = {
                        "current_step": state.global_step,
                        "total_steps": total_steps,
                        "epoch": int(current_epoch) if current_epoch else 0
                    }
                    training_jobs[self.job_id]["current_step"] = state.global_step
                    training_jobs[self.job_id]["total_steps"] = total_steps
                    training_jobs[self.job_id]["current_epoch"] = int(current_epoch) if current_epoch else 0
                
                # Send WebSocket update
                asyncio.run(self.manager.send_update(self.job_id, {
                    "type": "progress",
                    "job_id": self.job_id,
                    "progress": {
                        "current_step": state.global_step,
                        "total_steps": total_steps,
                        "epoch": int(current_epoch) if current_epoch else 0
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }))
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called when logging occurs."""
            # Also check cancellation here in case cancellation happens between steps
            try:
                if hasattr(self, "cancellation_event") and self.cancellation_event is not None:
                    if self.cancellation_event.is_set():
                        print(f"✋ Cancellation requested for job {self.job_id} during logging - stopping training")
                        raise KeyboardInterrupt()
            except Exception:
                pass
            
            # Check for early stopping request (graceful stop)
            try:
                early_stop_map = globals().get("early_stop_requests", {})
                if early_stop_map.get(self.job_id, False):
                    print(f"🛑 Early stopping requested for job {self.job_id} - will stop gracefully")
                    control.should_training_stop = True
                    # Clear the flag
                    del early_stop_map[self.job_id]
            except Exception:
                pass
            
            if logs:
                # Extract metrics from logs
                current_step = state.global_step
                timestamp = datetime.utcnow().isoformat() + "Z"
                
                # Separate training and evaluation logs
                is_eval = any(k.startswith('eval_') for k in logs.keys())
                
                if is_eval:
                    # Validation metrics
                    eval_loss = logs.get('eval_loss')
                    metric_point = {
                        "step": current_step,
                        "loss": eval_loss,
                        "timestamp": timestamp,
                    }
                    
                    # Add any additional eval metrics
                    for key, value in logs.items():
                        if key.startswith('eval_') and key != 'eval_loss':
                            metric_name = key.replace('eval_', '')
                            metric_point[metric_name] = value
                    
                    self.validation_metrics.append(metric_point)
                    
                    # Update job metrics
                    if self.job_id in training_jobs:
                        if "metrics" not in training_jobs[self.job_id]:
                            training_jobs[self.job_id]["metrics"] = {}
                        training_jobs[self.job_id]["metrics"]["validation"] = self.validation_metrics
                    
                    # Send metrics via WebSocket
                    asyncio.run(self.manager.send_update(self.job_id, {
                        "type": "validation_metrics",
                        "job_id": self.job_id,
                        "metrics": metric_point,
                        "timestamp": timestamp
                    }))
                else:
                    # Training metrics
                    train_loss = logs.get('loss')
                    learning_rate = logs.get('learning_rate')
                    
                    if train_loss is not None:
                        metric_point = {
                            "step": current_step,
                            "loss": train_loss,
                            "learning_rate": learning_rate,
                            "timestamp": timestamp,
                        }
                        
                        # Add any additional metrics
                        for key, value in logs.items():
                            if key not in ['loss', 'learning_rate', 'epoch']:
                                metric_point[key] = value
                        
                        self.training_metrics.append(metric_point)
                        
                        # Update job metrics
                        if self.job_id in training_jobs:
                            if "metrics" not in training_jobs[self.job_id]:
                                training_jobs[self.job_id]["metrics"] = {}
                            training_jobs[self.job_id]["metrics"]["training"] = self.training_metrics
                        
                        # Send metrics via WebSocket
                        asyncio.run(self.manager.send_update(self.job_id, {
                            "type": "training_metrics",
                            "job_id": self.job_id,
                            "metrics": metric_point,
                            "timestamp": timestamp
                        }))
                
                # Send formatted log message
                log_message = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                           for k, v in logs.items() if k != "epoch"])
                
                # Send log via WebSocket
                asyncio.run(self.manager.send_update(self.job_id, {
                    "type": "log",
                    "job_id": self.job_id,
                    "message": log_message,
                    "timestamp": timestamp
                }))
    
    # Return instance of the callback
    return ProgressCallback(job_id, manager)


async def run_model_loading(
    job_id: str,
    model_path: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    dtype: str,
    quantization: Optional[str]
):
    """Execute model loading in the background."""
    from model_garden.inference import InferenceService, set_inference_service
    from model_garden.job_queue import get_job_queue
    
    queue = get_job_queue()
    
    try:
        # Mark job as running
        await queue.start_job(job_id)
        
        print(f"🔄 Loading model: {model_path}")
        
        # Create inference service
        service = InferenceService(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            quantization=quantization,
        )
        
        # Load the model
        await service.load_model()
        
        # Set as global service
        set_inference_service(service)
        
        # Mark job as completed
        model_info = service.get_model_info()
        await queue.complete_job(job_id, result=model_info)
        
        print(f"✅ Model loaded successfully: {model_path}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Model loading failed: {error_msg}")
        await queue.fail_job(job_id, error=error_msg)
        raise


def run_training_job(job_id: str):
    """Execute a training job in the background."""
    try:
        # Get queue and mark job as running
        queue = get_job_queue()
        asyncio.run(queue.start_job(job_id))
        
        # Ensure there's a cancellation event map and register an event for this job
        _ce_map = globals().setdefault("cancellation_events", {})
        _ce_map[job_id] = threading.Event()

        job = training_jobs[job_id]
        
        # Update job status to running
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Persist status change
        storage_manager.save_training_jobs(training_jobs)
        
        # Notify WebSocket clients
        asyncio.run(manager.send_update(job_id, {
            "type": "status_update",
            "job_id": job_id,
            "status": "running",
            "started_at": job["started_at"],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }))
        
        print(f"🚀 Starting training job {job_id}: {job['name']}")
        
        # Check if this is a vision-language model
        is_vision = job.get("is_vision", False)
        from_hub = job.get("from_hub", False)
        validation_from_hub = job.get("validation_from_hub", False)
        validation_dataset_path = job.get("validation_dataset_path")
        
        # Handle quality mode settings
        quality_mode = job.get("quality_mode", False)
        load_in_16bit = job.get("load_in_16bit", False)
        load_in_8bit = job.get("load_in_8bit", False)
        
        # Apply quality mode overrides
        if quality_mode:
            print("🎯 Quality mode enabled - using higher precision settings")
            load_in_16bit = True
            load_in_8bit = False
            # Override lora_config gradient checkpointing if not explicitly set
            lora_config = job.get("lora_config", {})
            if "use_gradient_checkpointing" not in lora_config or lora_config["use_gradient_checkpointing"] == "unsloth":
                lora_config["use_gradient_checkpointing"] = True
            # Override optimizer if not explicitly set
            hyperparams = job.get("hyperparameters", {})
            if "optim" not in hyperparams or hyperparams["optim"] == "adamw_8bit":
                hyperparams["optim"] = "adamw_torch"
            # Enable RSLoRA for high ranks
            if lora_config.get("r", 16) >= 32 and not lora_config.get("use_rslora"):
                lora_config["use_rslora"] = True
                print(f"🎯 LoRA rank {lora_config['r']} >= 32, enabling RSLoRA")
            job["lora_config"] = lora_config
            job["hyperparameters"] = hyperparams
            print("⚠️  Warning: Quality mode uses ~4x more VRAM than default settings")
        
        # Determine quantization (after quality mode overrides)
        load_in_4bit = not (load_in_16bit or load_in_8bit)
        
        if is_vision:
            # Use VisionLanguageTrainer for vision models
            from model_garden.vision_training import VisionLanguageTrainer
            
            print(f"🎨 Using VisionLanguageTrainer for {job['base_model']}")
            
            trainer = VisionLanguageTrainer(
                base_model=job["base_model"],
                max_seq_length=job["hyperparameters"].get("max_seq_length", 16384),  # Default 16384 for vision models
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,  # Add 8-bit support for vision models
            )
            
            # Load model
            trainer.load_model()
            
            # Prepare for training with LoRA and selective layer fine-tuning
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
            
            # Load and format training dataset
            train_dataset = trainer.load_dataset(
                dataset_path=job["dataset_path"],
                from_hub=from_hub,
                split="train",
            )
            formatted_train_dataset = trainer.format_dataset(train_dataset)
            
            # Load and format validation dataset if provided
            formatted_val_dataset = None
            if validation_dataset_path:
                print(f"📊 Loading validation dataset: {validation_dataset_path}")
                val_dataset = trainer.load_dataset(
                    dataset_path=validation_dataset_path,
                    from_hub=validation_from_hub,
                    split="validation",
                )
                formatted_val_dataset = trainer.format_dataset(val_dataset)
                print(f"✓ Validation dataset loaded ({len(formatted_val_dataset)} examples)")
            
            # Train with progress callback and optional early stopping
            hyperparams = job["hyperparameters"]
            progress_callback = create_progress_callback(job_id, manager)
            # Attach cancellation event so callback/trainer can stop gracefully
            progress_callback.cancellation_event = globals().get("cancellation_events", {}).get(job_id)
            
            # Build callbacks list
            from typing import List as TypingList
            from transformers import TrainerCallback as BaseCallback
            callbacks: TypingList[BaseCallback] = [progress_callback]
            
            # Add early stopping if enabled
            if job.get("early_stopping_enabled", False):
                from model_garden.early_stopping import EarlyStoppingCallback
                early_stopping = EarlyStoppingCallback(
                    patience=job.get("early_stopping_patience", 3),
                    threshold=job.get("early_stopping_threshold", 0.0),
                    metric="eval_loss",
                    greater_is_better=False
                )
                callbacks.append(early_stopping)
                print(f"📊 Early stopping enabled (patience={early_stopping.patience}, threshold={early_stopping.threshold})")
            
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
                lr_scheduler_type=hyperparams.get("lr_scheduler_type", "cosine"),  # Cosine better for vision
                max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
                adam_beta1=hyperparams.get("adam_beta1", 0.9),
                adam_beta2=hyperparams.get("adam_beta2", 0.999),
                adam_epsilon=hyperparams.get("adam_epsilon", 1e-8),
                dataloader_num_workers=hyperparams.get("dataloader_num_workers", 0),
                eval_strategy=hyperparams.get("eval_strategy", "steps"),
                load_best_model_at_end=hyperparams.get("load_best_model_at_end", True),
                metric_for_best_model=hyperparams.get("metric_for_best_model", "eval_loss"),
                save_total_limit=hyperparams.get("save_total_limit", 3),
                callbacks=callbacks,
                selective_loss=job.get("selective_loss", False),
                selective_loss_level=job.get("selective_loss_level", "conservative"),
                selective_loss_schema_keys=job.get("selective_loss_schema_keys"),
                selective_loss_masking_start_step=job.get("selective_loss_masking_start_step", 0),
                selective_loss_masking_start_epoch=job.get("selective_loss_masking_start_epoch", 0.0),
                selective_loss_verbose=job.get("selective_loss_verbose", False),
            )
            
            # Save model
            save_method = job.get("save_method", "merged_16bit")
            trainer.save_model(job["output_dir"], save_method=save_method)
            
            # CRITICAL: Clear trainer's dataset references BEFORE cleanup
            # to break circular references and enable garbage collection
            try:
                val_dataset = None  # Clear local reference
                if hasattr(trainer, 'train_dataset'):
                    setattr(trainer, 'train_dataset', None)
                if hasattr(trainer, 'eval_dataset'):
                    setattr(trainer, 'eval_dataset', None)
            except Exception:
                pass
            
            # Aggressively free memory after training completes
            cleanup_training_resources(
                trainer.model,
                trainer.tokenizer,
                trainer.processor,
                trainer,
                formatted_train_dataset,
                formatted_val_dataset,
                train_dataset,
                progress_callback
            )
            
            # Clear cancellation event on normal completion
            try:
                _ce_map = globals().get("cancellation_events")
                if _ce_map and job_id in _ce_map:
                    del _ce_map[job_id]
            except Exception:
                pass
        else:
            # Lazy import ModelTrainer to avoid spawning torch workers at API startup
            from model_garden.training import ModelTrainer
            
            # Use standard ModelTrainer for text-only models
            trainer = ModelTrainer(
                base_model=job["base_model"],
                max_seq_length=job["hyperparameters"].get("max_seq_length", 2048),
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,  # Add 8-bit support for text models
            )
            
            # Load model
            trainer.load_model()
            
            # Prepare for training with LoRA
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
            
            # Load training dataset
            if from_hub:
                train_dataset = trainer.load_dataset_from_hub(job["dataset_path"], split="train")
            else:
                train_dataset = trainer.load_dataset_from_file(job["dataset_path"])
            
            # Format training dataset
            train_dataset = trainer.format_dataset(
                train_dataset,
                instruction_field=job["hyperparameters"].get("instruction_field", "instruction"),
                input_field=job["hyperparameters"].get("input_field", "input"),
                output_field=job["hyperparameters"].get("output_field", "output"),
            )
            
            # Load and format validation dataset if provided
            val_dataset = None
            if validation_dataset_path:
                print(f"📊 Loading validation dataset: {validation_dataset_path}")
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
                print(f"✓ Validation dataset loaded ({len(val_dataset)} examples)")
            
            # Train with progress callback and optional early stopping
            hyperparams = job["hyperparameters"]
            progress_callback = create_progress_callback(job_id, manager)
            
            # Build callbacks list
            from typing import List as TypingList
            from transformers import TrainerCallback as BaseCallback
            callbacks: TypingList[BaseCallback] = [progress_callback]
            
            # Add early stopping if enabled
            if job.get("early_stopping_enabled", False):
                from model_garden.early_stopping import EarlyStoppingCallback
                early_stopping = EarlyStoppingCallback(
                    patience=job.get("early_stopping_patience", 3),
                    threshold=job.get("early_stopping_threshold", 0.0),
                    metric="eval_loss",
                    greater_is_better=False
                )
                callbacks.append(early_stopping)
                print(f"📊 Early stopping enabled (patience={early_stopping.patience}, threshold={early_stopping.threshold})")
            
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
                max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
                adam_beta1=hyperparams.get("adam_beta1", 0.9),
                adam_beta2=hyperparams.get("adam_beta2", 0.999),
                adam_epsilon=hyperparams.get("adam_epsilon", 1e-8),
                dataloader_num_workers=hyperparams.get("dataloader_num_workers", 0),
                eval_strategy=hyperparams.get("eval_strategy", "steps"),
                load_best_model_at_end=hyperparams.get("load_best_model_at_end", True),
                metric_for_best_model=hyperparams.get("metric_for_best_model", "eval_loss"),
                save_total_limit=hyperparams.get("save_total_limit", 3),
                callbacks=callbacks,
            )
            
            # Save final model
            save_method = hyperparams.get("save_method", "merged_16bit")
            if save_method != "lora":
                trainer.save_model(job["output_dir"], save_method=save_method)
        
            # CRITICAL: Clear trainer's dataset references BEFORE cleanup
            # to break circular references and enable garbage collection
            try:
                val_dataset = None  # Clear local reference
                train_dataset = None  # Clear local reference
                if hasattr(trainer, 'train_dataset'):
                    setattr(trainer, 'train_dataset', None)
                if hasattr(trainer, 'eval_dataset'):
                    setattr(trainer, 'eval_dataset', None)
            except Exception:
                pass
        
            # Aggressively free memory after training completes
            cleanup_training_resources(
                trainer.model,
                trainer.tokenizer,
                trainer,
                train_dataset,
                val_dataset,
                progress_callback
            )
        
        # Update job status to completed
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job["progress"] = {"current_step": 100, "total_steps": 100, "epoch": hyperparams.get("num_epochs", 3)}
        
        # Persist status change
        storage_manager.save_training_jobs(training_jobs)
        
        # Notify WebSocket clients
        asyncio.run(manager.send_update(job_id, {
            "type": "status_update",
            "job_id": job_id,
            "status": "completed",
            "completed_at": job["completed_at"],
            "progress": job["progress"],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }))
        
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
        
        # Persist model storage
        storage_manager.save_models(models_storage)
        
        # Mark job as completed in queue
        queue = get_job_queue()
        asyncio.run(queue.complete_job(job_id, result={
            "model_id": model_id,
            "output_dir": job["output_dir"]
        }))
        
        print(f"✅ Training job {job_id} completed successfully!")
        
    except KeyboardInterrupt:
        import traceback
        print(f"✋ Training job {job_id} cancelled by user")
        
        # Cleanup training resources
        try:
            # Collect objects that need cleanup (use locals() to safely get defined variables)
            cleanup_objects = []
            for var_name in ['trainer', 'formatted_train_dataset', 'formatted_val_dataset', 
                           'train_dataset', 'val_dataset', 'progress_callback']:
                if var_name in locals():
                    obj = locals()[var_name]
                    if obj is not None:
                        cleanup_objects.append(obj)
            
            if cleanup_objects:
                cleanup_training_resources(*cleanup_objects)
        except Exception as cleanup_error:
            print(f"⚠️  Error during cleanup: {cleanup_error}")
        
        # Update job status to cancelled - ensure job exists
        if job_id in training_jobs:
            job = training_jobs[job_id]
            job["status"] = "cancelled"
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job["error_message"] = "Training cancelled by user"
        
            # Persist status change
            storage_manager.save_training_jobs(training_jobs)
            
            # Mark job as cancelled in queue
            queue = get_job_queue()
            asyncio.run(queue.cancel_job(job_id))
            
            # Notify WebSocket clients
            asyncio.run(manager.send_update(job_id, {
                "type": "status_update",
                "job_id": job_id,
                "status": "cancelled",
                "completed_at": job["completed_at"],
                "error_message": "Training cancelled by user",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }))
        
        # Clear cancellation event
        try:
            _ce_map = globals().get("cancellation_events")
            if _ce_map and job_id in _ce_map:
                del _ce_map[job_id]
        except Exception:
            pass
        
    except Exception as e:
        import traceback
        print(f"❌ Training job {job_id} failed: {e}")
        
        # Cleanup training resources even on failure
        try:
            # Collect objects that need cleanup (use locals() to safely get defined variables)
            cleanup_objects = []
            for var_name in ['trainer', 'formatted_train_dataset', 'formatted_val_dataset', 
                           'train_dataset', 'val_dataset', 'progress_callback']:
                if var_name in locals():
                    obj = locals()[var_name]
                    if obj is not None:
                        cleanup_objects.append(obj)
            
            if cleanup_objects:
                cleanup_training_resources(*cleanup_objects)
        except Exception as cleanup_error:
            print(f"⚠️  Error during cleanup: {cleanup_error}")
        
        # Update job status to failed - ensure job exists
        if job_id in training_jobs:
            job = training_jobs[job_id]
            job["status"] = "failed"
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job["error_message"] = str(e)
        
            # Persist status change
            storage_manager.save_training_jobs(training_jobs)
            
            # Mark job as failed in queue
            queue = get_job_queue()
            asyncio.run(queue.fail_job(job_id, str(e)))
            
            # Notify WebSocket clients
            asyncio.run(manager.send_update(job_id, {
                "type": "status_update",
                "job_id": job_id,
                "status": "failed",
                "completed_at": job["completed_at"],
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }))
        else:
            print(f"❌ Training job {job_id} failed but job not found in storage: {e}")
        
        # Print full traceback for debugging
        traceback.print_exc()
        
        # Clear cancellation event
        try:
            _ce_map = globals().get("cancellation_events")
            if _ce_map and job_id in _ce_map:
                del _ce_map[job_id]
        except Exception:
            pass
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
            
            # Get optional config from environment
            base_model = os.getenv("MODEL_GARDEN_BASE_MODEL")
            tensor_parallel_size = int(os.getenv("MODEL_GARDEN_TENSOR_PARALLEL_SIZE", "1"))
            gpu_memory_utilization = float(os.getenv("MODEL_GARDEN_GPU_MEMORY_UTILIZATION", "0.0"))  # Default to auto
            quantization = os.getenv("MODEL_GARDEN_QUANTIZATION", "auto")  # Default to auto-detection
            max_model_len_str = os.getenv("MODEL_GARDEN_MAX_MODEL_LEN")
            max_model_len = int(max_model_len_str) if max_model_len_str else None
            dtype = os.getenv("MODEL_GARDEN_DTYPE", "auto")
            
            # LoRA parameters
            enable_lora = os.getenv("MODEL_GARDEN_ENABLE_LORA", "true").lower() == "true"
            max_loras = int(os.getenv("MODEL_GARDEN_MAX_LORAS", "1"))
            max_lora_rank = int(os.getenv("MODEL_GARDEN_MAX_LORA_RANK", "64"))
            
            inference_service = InferenceService(
                model_path=autoload_model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                quantization=quantization,
                max_model_len=max_model_len,
                dtype=dtype,
                enable_lora=enable_lora,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank
            )
            
            # If base_model is explicitly provided, override auto-detection
            if base_model:
                print(f"  Using explicit base model: {base_model}")
                inference_service.base_model_path = base_model
                inference_service.is_adapter = True
                inference_service.adapter_path = autoload_model
            
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
    
    # Cleanup torch compile workers to free memory
    try:
        import signal
        import subprocess
        print("🧹 Cleaning up PyTorch compile workers...")
        result = subprocess.run(
            ["pkill", "-9", "-f", "torch._inductor.compile_worker"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ PyTorch compile workers terminated")
        else:
            print("  (No compile workers found)")
    except Exception as e:
        print(f"Warning: Error cleaning up compile workers: {e}")
    
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
    models_dir = Path(MODELS_DIR)
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


# Model Registry endpoints
@app.get("/api/v1/registry/models")
async def get_registry_models(category: Optional[str] = None):
    """Get all supported models from the registry.
    
    Args:
        category: Optional category filter (text-llm, vision-vlm)
    
    Returns:
        List of supported models with their configurations
    """
    from model_garden.model_registry import get_registry
    
    registry = get_registry()
    models = registry.get_model_list_for_ui(category=category)
    
    return {
        "success": True,
        "data": models,
        "total": len(models)
    }


@app.get("/api/v1/registry/models/{model_id:path}")
async def get_registry_model(model_id: str):
    """Get detailed information about a specific model from the registry.
    
    Args:
        model_id: Model identifier (HuggingFace model ID, can include slashes)
    
    Returns:
        Complete model information including defaults and capabilities
    """
    from model_garden.model_registry import get_model
    
    model = get_model(model_id)
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in registry"
        )
    
    return {
        "success": True,
        "data": {
            "id": model.id,
            "name": model.name,
            "category": model.category,
            "provider": model.provider,
            "base_architecture": model.base_architecture,
            "parameters": model.parameters,
            "description": model.description,
            "tags": model.tags,
            "status": model.status,
            "quantization": model.quantization,
            "requirements": {
                "min_vram_gb": model.requirements.min_vram_gb,
                "recommended_vram_gb": model.requirements.recommended_vram_gb,
                "min_ram_gb": model.requirements.min_ram_gb,
                "cuda_compute_capability": model.requirements.cuda_compute_capability,
                "min_gpus": model.requirements.min_gpus,
            },
            "capabilities": {
                "training": model.capabilities.training,
                "inference": model.capabilities.inference,
                "vision": model.capabilities.vision,
                "structured_outputs": model.capabilities.structured_outputs,
                "streaming": model.capabilities.streaming,
                "function_calling": model.capabilities.function_calling,
            },
            "training_defaults": model.training_defaults,
            "inference_defaults": model.get_inference_config(),
            "urls": model.urls,
        }
    }


@app.get("/api/v1/registry/categories")
async def get_registry_categories():
    """Get all available model categories.
    
    Returns:
        List of categories with their metadata
    """
    from model_garden.model_registry import get_registry
    
    registry = get_registry()
    categories = registry.get_categories()
    
    return {
        "success": True,
        "data": categories
    }


@app.post("/api/v1/registry/validate/training")
async def validate_training_model(request: Dict):
    """Validate if a model can be used for training.
    
    Args:
        request: Dict with 'model_id' key
    
    Returns:
        Validation result with error message if invalid
    """
    from model_garden.model_registry import validate_model_for_training
    
    model_id = request.get("model_id")
    if not model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'model_id' in request"
        )
    
    is_valid, error_message = validate_model_for_training(model_id)
    
    return {
        "success": True,
        "data": {
            "is_valid": is_valid,
            "error_message": error_message
        }
    }


@app.post("/api/v1/registry/validate/inference")
async def validate_inference_model(request: Dict):
    """Validate if a model can be used for inference.
    
    Args:
        request: Dict with 'model_id' key
    
    Returns:
        Validation result with error message if invalid
    """
    from model_garden.model_registry import validate_model_for_inference
    
    model_id = request.get("model_id")
    if not model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'model_id' in request"
        )
    
    is_valid, error_message = validate_model_for_inference(model_id)
    
    return {
        "success": True,
        "data": {
            "is_valid": is_valid,
            "error_message": error_message
        }
    }


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


@app.post("/api/v1/models/{model_id}/upload-to-hub")
async def upload_model_to_hub(
    model_id: str,
    request: dict,
):
    """Upload a model to HuggingFace Hub.
    
    Args:
        model_id: The ID of the model to upload
        request: JSON body containing:
            - repo_id: HuggingFace repository ID (username/repo-name)
            - private: Whether the repository should be private (default: False)
            - commit_message: Optional commit message (default: "Upload model from Model Garden")
            - repo_description: Optional repository description
    
    Returns:
        Success response with repository URL
    """
    if model_id not in models_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    model_data = models_storage[model_id]
    model_path = Path(model_data["path"])
    
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model directory not found: {model_path}"
        )
    
    # Extract request parameters
    repo_id = request.get("repo_id")
    if not repo_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="repo_id is required"
        )
    
    # Validate repo_id format (should be username/repo-name)
    if "/" not in repo_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="repo_id must be in the format 'username/repo-name'"
        )
    
    private = request.get("private", False)
    commit_message = request.get("commit_message", "Upload model from Model Garden")
    repo_description = request.get("repo_description", f"Model fine-tuned with Model Garden. Base model: {model_data.get('base_model', 'unknown')}")
    
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
        import os
        
        # Get HuggingFace token
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="HF_TOKEN environment variable not set. Please configure your HuggingFace token."
            )
        
        api = HfApi(token=hf_token)
        
        # Create the repository if it doesn't exist
        print(f"Creating repository: {repo_id} (private={private})")
        try:
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                private=private,
                exist_ok=True,
                repo_type="model"
            )
            
            # Update repo description if provided
            if repo_description:
                api.update_repo_visibility(
                    repo_id=repo_id,
                    private=private,
                    token=hf_token
                )
        except Exception as e:
            print(f"Repository creation note: {e}")
        
        # Upload the entire model directory
        print(f"Uploading model from {model_path} to {repo_id}...")
        url = upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            token=hf_token,
            commit_message=commit_message,
            repo_type="model"
        )
        
        print(f"✓ Model uploaded successfully to {url}")
        
        # Create a README if it doesn't exist
        readme_path = model_path / "README.md"
        if not readme_path.exists():
            try:
                readme_content = f"""---
license: apache-2.0
base_model: {model_data.get('base_model', 'unknown')}
tags:
  - model-garden
  - fine-tuned
  - {model_data.get('model_type', 'language-model')}
---

# {model_data.get('name', model_id)}

{repo_description}

## Model Details

- **Base Model**: {model_data.get('base_model', 'unknown')}
- **Fine-tuned with**: [Model Garden](https://github.com/leokeba/model-garden)
- **Training Date**: {model_data.get('created_at', 'unknown')}
- **Model Type**: {model_data.get('model_type', 'unknown')}

## Usage

### With Model Garden

```bash
# Serve the model
uv run model-garden serve-model --model-path {repo_id}

# Generate text
uv run model-garden inference-generate \\
    --model-path {repo_id} \\
    --prompt "Your prompt here"
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Training Details

This model was fine-tuned using Model Garden with the following configuration:

- **Dataset**: {model_data.get('training_dataset', 'custom')}
- **Training Steps**: {model_data.get('training_steps', 'unknown')}
- **LoRA Rank**: {model_data.get('lora_rank', 'unknown')}

## Carbon Footprint

Training emissions: {model_data.get('carbon_emissions_g', 'unknown')} gCO2eq

---

*Generated with [Model Garden](https://github.com/leokeba/model-garden)*
"""
                # Upload README
                api.upload_file(
                    path_or_fileobj=readme_content.encode(),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=hf_token,
                    commit_message="Add model card"
                )
                print("✓ README.md created and uploaded")
            except Exception as e:
                print(f"Warning: Could not create README: {e}")
        
        # Update model storage with Hub URL
        models_storage[model_id]["hub_url"] = f"https://huggingface.co/{repo_id}"
        models_storage[model_id]["hub_repo_id"] = repo_id
        storage_manager.save_models(models_storage)
        
        return {
            "success": True,
            "message": f"Model uploaded successfully to HuggingFace Hub",
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "commit_url": url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload model: {str(e)}"
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


@app.post("/api/v1/training/jobs", response_model=APIResponse)
async def create_training_job(job_request: TrainingJobRequest, background_tasks: BackgroundTasks):
    """Create a new training job."""
    import uuid
    
    job_id = str(uuid.uuid4())
    
    # Only resolve paths for local files, not HuggingFace Hub datasets
    dataset_path = job_request.dataset_path if job_request.from_hub else resolve_path(job_request.dataset_path)
    
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
    job_info = {
        "id": job_id,
        "name": job_request.name,
        "status": "queued",
        "base_model": job_request.base_model,
        "dataset_path": dataset_path,
        "validation_dataset_path": validation_dataset_path,
        "output_dir": output_dir,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "started_at": None,
        "completed_at": None,
        "progress": {"current_step": 0, "total_steps": 0, "epoch": 0},
        "error_message": None,
        "hyperparameters": job_request.hyperparameters or {},
        "lora_config": job_request.lora_config or {},
        "from_hub": job_request.from_hub,
        "validation_from_hub": job_request.validation_from_hub,
        "is_vision": job_request.is_vision or False,
        "model_type": job_request.model_type or ("vision" if job_request.is_vision else "text"),
        "save_method": job_request.save_method,
        "metrics": {"training": [], "validation": []},  # Initialize metrics storage
        "selective_loss": job_request.selective_loss,
        "selective_loss_level": job_request.selective_loss_level,
        "selective_loss_schema_keys": job_request.selective_loss_schema_keys,
        "selective_loss_masking_start_step": job_request.selective_loss_masking_start_step,
        "selective_loss_masking_start_epoch": job_request.selective_loss_masking_start_epoch,
        "selective_loss_verbose": job_request.selective_loss_verbose,
        # Quality settings
        "quality_mode": job_request.quality_mode,
        "load_in_16bit": job_request.load_in_16bit,
        "load_in_8bit": job_request.load_in_8bit,
    }
    
    training_jobs[job_id] = job_info
    
    # Persist to disk
    storage_manager.save_training_jobs(training_jobs)
    
    # Add to job queue
    queue = get_job_queue()
    await queue.add_job(
        job_id=job_id,
        job_type="training",
        job_config=job_info,
        priority=0
    )
    
    # Get queue position
    position = await queue.get_queue_position(job_id)
    position_msg = f" (position in queue: {position})" if position and position > 1 else ""
    
    # Start training job in background
    background_tasks.add_task(run_training_job, job_id)
    
    return APIResponse(
        success=True,
        data={
            "job_id": job_id,
            "queue_position": position
        },
        message=f"Training job {job_id} created and queued for execution{position_msg}"
    )


@app.get("/api/v1/training/jobs/{job_id}", response_model=TrainingJobInfo)
async def get_training_job(job_id: str):
    """Get details for a specific training job."""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    job_data = training_jobs[job_id].copy()
    
    # Add queue information if job is queued
    if job_data["status"] == "queued":
        queue = get_job_queue()
        position = await queue.get_queue_position(job_id)
        if position:
            job_data["queue_position"] = position
    
    return TrainingJobInfo(**job_data)


@app.delete("/api/v1/training/jobs/{job_id}", response_model=APIResponse)
async def delete_or_cancel_training_job(job_id: str):
    """Delete or cancel a training job.
    
    - For running/queued jobs: cancels the job
    - For completed/failed/cancelled jobs: removes the job from the list
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    job = training_jobs[job_id]
    
    # If job is finished (completed/failed/cancelled), delete it from the list
    if job["status"] in ["completed", "failed", "cancelled"]:
        del training_jobs[job_id]
        storage_manager.save_training_jobs(training_jobs)
        
        # Notify WebSocket clients about deletion
        await manager.send_update(job_id, {
            "type": "job_deleted",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        return APIResponse(
            success=True,
            message=f"Training job {job_id} deleted successfully"
        )
    
    # Try to cancel in queue (only works if job is queued, not running)
    queue = get_job_queue()
    cancelled_in_queue = await queue.cancel_job(job_id)
    
    # If job is running/queued, cancel it
    job["status"] = "cancelled"
    job["completed_at"] = datetime.utcnow().isoformat() + "Z"
    storage_manager.save_training_jobs(training_jobs)
    
    # Notify WebSocket clients
    await manager.send_update(job_id, {
        "type": "status_update",
        "job_id": job_id,
        "status": "cancelled",
        "completed_at": job["completed_at"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    # If a cancellation event exists for this job, set it to signal the training loop
    try:
        _ce_map = globals().get("cancellation_events")
        if _ce_map and job_id in _ce_map:
            _ce_map[job_id].set()
    except Exception:
        pass
    
    status_msg = "cancelled from queue" if cancelled_in_queue else "cancellation requested (job may be running)"
    
    return APIResponse(
        success=True,
        message=f"Training job {job_id} {status_msg}"
    )


@app.post("/api/v1/training/jobs/{job_id}/stop", response_model=APIResponse)
async def request_early_stop(job_id: str):
    """Request early stopping for a running training job.
    
    This will gracefully stop training at the next evaluation, allowing the model
    to be saved properly. Different from cancellation which stops immediately.
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    job = training_jobs[job_id]
    
    # Only allow early stopping for running jobs
    if job["status"] != "running":
        return APIResponse(
            success=False,
            message=f"Cannot request early stopping for job with status: {job['status']}"
        )
    
    # Set a flag that the ProgressCallback can check
    try:
        early_stop_map = globals().setdefault("early_stop_requests", {})
        early_stop_map[job_id] = True
        print(f"🛑 Early stopping requested for job {job_id}")
        
        # Notify WebSocket clients
        await manager.send_update(job_id, {
            "type": "early_stop_requested",
            "job_id": job_id,
            "message": "Early stopping requested - will stop at next evaluation",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        return APIResponse(
            success=True,
            message="Early stopping requested - training will stop at next evaluation"
        )
    except Exception as e:
        return APIResponse(
            success=False,
            message=f"Failed to request early stopping: {str(e)}"
        )


@app.get("/api/v1/training/queue", response_model=APIResponse)
async def get_training_queue():
    """Get current training job queue status."""
    queue = get_job_queue()
    
    # Get all queued jobs
    queued_jobs = await queue.list_jobs(status=JobStatus.QUEUED, job_type="training")
    
    # Get running jobs
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
                    "priority": j["priority"]
                }
                for i, j in enumerate(queued_jobs)
            ],
            "running_jobs": [
                {
                    "job_id": j["job_id"],
                    "name": j["job_config"].get("name", "Unnamed"),
                    "started_at": j["started_at"],
                    "status_message": j["status_message"]
                }
                for j in running_jobs
            ]
        },
        message=f"{len(queued_jobs)} jobs queued, {len(running_jobs)} running"
    )


@app.websocket("/ws/training/{job_id}")
async def websocket_training_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time training job updates.
    
    Sends updates about:
    - Job status changes (queued -> running -> completed/failed)
    - Training progress (steps, epochs, loss)
    - Logs and error messages
    """
    await manager.connect(websocket, job_id)
    
    try:
        # Send initial job status
        if job_id in training_jobs:
            await websocket.send_json({
                "type": "initial_state",
                "job": training_jobs[job_id],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Training job {job_id} not found",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
            await websocket.close()
            return
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client (e.g., ping/pong for keepalive)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
    
    finally:
        manager.disconnect(websocket, job_id)


# ============================================================================
# Inference Endpoints
# ============================================================================

class InferenceRequest(BaseModel):
    """Request for text generation."""
    prompt: str
    max_tokens: Optional[int] = None  # None = auto-determine (8192 for structured, 512 otherwise)
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
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
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
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
    gpu_memory_utilization: float = 0.0  # 0 = auto mode
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    quantization: Optional[str] = None


@app.post("/api/v1/inference/load", response_model=APIResponse)
async def load_inference_model(request: LoadModelRequest, background_tasks: BackgroundTasks):
    """Load a model for inference (queued if another model is loading).
    
    If parameters are not specified, defaults from the model registry are applied.
    """
    from model_garden.inference import InferenceService, get_inference_service, set_inference_service
    from model_garden.job_queue import get_job_queue, JobType, JobStatus
    from model_garden.model_registry import get_model
    
    queue = get_job_queue()
    
    # Resolve model path (handles simple names, ./models/ prefix, and HF IDs)
    model_path = resolve_model_path(request.model_path)
    
    # Try to get model defaults from registry
    model_info = None
    try:
        model_info = get_model(model_path)
        if model_info:
            print(f"📋 Found model in registry: {model_info.name}")
    except Exception as e:
        print(f"ℹ️  Model not in registry, using provided parameters: {e}")
    
    # Apply defaults from registry if parameters not specified
    max_model_len = request.max_model_len
    dtype = request.dtype
    gpu_memory_utilization = request.gpu_memory_utilization
    quantization = request.quantization
    tensor_parallel_size = request.tensor_parallel_size
    
    if model_info:
        # Apply defaults for unspecified parameters
        if max_model_len is None:
            max_model_len = model_info.inference_defaults.max_model_len
            print(f"  Using registry default max_model_len: {max_model_len}")
        
        if dtype == "auto":
            dtype = model_info.inference_defaults.dtype
            print(f"  Using registry default dtype: {dtype}")
        
        if gpu_memory_utilization == 0.0:
            gpu_memory_utilization = model_info.inference_defaults.gpu_memory_utilization
            print(f"  Using registry default gpu_memory_utilization: {gpu_memory_utilization}")
        
        if quantization is None and model_info.inference_defaults.quantization:
            quantization = model_info.inference_defaults.quantization
            print(f"  Using registry default quantization: {quantization}")
        
        if tensor_parallel_size == 1 and model_info.inference_defaults.tensor_parallel_size > 1:
            tensor_parallel_size = model_info.inference_defaults.tensor_parallel_size
            print(f"  Using registry default tensor_parallel_size: {tensor_parallel_size}")
    
    # Check if a model is already loaded
    current_service = get_inference_service()
    if current_service and current_service.is_loaded:
        return APIResponse(
            success=False,
            message=f"Model already loaded: {current_service.model_path}. Unload it first."
        )
    
    # Check if a model is currently loading
    loading_job = await queue.get_running_job(JobType.MODEL_LOADING)
    if loading_job:
        return APIResponse(
            success=False,
            message=f"A model is already loading: {loading_job.get('job_config', {}).get('model_path', 'unknown')}. Please wait or cancel it first."
        )
    
    # Check if there are queued model loading jobs
    has_queued = await queue.has_running_job(JobType.MODEL_LOADING)
    if has_queued:
        # Add to queue
        job_id = f"model-loading-{int(datetime.now().timestamp())}"
        await queue.add_job(
            job_id=job_id,
            job_type=JobType.MODEL_LOADING,
            job_config={
                "model_path": model_path,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "dtype": dtype,
                "quantization": quantization,
            },
            priority=0
        )
        
        position = await queue.get_queue_position(job_id)
        
        return APIResponse(
            success=True,
            data={"job_id": job_id, "status": "queued", "queue_position": position},
            message=f"Model loading queued (position: {position})"
        )
    
    # No model loading in progress, start immediately
    job_id = f"model-loading-{int(datetime.now().timestamp())}"
    
    # Add to queue as running
    await queue.add_job(
        job_id=job_id,
        job_type=JobType.MODEL_LOADING,
        job_config={
            "model_path": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "quantization": quantization,
        },
        priority=0
    )
    
    # Start loading in background
    background_tasks.add_task(
        run_model_loading,
        job_id,
        model_path,
        tensor_parallel_size,
        gpu_memory_utilization,
        max_model_len,
        dtype,
        quantization
    )
    
    return APIResponse(
        success=True,
        data={"job_id": job_id, "status": "loading"},
        message=f"Model loading started: {request.model_path}"
    )


@app.post("/api/v1/inference/unload", response_model=APIResponse)
async def unload_inference_model():
    """Unload the currently loaded model."""
    from model_garden.inference import get_inference_service, set_inference_service
    from model_garden.carbon import stop_inference_tracker
    
    service = get_inference_service()
    if not service or not service.is_loaded:
        return APIResponse(
            success=False,
            message="No model currently loaded"
        )
    
    try:
        # Stop carbon tracking and save emissions
        try:
            emissions_data = stop_inference_tracker()
            if emissions_data:
                print(f"✅ Inference emissions saved: {emissions_data['emissions_kg_co2']:.6f} kg CO2")
                print(f"   Requests: {emissions_data.get('request_count', 0)}")
                print(f"   Tokens: {emissions_data.get('total_tokens', 0)}")
        except Exception as e:
            print(f"⚠️  Failed to stop inference carbon tracking: {e}")
        
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
    """Get inference service status including queue information."""
    from model_garden.inference import get_inference_service
    from model_garden.job_queue import get_job_queue, JobType
    
    queue = get_job_queue()
    
    # Check for running or queued model loading jobs
    loading_job = await queue.get_running_job(JobType.MODEL_LOADING)
    queued_jobs = await queue.list_jobs(status=JobStatus.QUEUED, job_type=JobType.MODEL_LOADING.value)
    
    service = get_inference_service()
    
    response = {
        "loaded": service is not None and service.is_loaded,
        "model_info": service.get_model_info() if service and service.is_loaded else None,
    }
    
    # Add loading status if a model is being loaded
    if loading_job:
        response["loading"] = {
            "job_id": loading_job["job_id"],
            "model_path": loading_job["job_config"].get("model_path"),
            "started_at": loading_job["started_at"],
            "status_message": loading_job["status_message"]
        }
    
    # Add queue information if models are queued
    if queued_jobs:
        response["queue"] = {
            "count": len(queued_jobs),
            "jobs": [
                {
                    "job_id": job["job_id"],
                    "model_path": job["job_config"].get("model_path"),
                    "queued_at": job["queued_at"],
                    "position": i + 1
                }
                for i, job in enumerate(queued_jobs)
            ]
        }
    
    return response


@app.get("/api/v1/inference/queue")
async def get_model_loading_queue():
    """Get model loading queue status."""
    from model_garden.job_queue import get_job_queue, JobType, JobStatus
    
    queue = get_job_queue()
    
    # Get all model loading jobs
    all_jobs = await queue.list_jobs(job_type=JobType.MODEL_LOADING.value)
    
    running = [j for j in all_jobs if j["status"] == JobStatus.RUNNING]
    queued = [j for j in all_jobs if j["status"] == JobStatus.QUEUED]
    completed = [j for j in all_jobs if j["status"] == JobStatus.COMPLETED][:10]  # Last 10
    failed = [j for j in all_jobs if j["status"] == JobStatus.FAILED][:10]  # Last 10
    
    return {
        "running": running,
        "queued": queued,
        "completed": completed,
        "failed": failed,
        "summary": {
            "running_count": len(running),
            "queued_count": len(queued),
            "total_completed": len([j for j in all_jobs if j["status"] == JobStatus.COMPLETED]),
            "total_failed": len([j for j in all_jobs if j["status"] == JobStatus.FAILED]),
        }
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
                
                async for chunk in stream:  # type: ignore
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
                
                # Let InferenceService.generate() handle max_tokens defaults
                # It will auto-set to 8192 for structured outputs if not specified
                # No need to override here - the inference layer knows best
        
        # IMPORTANT: Do NOT set max_tokens here!
        # The inference layer (inference.py) handles smart defaults:
        # - 8192 for structured outputs
        # - 512 for regular generation
        # If we set it here, the inference layer can't detect "not specified"
        
        if request.stream:
            # Streaming response
            async def generate_stream():
                total_tokens = 0
                stream = await service.chat_completion(**gen_params)
                
                async for chunk in stream:  # type: ignore
                    # Count tokens if available
                    if isinstance(chunk, dict) and 'usage' in chunk:
                        total_tokens = chunk['usage'].get('completion_tokens', 0)
                    
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Record request in carbon tracker
                try:
                    from model_garden.carbon import get_inference_tracker
                    tracker = get_inference_tracker()
                    if tracker:
                        tracker.record_request(tokens_generated=total_tokens)
                except Exception:
                    pass  # Silently ignore tracking errors
                
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Complete response
            response = await service.chat_completion(**gen_params)
            
            # Record request in carbon tracker and get emissions estimate
            carbon_data = None
            try:
                from model_garden.carbon import get_inference_tracker
                tracker = get_inference_tracker()
                if tracker:
                    tokens = 0
                    if isinstance(response, dict) and 'usage' in response:
                        tokens = response['usage'].get('completion_tokens', 0)
                    tracker.record_request(tokens_generated=tokens)
                    
                    # Get current stats with REAL measured emissions from CodeCarbon
                    stats = tracker.get_current_stats()
                    if stats and stats.get('tracking', False):
                        carbon_data = {
                            "emissions_g_co2": stats.get('emissions_g_co2', 0.0),
                            "emissions_per_request_g": stats.get('emissions_per_request_g', 0.0),
                            "session_total_kg_co2": stats.get('emissions_kg_co2', 0.0),
                            "session_requests": stats.get('request_count', 0),
                            "session_tokens": stats.get('total_tokens', 0),
                            "tracking_active": True,
                            "measured": True  # Flag to indicate this is real data, not estimated
                        }
            except Exception as e:
                print(f"Warning: Could not add carbon tracking data: {e}")
                pass  # Silently ignore tracking errors
            
            # Add carbon data to response if available
            if carbon_data and isinstance(response, dict):
                response['x_carbon_trace'] = carbon_data
            
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
# Dataset endpoints
# ============================================================================
from fastapi import UploadFile, File
import json
import pandas as pd

@app.get("/api/v1/datasets")
async def list_datasets():
    """List all available datasets."""
    datasets_dir = Path("./storage/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = []
    for dataset_file in datasets_dir.iterdir():
        if dataset_file.is_file():
            # Get file stats
            stat = dataset_file.stat()
            
            # Try to count examples
            example_count = 0
            try:
                if dataset_file.suffix == ".jsonl":
                    with open(dataset_file, "r") as f:
                        example_count = sum(1 for _ in f)
                elif dataset_file.suffix == ".json":
                    with open(dataset_file, "r") as f:
                        data = json.load(f)
                        example_count = len(data) if isinstance(data, list) else 1
                elif dataset_file.suffix == ".csv":
                    df = pd.read_csv(dataset_file)
                    example_count = len(df)
                elif dataset_file.suffix == ".parquet":
                    df = pd.read_parquet(dataset_file)
                    example_count = len(df)
            except Exception as e:
                print(f"Warning: Could not count examples in {dataset_file.name}: {e}")
            
            datasets.append({
                "name": dataset_file.name,
                "path": str(dataset_file),
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
                "format": dataset_file.suffix.lstrip('.'),
                "examples": example_count,
            })
    
    # Sort by modified date (newest first)
    datasets.sort(key=lambda x: x["modified_at"], reverse=True)
    
    return {"datasets": datasets}


@app.post("/api/v1/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file."""
    from model_garden.dataset_validator import DatasetValidator
    
    datasets_dir = Path("./storage/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate file extension
    allowed_extensions = [".json", ".jsonl", ".csv", ".txt", ".parquet"]
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a name"
        )
    
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    # Save file
    file_path = datasets_dir / file.filename
    
    # Check if file already exists
    if file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dataset {file.filename} already exists"
        )
    
    try:
        # Write file in chunks
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):  # Read 8KB at a time
                f.write(chunk)
        
        # Validate dataset with new validator
        validation_stats = None
        if file_ext in [".json", ".jsonl", ".csv"]:
            try:
                validation_stats = DatasetValidator.validate_dataset(file_path)
                
                # If there are critical errors, delete the file and fail
                if validation_stats.validation_errors:
                    file_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "message": "Dataset validation failed",
                            "errors": validation_stats.validation_errors,
                            "warnings": validation_stats.warnings
                        }
                    )
            except HTTPException:
                raise
            except Exception as e:
                print(f"Warning: Dataset validation failed: {e}")
        
        # Get file stats
        stat = file_path.stat()
        
        # Use validation stats if available, otherwise fallback to old counting
        example_count = 0
        if validation_stats:
            example_count = validation_stats.total_rows
        else:
            try:
                if file_ext == ".jsonl":
                    with open(file_path, "r") as f:
                        example_count = sum(1 for _ in f)
                elif file_ext == ".json":
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        example_count = len(data) if isinstance(data, list) else 1
                elif file_ext == ".csv":
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    example_count = len(df)
                elif file_ext == ".parquet":
                    import pandas as pd
                    df = pd.read_parquet(file_path)
                    example_count = len(df)
            except Exception as e:
                print(f"Warning: Could not count examples: {e}")
        
        response_data = {
            "success": True,
            "message": f"Dataset {file.filename} uploaded successfully",
            "dataset": {
                "name": file.filename,
                "path": str(file_path),
                "size": stat.st_size,
                "format": file_ext.lstrip('.'),
                "examples": example_count,
            }
        }
        
        # Add validation details if available
        if validation_stats:
            response_data["validation"] = {
                "schema_type": DatasetValidator.detect_schema_type(validation_stats.sample_rows) if validation_stats.sample_rows else "unknown",
                "fields": validation_stats.fields,
                "warnings": validation_stats.warnings,
                "has_images": validation_stats.has_images,
                "image_count": validation_stats.image_count,
                "avg_input_length": validation_stats.avg_input_length,
                "avg_output_length": validation_stats.avg_output_length,
                "total_tokens_estimate": validation_stats.total_tokens_estimate,
            }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload dataset: {str(e)}"
        )


@app.get("/api/v1/datasets/{dataset_name}/stats")
async def get_dataset_stats(dataset_name: str):
    """Get detailed statistics for a dataset."""
    from model_garden.dataset_validator import DatasetValidator
    
    datasets_dir = Path("./storage/datasets")
    file_path = datasets_dir / dataset_name
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_name} not found"
        )
    
    try:
        stats = DatasetValidator.validate_dataset(file_path)
        
        return {
            "name": dataset_name,
            "total_rows": stats.total_rows,
            "format": stats.format,
            "fields": stats.fields,
            "field_types": stats.field_types,
            "missing_fields": stats.missing_fields,
            "file_size_bytes": stats.file_size_bytes,
            "validation_errors": stats.validation_errors,
            "warnings": stats.warnings,
            "sample_rows": stats.sample_rows,
            "schema_type": DatasetValidator.detect_schema_type(stats.sample_rows) if stats.sample_rows else "unknown",
            "text_stats": {
                "avg_input_length": stats.avg_input_length,
                "avg_output_length": stats.avg_output_length,
                "total_tokens_estimate": stats.total_tokens_estimate,
            } if stats.avg_input_length is not None else None,
            "vision_stats": {
                "has_images": stats.has_images,
                "image_count": stats.image_count,
                "sample_image_paths": stats.image_paths,
            } if stats.has_images else None,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze dataset: {str(e)}"
        )


@app.get("/api/v1/datasets/{dataset_name}/preview")
async def preview_dataset(dataset_name: str, limit: int = 10):
    """Preview samples from a dataset."""
    datasets_dir = Path("./storage/datasets")
    file_path = datasets_dir / dataset_name
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_name} not found"
        )
    
    try:
        samples = []
        file_ext = file_path.suffix.lower()
        
        if file_ext == ".jsonl":
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        samples.append({"_error": "Invalid JSON", "_raw": line})
        
        elif file_ext == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:limit]
                else:
                    samples = [data]
        
        elif file_ext == ".csv":
            df = pd.read_csv(file_path, nrows=limit)
            samples = df.to_dict('records')
        
        elif file_ext == ".parquet":
            df = pd.read_parquet(file_path)
            samples = df.head(limit).to_dict('records')
        
        elif file_ext == ".txt":
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    samples.append({"text": line.strip()})
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot preview {file_ext} files"
            )
        
        return {"samples": samples, "count": len(samples)}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to preview dataset: {str(e)}"
        )


@app.delete("/api/v1/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset."""
    datasets_dir = Path("./storage/datasets")
    file_path = datasets_dir / dataset_name
    
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {dataset_name} not found"
        )
    
    try:
        file_path.unlink()
        return {
            "success": True,
            "message": f"Dataset {dataset_name} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete dataset: {str(e)}"
        )


@app.post("/api/v1/datasets/from-hub")
async def load_dataset_from_hub(request: dict):
    """Load a dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
        import json
        
        dataset_id = request.get("dataset_id")
        split = request.get("split", "train")
        
        if not dataset_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="dataset_id is required"
            )
        
        print(f"📥 Loading dataset {dataset_id} from HuggingFace Hub...")
        
        # Load dataset from Hub
        dataset = load_dataset(dataset_id, split=split)
        
        # Save to storage
        datasets_dir = Path("./storage/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a safe filename from dataset_id
        safe_name = dataset_id.replace("/", "_") + ".jsonl"
        output_path = datasets_dir / safe_name
        
        # Convert to JSONL format and count examples
        example_count = 0
        with open(output_path, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
                example_count += 1
        
        print(f"✓ Dataset saved to {output_path}")
        print(f"✓ Total examples: {example_count}")
        
        return {
            "success": True,
            "message": f"Dataset {dataset_id} loaded successfully",
            "dataset_name": safe_name,
            "examples": example_count,
            "path": str(output_path)
        }
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset from Hub: {str(e)}"
        )


# ============================================================================
# Carbon/Emissions endpoints
# ============================================================================

from model_garden.carbon import get_emissions_db

@app.get("/api/v1/carbon/emissions")
async def list_emissions(
    job_type: Optional[str] = None,
    limit: Optional[int] = None
):
    """
    List all carbon emissions records.
    
    Args:
        job_type: Filter by job type ('training', 'inference', or None for all)
        limit: Maximum number of records to return
    """
    try:
        # Get emissions from persistent database
        db = get_emissions_db()
        emissions_records = db.get_all_emissions(job_type=job_type, limit=limit)
        
        # Convert to API format (matching frontend expectations)
        formatted_emissions = []
        for record in emissions_records:
            # Get job name from training jobs or use job_id
            job_name = record.get("job_id", "Unknown")
            if record["job_id"] in training_jobs:
                job_name = training_jobs[record["job_id"]].get("name", job_name)
            
            # Get model name
            model_name = record.get("model_name", "Unknown")
            if not model_name or model_name == "Unknown":
                if record["job_id"] in training_jobs:
                    model_name = training_jobs[record["job_id"]].get("base_model", "Unknown")
            
            formatted_emissions.append({
                "id": f"emission-{record['job_id']}",
                "job_id": record["job_id"],
                "job_name": job_name,
                "stage": record.get("job_type", "training"),  # 'training' or 'inference'
                "model_name": model_name,
                "timestamp": record.get("timestamp", ""),
                "duration": record.get("duration_seconds", 0.0),
                "energy_consumed": record.get("energy_consumed_kwh", 0.0),
                "emissions_kg": record.get("emissions_kg_co2", 0.0),
                "emissions_rate": record.get("emissions_rate_kg_per_sec", 0.0),
                "cpu_energy": record.get("cpu_energy_kwh", 0.0),
                "gpu_energy": record.get("gpu_energy_kwh", 0.0),
                "ram_energy": record.get("ram_energy_kwh", 0.0),
                "carbon_intensity": record.get("carbon_intensity_g_per_kwh", 0.0),
                "country": record.get("country_name", "Unknown"),
                "region": record.get("region", "Unknown"),
                "equivalents": record.get("equivalents", {}),
                "boamps_report": True  # Indicate BoAmps report is available
            })
        
        return {"emissions": formatted_emissions, "count": len(formatted_emissions)}
        
    except Exception as e:
        # Fallback to empty list if there's an error
        import logging
        logging.warning(f"Could not load emissions data: {e}")
        return {"emissions": [], "count": 0}


@app.get("/api/v1/carbon/summary")
async def get_emissions_summary_endpoint():
    """Get aggregate emissions statistics."""
    try:
        db = get_emissions_db()
        summary = db.get_total_emissions()
        return summary
    except Exception as e:
        import logging
        logging.warning(f"Could not load emissions summary: {e}")
        return {
            "total_emissions_kg_co2": 0.0,
            "total_energy_kwh": 0.0,
            "total_duration_seconds": 0.0,
            "total_count": 0,
            "by_type": {},
            "equivalents": {}
        }


@app.get("/api/v1/carbon/inference/stats")
async def get_inference_stats():
    """Get current inference carbon tracking statistics."""
    try:
        from model_garden.carbon import get_inference_tracker
        tracker = get_inference_tracker()
        
        if tracker:
            stats = tracker.get_current_stats()
            return {
                "tracking": True,
                **stats
            }
        else:
            return {
                "tracking": False,
                "message": "No inference tracking active. Load a model to start tracking."
            }
    except Exception as e:
        return {
            "tracking": False,
            "error": str(e)
        }


@app.get("/api/v1/carbon/boamps/{job_id}")
async def get_boamps_report(job_id: str):
    """Get BoAmps report for a specific job."""
    try:
        from model_garden.carbon import get_emissions_db, get_boamps_generator
        
        # Get emissions data from database
        db = get_emissions_db()
        emission_data = db.get_emission(job_id)
        
        if not emission_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No emissions data found for job {job_id}"
            )
        
        # Get job config if available
        job_config = {}
        if job_id in training_jobs:
            job = training_jobs[job_id]
            job_config = {
                "base_model": job.get("base_model"),
                "dataset_path": job.get("dataset_path"),
                "hyperparameters": job.get("hyperparameters"),
                "lora_config": job.get("lora_config")
            }
        
        # Generate BoAmps report
        generator = get_boamps_generator()
        report = generator.generate_report(
            emissions_data=emission_data,
            job_config=job_config,
            report_status="final"
        )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating BoAmps report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate BoAmps report: {str(e)}"
        )


# ============================================================================
# System endpoints
# ============================================================================
@app.get("/api/v1/system/status")
async def system_status():
    """Get system status information."""
    import psutil
    import torch
    
    # GPU information with detailed metrics
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            # Try to use pynvml for detailed GPU metrics
            import pynvml
            pynvml.nvmlInit()
            
            device_count = torch.cuda.device_count()
            gpus = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Utilization info
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    mem_util = utilization.memory
                except:
                    gpu_util = None
                    mem_util = None
                
                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except:
                    power_usage = None
                    power_limit = None
                
                gpus.append({
                    "id": i,
                    "name": name,
                    "memory": {
                        "total": mem_info.total,
                        "used": mem_info.used,
                        "free": mem_info.free,
                        "used_percent": round((float(mem_info.used) / float(mem_info.total)) * 100, 1),
                    },
                    "utilization": {
                        "gpu": gpu_util,
                        "memory": mem_util,
                    },
                    "temperature": temperature,
                    "power": {
                        "usage": power_usage,
                        "limit": power_limit,
                    } if power_usage is not None else None,
                })
            
            pynvml.nvmlShutdown()
            
            gpu_info = {
                "available": True,
                "device_count": device_count,
                "devices": gpus,
            }
            
        except Exception as e:
            # Fallback to basic torch info if pynvml fails
            print(f"Failed to get detailed GPU info: {e}")
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
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_used": psutil.virtual_memory().used,
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
                "percent": psutil.disk_usage("/").percent,
            },
        },
        "gpu": gpu_info,
        "storage": {
            "models_count": len(models_storage),
            "training_jobs_count": len(training_jobs),
            "active_jobs": len([j for j in training_jobs.values() if j["status"] in ["running", "queued"]]),
        },
    }


@app.post("/api/v1/system/cleanup")
async def cleanup_gpu_memory():
    """Force cleanup of GPU memory and Python garbage collection.
    
    This endpoint is useful when GPU memory isn't properly released
    after model operations. It performs:
    - Garbage collection
    - CUDA cache clearing
    - Memory synchronization
    
    Note: This doesn't unload models, use /inference/unload for that.
    """
    import gc
    
    result = {
        "success": True,
        "actions": [],
        "gpu_memory_before": None,
        "gpu_memory_after": None,
    }
    
    try:
        # Get initial GPU memory if available
        import torch
        mem_before = 0.0  # Initialize to avoid unbound variable
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            result["gpu_memory_before"] = f"{mem_before:.2f} GB"
        
        # Force garbage collection
        collected = gc.collect()
        result["actions"].append(f"Garbage collection: {collected} objects collected")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result["actions"].append("CUDA cache cleared")
            
            # Get final GPU memory
            mem_after = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            result["gpu_memory_after"] = f"{mem_after:.2f} GB"
            result["actions"].append(f"Freed: {mem_before - mem_after:.2f} GB")
        else:
            result["actions"].append("CUDA not available")
        
        result["message"] = "GPU memory cleanup completed"
        
    except Exception as e:
        result["success"] = False
        result["message"] = f"Cleanup failed: {str(e)}"
        result["actions"].append(f"Error: {str(e)}")
    
    return result


# Mount static files for the frontend (must be last to not override API routes)
frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
if frontend_build_path.exists():
    # Serve static assets
    app.mount("/_app", StaticFiles(directory=str(frontend_build_path / "_app")), name="static-assets")
    
    # Catch-all route for SvelteKit client-side routing (must be last!)
    from fastapi.responses import FileResponse
    
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve the SvelteKit SPA for all non-API routes.
        
        Note: This route is defined last so API routes take priority.
        FastAPI matches routes in the order they're defined.
        """
        # Try to serve specific HTML files first (e.g., datasets.html, models.html)
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