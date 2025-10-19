"""Job queue management for background tasks (training, model loading, etc.)."""

import asyncio
from enum import Enum
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import json


class JobType(str, Enum):
    """Job type enumeration."""
    TRAINING = "training"
    MODEL_LOADING = "model_loading"
    MODEL_UNLOADING = "model_unloading"
    DATASET_PROCESSING = "dataset_processing"


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobQueue:
    """
    Simple job queue manager using FastAPI BackgroundTasks.
    
    Manages job lifecycle: queued -> running -> completed/failed/cancelled
    """
    
    def __init__(self):
        """Initialize job queue."""
        self._queue: Dict[str, Dict[str, Any]] = {}
        self._queue_lock = asyncio.Lock()
        self._storage_file = Path("storage/job_queue.json")
        self._load_queue()
    
    def _load_queue(self) -> None:
        """Load queue from disk."""
        if self._storage_file.exists():
            try:
                with open(self._storage_file, 'r') as f:
                    data = json.load(f)
                    self._queue = data.get("queue", {})
                    
                    # Reset any running jobs to queued (in case of crash)
                    for job_id, job_data in self._queue.items():
                        if job_data.get("status") == JobStatus.RUNNING:
                            job_data["status"] = JobStatus.QUEUED
                            job_data["status_message"] = "Reset after restart"
            except Exception as e:
                print(f"Warning: Could not load job queue: {e}")
                self._queue = {}
    
    def _save_queue(self) -> None:
        """Save queue to disk."""
        try:
            self._storage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._storage_file, 'w') as f:
                json.dump({
                    "queue": self._queue,
                    "updated_at": datetime.utcnow().isoformat() + "Z"
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save job queue: {e}")
    
    async def add_job(
        self,
        job_id: str,
        job_type: str,
        job_config: Dict[str, Any],
        priority: int = 0
    ) -> None:
        """
        Add a job to the queue.
        
        Args:
            job_id: Unique job identifier
            job_type: Type of job ("training", "inference", etc.)
            job_config: Job configuration dict
            priority: Job priority (higher = more important)
        """
        async with self._queue_lock:
            self._queue[job_id] = {
                "job_id": job_id,
                "job_type": job_type,
                "job_config": job_config,
                "status": JobStatus.QUEUED,
                "status_message": "Queued for processing",
                "priority": priority,
                "queued_at": datetime.utcnow().isoformat() + "Z",
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result": None
            }
            self._save_queue()
    
    async def start_job(self, job_id: str) -> bool:
        """
        Mark a job as running.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was started, False if not found or already running
        """
        async with self._queue_lock:
            if job_id not in self._queue:
                return False
            
            job = self._queue[job_id]
            if job["status"] != JobStatus.QUEUED:
                return False
            
            job["status"] = JobStatus.RUNNING
            job["status_message"] = "Running"
            job["started_at"] = datetime.utcnow().isoformat() + "Z"
            self._save_queue()
            return True
    
    async def complete_job(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a job as completed.
        
        Args:
            job_id: Job identifier
            result: Optional result data
            
        Returns:
            True if job was marked complete, False if not found
        """
        async with self._queue_lock:
            if job_id not in self._queue:
                return False
            
            job = self._queue[job_id]
            job["status"] = JobStatus.COMPLETED
            job["status_message"] = "Completed successfully"
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job["result"] = result
            self._save_queue()
            return True
    
    async def fail_job(
        self,
        job_id: str,
        error: str
    ) -> bool:
        """
        Mark a job as failed.
        
        Args:
            job_id: Job identifier
            error: Error message
            
        Returns:
            True if job was marked failed, False if not found
        """
        async with self._queue_lock:
            if job_id not in self._queue:
                return False
            
            job = self._queue[job_id]
            job["status"] = JobStatus.FAILED
            job["status_message"] = f"Failed: {error}"
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"
            job["error"] = error
            self._save_queue()
            return True
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled, False if not found or already running
        """
        async with self._queue_lock:
            if job_id not in self._queue:
                return False
            
            job = self._queue[job_id]
            
            # Can only cancel queued jobs
            if job["status"] != JobStatus.QUEUED:
                return False
            
            job["status"] = JobStatus.CANCELLED
            job["status_message"] = "Cancelled by user"
            job["completed_at"] = datetime.utcnow().isoformat() + "Z"
            self._save_queue()
            return True
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job information.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job data dict or None if not found
        """
        async with self._queue_lock:
            return self._queue.get(job_id)
    
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """
        List all jobs, optionally filtered.
        
        Args:
            status: Filter by status
            job_type: Filter by job type
            
        Returns:
            List of job data dicts
        """
        async with self._queue_lock:
            jobs = list(self._queue.values())
            
            if status:
                jobs = [j for j in jobs if j["status"] == status]
            
            if job_type:
                jobs = [j for j in jobs if j["job_type"] == job_type]
            
            # Sort by priority (desc) then queued_at (asc)
            jobs.sort(key=lambda x: (-x.get("priority", 0), x.get("queued_at", "")))
            
            return jobs
    
    async def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get position of job in queue.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Position (1-indexed) or None if not found or not queued
        """
        queued_jobs = await self.list_jobs(status=JobStatus.QUEUED)
        
        for i, job in enumerate(queued_jobs):
            if job["job_id"] == job_id:
                return i + 1
        
        return None
    
    async def get_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Get next job to process (highest priority, oldest).
        
        Returns:
            Job data dict or None if queue is empty
        """
        queued_jobs = await self.list_jobs(status=JobStatus.QUEUED)
        
        if not queued_jobs:
            return None
        
        # First job in sorted list (highest priority, oldest)
        return queued_jobs[0]
    
    async def update_job_status_message(
        self,
        job_id: str,
        message: str
    ) -> bool:
        """
        Update job status message (for progress updates).
        
        Args:
            job_id: Job identifier
            message: Status message
            
        Returns:
            True if updated, False if not found
        """
        async with self._queue_lock:
            if job_id not in self._queue:
                return False
            
            self._queue[job_id]["status_message"] = message
            self._save_queue()
            return True
    
    async def has_running_job(self, job_type: Optional[JobType] = None) -> bool:
        """
        Check if there are any running jobs, optionally of a specific type.
        
        Args:
            job_type: Optional job type to filter by
            
        Returns:
            True if there are running jobs (of the specified type)
        """
        running_jobs = await self.list_jobs(
            status=JobStatus.RUNNING,
            job_type=job_type.value if job_type else None
        )
        return len(running_jobs) > 0
    
    async def get_running_job(self, job_type: Optional[JobType] = None) -> Optional[Dict[str, Any]]:
        """
        Get currently running job, optionally of a specific type.
        
        Args:
            job_type: Optional job type to filter by
            
        Returns:
            Running job dict or None
        """
        running_jobs = await self.list_jobs(
            status=JobStatus.RUNNING,
            job_type=job_type.value if job_type else None
        )
        return running_jobs[0] if running_jobs else None
    
    async def get_next_job_by_type(self, job_type: JobType) -> Optional[Dict[str, Any]]:
        """
        Get next queued job of a specific type.
        
        Args:
            job_type: Type of job to get
            
        Returns:
            Job data dict or None if no queued jobs of this type
        """
        queued_jobs = await self.list_jobs(
            status=JobStatus.QUEUED,
            job_type=job_type.value
        )
        
        if not queued_jobs:
            return None
        
        # First job in sorted list (highest priority, oldest)
        return queued_jobs[0]


# Global queue instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
