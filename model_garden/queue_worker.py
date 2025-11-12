"""Background worker for processing jobs from the queue sequentially.

This module provides a queue worker that monitors the job queue and automatically
starts jobs one at a time, ensuring sequential execution and preventing resource
conflicts (especially GPU memory).
"""

import asyncio
import logging
import os
import threading
from typing import Optional

from model_garden.job_queue import get_job_queue, JobType, JobStatus

logger = logging.getLogger(__name__)


class QueueWorker:
    """Background worker that processes jobs from the queue sequentially.
    
    The worker runs as an asyncio task and continuously monitors the job queue.
    When a job completes, it automatically starts the next queued job, ensuring
    only a limited number of jobs run concurrently (default: 1 for training jobs).
    
    Features:
    - Sequential execution (one training job at a time by default)
    - Automatic job processing without manual intervention
    - Graceful startup/shutdown
    - Configurable concurrency limits
    - Support for different job types
    """
    
    def __init__(
        self, 
        max_concurrent_training_jobs: int = 1,
        poll_interval: float = 5.0,
        enabled: bool = True
    ):
        """Initialize queue worker.
        
        Args:
            max_concurrent_training_jobs: Maximum number of training jobs to run concurrently
            poll_interval: Seconds to wait between queue checks
            enabled: Whether the worker is enabled (for feature flagging)
        """
        self.max_concurrent = max_concurrent_training_jobs
        self.poll_interval = poll_interval
        self.enabled = enabled
        self.running = False
        self.worker_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"QueueWorker initialized: max_concurrent={max_concurrent_training_jobs}, "
            f"poll_interval={poll_interval}s, enabled={enabled}"
        )
    
    async def start(self):
        """Start the queue worker.
        
        Creates and starts the background asyncio task that processes the queue.
        If the worker is already running or disabled, this is a no-op.
        """
        if not self.enabled:
            logger.info("⚠️  Queue worker is disabled (set MODEL_GARDEN_QUEUE_WORKER_ENABLED=true to enable)")
            print("⚠️  Queue worker is disabled (set MODEL_GARDEN_QUEUE_WORKER_ENABLED=true to enable)", flush=True)
            return
        
        if self.running:
            logger.warning("Queue worker already running")
            print("⚠️  Queue worker already running", flush=True)
            return
        
        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("🚀 Queue worker started")
        print("🚀 Queue worker started", flush=True)
    
    async def stop(self):
        """Stop the queue worker gracefully.
        
        Cancels the background task and waits for it to complete. Running jobs
        will continue to completion, but no new jobs will be started.
        """
        if not self.running:
            return
        
        logger.info("🛑 Stopping queue worker...")
        print("🛑 Stopping queue worker...")
        self.running = False
        
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✓ Queue worker stopped")
        print("✓ Queue worker stopped")
    
    async def _worker_loop(self):
        """Main worker loop that processes queued jobs.
        
        This loop continuously:
        1. Checks how many training jobs are currently running
        2. If under the limit, gets the next queued job
        3. Starts the job in a background thread
        4. Waits before checking again
        
        The loop runs until stop() is called.
        """
        queue = get_job_queue()
        logger.info("🔄 Queue worker loop started")
        print("🔄 Queue worker loop started")
        
        while self.running:
            try:
                # Check if we can start a new training job
                running_jobs = await queue.list_jobs(
                    status=JobStatus.RUNNING, 
                    job_type=JobType.TRAINING.value
                )
                running_count = len(running_jobs)
                
                if running_count >= self.max_concurrent:
                    # Wait for current jobs to complete
                    if running_count > 0:
                        logger.debug(
                            f"Queue worker: {running_count} training job(s) running, "
                            f"waiting for completion..."
                        )
                    await asyncio.sleep(self.poll_interval)
                    continue
                
                # Get next queued training job
                next_job = await queue.get_next_job_by_type(JobType.TRAINING)
                
                if next_job is None:
                    # No jobs in queue, wait
                    logger.debug("Queue worker: No jobs in queue")
                    await asyncio.sleep(self.poll_interval)
                    continue
                
                # Start the job
                job_id = next_job["job_id"]
                job_name = next_job.get("job_config", {}).get("name", "Unnamed Job")
                logger.info(f"🎯 Queue worker starting job: {job_id} ({job_name})")
                print(f"🎯 Queue worker starting job: {job_id} ({job_name})")
                
                # Import here to avoid circular dependency
                from model_garden.api import run_training_job
                
                # Run in a separate daemon thread to avoid blocking the worker
                # The thread will execute the training job independently
                thread = threading.Thread(
                    target=run_training_job, 
                    args=(job_id,),
                    daemon=True,
                    name=f"training-job-{job_id}"
                )
                thread.start()
                
                # Wait a bit before checking queue again to allow job to transition
                # from QUEUED to RUNNING in the job execution code
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                # Worker is being stopped
                logger.info("Queue worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Queue worker error: {e}", exc_info=True)
                # Wait before retrying to avoid tight error loop
                await asyncio.sleep(self.poll_interval)
        
        logger.info("Queue worker loop exited")


# Global worker instance
_queue_worker: Optional[QueueWorker] = None


def get_queue_worker() -> QueueWorker:
    """Get the global queue worker instance.
    
    Creates the worker on first access with configuration from environment variables.
    
    Returns:
        The global QueueWorker instance
    """
    global _queue_worker
    if _queue_worker is None:
        # Read configuration from environment
        max_concurrent = int(os.getenv("MODEL_GARDEN_MAX_CONCURRENT_TRAINING_JOBS", "1"))
        poll_interval = float(os.getenv("MODEL_GARDEN_QUEUE_POLL_INTERVAL", "5.0"))
        enabled = os.getenv("MODEL_GARDEN_QUEUE_WORKER_ENABLED", "true").lower() == "true"
        
        _queue_worker = QueueWorker(
            max_concurrent_training_jobs=max_concurrent,
            poll_interval=poll_interval,
            enabled=enabled
        )
    return _queue_worker
