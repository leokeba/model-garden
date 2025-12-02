"""Tests for model_garden.queue module."""

from pathlib import Path

import pytest

from model_garden.queue.job_queue import JobQueue, JobStatus, JobType, get_job_queue


@pytest.fixture
def temp_queue(temp_dir: Path) -> JobQueue:
    """Create a JobQueue instance with temporary storage."""
    queue = JobQueue()
    queue._storage_file = temp_dir / "job_queue.json"
    queue._queue = {}
    return queue


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self):
        """Test JobStatus enum values."""
        assert JobStatus.QUEUED == "queued"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_status_is_string(self):
        """Test JobStatus is a string enum."""
        assert isinstance(JobStatus.QUEUED, str)
        assert JobStatus.QUEUED.value == "queued"


class TestJobType:
    """Tests for JobType enum."""

    def test_type_values(self):
        """Test JobType enum values."""
        assert JobType.TRAINING == "training"
        assert JobType.MODEL_LOADING == "model_loading"
        assert JobType.MODEL_UNLOADING == "model_unloading"
        assert JobType.DATASET_PROCESSING == "dataset_processing"


class TestJobQueue:
    """Tests for JobQueue class."""

    @pytest.mark.asyncio
    async def test_add_job(self, temp_queue: JobQueue):
        """Test adding a job to the queue."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={"model": "test-model"},
            priority=1,
        )

        job = await temp_queue.get_job("test-123")
        assert job is not None
        assert job["job_id"] == "test-123"
        assert job["job_type"] == "training"
        assert job["status"] == JobStatus.QUEUED
        assert job["priority"] == 1

    @pytest.mark.asyncio
    async def test_start_job(self, temp_queue: JobQueue):
        """Test starting a queued job."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={},
        )

        result = await temp_queue.start_job("test-123")
        assert result is True

        job = await temp_queue.get_job("test-123")
        assert job["status"] == JobStatus.RUNNING
        assert job["started_at"] is not None

    @pytest.mark.asyncio
    async def test_start_job_not_found(self, temp_queue: JobQueue):
        """Test starting a non-existent job returns False."""
        result = await temp_queue.start_job("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_start_job_already_running(self, temp_queue: JobQueue):
        """Test starting an already running job returns False."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={},
        )
        await temp_queue.start_job("test-123")

        # Try to start again
        result = await temp_queue.start_job("test-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_job(self, temp_queue: JobQueue):
        """Test completing a job."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={},
        )
        await temp_queue.start_job("test-123")

        result = await temp_queue.complete_job("test-123", result={"output": "success"})
        assert result is True

        job = await temp_queue.get_job("test-123")
        assert job["status"] == JobStatus.COMPLETED
        assert job["completed_at"] is not None
        assert job["result"] == {"output": "success"}

    @pytest.mark.asyncio
    async def test_fail_job(self, temp_queue: JobQueue):
        """Test failing a job."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={},
        )
        await temp_queue.start_job("test-123")

        result = await temp_queue.fail_job("test-123", error="Something went wrong")
        assert result is True

        job = await temp_queue.get_job("test-123")
        assert job["status"] == JobStatus.FAILED
        assert job["error"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_cancel_job_queued(self, temp_queue: JobQueue):
        """Test cancelling a queued job."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={},
        )

        result = await temp_queue.cancel_job("test-123")
        assert result is True

        job = await temp_queue.get_job("test-123")
        assert job["status"] == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_job_running(self, temp_queue: JobQueue):
        """Test cancelling a running job."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={},
        )
        await temp_queue.start_job("test-123")

        result = await temp_queue.cancel_job("test-123")
        assert result is True

        job = await temp_queue.get_job("test-123")
        assert job["status"] == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_job_completed(self, temp_queue: JobQueue):
        """Test that completed jobs cannot be cancelled."""
        await temp_queue.add_job(
            job_id="test-123",
            job_type=JobType.TRAINING.value,
            job_config={},
        )
        await temp_queue.start_job("test-123")
        await temp_queue.complete_job("test-123")

        result = await temp_queue.cancel_job("test-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_jobs(self, temp_queue: JobQueue):
        """Test listing all jobs."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {})
        await temp_queue.add_job("job-2", JobType.MODEL_LOADING.value, {})
        await temp_queue.add_job("job-3", JobType.TRAINING.value, {})

        jobs = await temp_queue.list_jobs()
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_status(self, temp_queue: JobQueue):
        """Test listing jobs filtered by status."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {})
        await temp_queue.add_job("job-2", JobType.TRAINING.value, {})
        await temp_queue.start_job("job-2")

        queued_jobs = await temp_queue.list_jobs(status=JobStatus.QUEUED)
        assert len(queued_jobs) == 1
        assert queued_jobs[0]["job_id"] == "job-1"

        running_jobs = await temp_queue.list_jobs(status=JobStatus.RUNNING)
        assert len(running_jobs) == 1
        assert running_jobs[0]["job_id"] == "job-2"

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_type(self, temp_queue: JobQueue):
        """Test listing jobs filtered by type."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {})
        await temp_queue.add_job("job-2", JobType.MODEL_LOADING.value, {})

        training_jobs = await temp_queue.list_jobs(job_type=JobType.TRAINING.value)
        assert len(training_jobs) == 1
        assert training_jobs[0]["job_id"] == "job-1"

    @pytest.mark.asyncio
    async def test_list_jobs_sorted_by_priority(self, temp_queue: JobQueue):
        """Test that jobs are sorted by priority (descending)."""
        await temp_queue.add_job("job-low", JobType.TRAINING.value, {}, priority=1)
        await temp_queue.add_job("job-high", JobType.TRAINING.value, {}, priority=10)
        await temp_queue.add_job("job-medium", JobType.TRAINING.value, {}, priority=5)

        jobs = await temp_queue.list_jobs()
        assert jobs[0]["job_id"] == "job-high"
        assert jobs[1]["job_id"] == "job-medium"
        assert jobs[2]["job_id"] == "job-low"

    @pytest.mark.asyncio
    async def test_get_queue_position(self, temp_queue: JobQueue):
        """Test getting queue position."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {}, priority=1)
        await temp_queue.add_job("job-2", JobType.TRAINING.value, {}, priority=2)
        await temp_queue.add_job("job-3", JobType.TRAINING.value, {}, priority=1)

        # job-2 has higher priority, so it's first
        pos = await temp_queue.get_queue_position("job-2")
        assert pos == 1

        # job-1 was added before job-3 with same priority
        pos = await temp_queue.get_queue_position("job-1")
        assert pos == 2

    @pytest.mark.asyncio
    async def test_get_queue_position_not_queued(self, temp_queue: JobQueue):
        """Test queue position for non-queued job."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {})
        await temp_queue.start_job("job-1")

        pos = await temp_queue.get_queue_position("job-1")
        assert pos is None

    @pytest.mark.asyncio
    async def test_get_next_job(self, temp_queue: JobQueue):
        """Test getting next job to process."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {}, priority=1)
        await temp_queue.add_job("job-2", JobType.TRAINING.value, {}, priority=5)

        next_job = await temp_queue.get_next_job()
        assert next_job is not None
        assert next_job["job_id"] == "job-2"  # Higher priority

    @pytest.mark.asyncio
    async def test_get_next_job_empty_queue(self, temp_queue: JobQueue):
        """Test getting next job from empty queue."""
        next_job = await temp_queue.get_next_job()
        assert next_job is None

    @pytest.mark.asyncio
    async def test_update_job_status_message(self, temp_queue: JobQueue):
        """Test updating job status message."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {})

        result = await temp_queue.update_job_status_message("job-1", "Processing step 1")
        assert result is True

        job = await temp_queue.get_job("job-1")
        assert job["status_message"] == "Processing step 1"

    @pytest.mark.asyncio
    async def test_has_running_job(self, temp_queue: JobQueue):
        """Test checking for running jobs."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {})

        # No running jobs yet
        assert await temp_queue.has_running_job() is False

        await temp_queue.start_job("job-1")

        # Now there's a running job
        assert await temp_queue.has_running_job() is True
        assert await temp_queue.has_running_job(JobType.TRAINING) is True
        assert await temp_queue.has_running_job(JobType.MODEL_LOADING) is False

    @pytest.mark.asyncio
    async def test_get_running_job(self, temp_queue: JobQueue):
        """Test getting running job."""
        await temp_queue.add_job("job-1", JobType.TRAINING.value, {})
        await temp_queue.start_job("job-1")

        running_job = await temp_queue.get_running_job()
        assert running_job is not None
        assert running_job["job_id"] == "job-1"

    @pytest.mark.asyncio
    async def test_get_next_job_by_type(self, temp_queue: JobQueue):
        """Test getting next job by type."""
        await temp_queue.add_job("job-training", JobType.TRAINING.value, {})
        await temp_queue.add_job("job-loading", JobType.MODEL_LOADING.value, {})

        next_training = await temp_queue.get_next_job_by_type(JobType.TRAINING)
        assert next_training is not None
        assert next_training["job_id"] == "job-training"

        next_loading = await temp_queue.get_next_job_by_type(JobType.MODEL_LOADING)
        assert next_loading is not None
        assert next_loading["job_id"] == "job-loading"

    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir: Path):
        """Test that queue persists to disk."""
        storage_file = temp_dir / "job_queue.json"

        # Create queue and add job
        queue1 = JobQueue()
        queue1._storage_file = storage_file
        queue1._queue = {}
        await queue1.add_job("job-1", JobType.TRAINING.value, {"data": "test"})

        # Verify file was created
        assert storage_file.exists()

        # Create new queue instance and verify data loaded
        queue2 = JobQueue()
        queue2._storage_file = storage_file
        queue2._queue = {}
        queue2._load_queue()

        job = queue2._queue.get("job-1")
        assert job is not None
        assert job["job_config"] == {"data": "test"}


class TestGetJobQueue:
    """Tests for get_job_queue function."""

    def test_get_job_queue_singleton(self):
        """Test that get_job_queue returns singleton instance."""
        # Note: This test may not be reliable in isolation since the global
        # instance may already exist from other tests
        queue1 = get_job_queue()
        queue2 = get_job_queue()
        assert queue1 is queue2
