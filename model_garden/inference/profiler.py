"""GPU Memory Profiler for inference model loading.

This module provides detailed GPU memory profiling during model loading,
tracking memory usage at different phases to show breakdown of:
- Model weights
- KV cache allocation
- CUDA graphs (if enabled)
- Other buffers and activations

We capture the real memory values that vLLM logs during engine initialization:
- "Model loading took X GiB memory"
- "Available KV cache memory: X GiB"
- "Graph capturing finished in X secs, took X GiB"
"""

from __future__ import annotations

import gc
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from model_garden.utils.console import console


def _bytes_to_gb(bytes_val: int | float) -> float:
    """Convert bytes to GB."""
    return bytes_val / (1024**3)


def _gib_to_gb(gib: float) -> float:
    """Convert GiB to GB (decimal)."""
    # 1 GiB = 1.073741824 GB
    return gib * 1.073741824


def _format_gb(gb: float) -> str:
    """Format GB value for display."""
    return f"{gb:.2f} GB"


def _format_percent(value: float, total: float) -> str:
    """Format value as percentage of total."""
    if total <= 0:
        return "N/A"
    return f"{(value / total) * 100:.1f}%"


def _get_gpu_memory_pynvml(device: int = 0) -> tuple[int, int, int]:
    """Get GPU memory using pynvml (works for all processes).

    Returns:
        Tuple of (total_bytes, used_bytes, free_bytes)
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(mem_info.total), int(mem_info.used), int(mem_info.free)
    except Exception:
        return 0, 0, 0


class VLLMLogCapture:
    """Captures vLLM memory statistics from systemd journal.

    vLLM logs memory stats during initialization in a subprocess, so we can't
    capture them directly via Python logging. Instead, we read the systemd
    journal after model loading completes.

    Expected log patterns:
    - "Model loading took 0.7523 GiB memory and 1.161985 seconds"
    - "Available KV cache memory: 13.21 GiB"
    - "GPU KV cache size: 629,808 tokens"
    - "Graph capturing finished in 1 secs, took 0.15 GiB"
    - "Maximum concurrency for 2,048 tokens per request: 307.52x"
    """

    # Regex patterns for vLLM memory logs
    PATTERNS = {
        "model_memory": re.compile(r"Model loading took ([\d.]+) GiB memory"),
        "kv_cache_memory": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
        "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
        "cuda_graphs": re.compile(r"Graph capturing finished in \d+ secs?, took ([\d.]+) GiB"),
        "max_concurrency": re.compile(
            r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"
        ),
    }

    def __init__(self):
        self.model_memory_gib: float = 0.0
        self.kv_cache_memory_gib: float = 0.0
        self.kv_cache_tokens: int = 0
        self.cuda_graphs_gib: float = 0.0
        self.max_concurrency: float = 0.0
        self._start_time: float = 0.0

    def start(self):
        """Record start time for journal query."""
        self._start_time = time.time()

    def stop(self):
        """Read systemd journal and parse vLLM logs."""
        self._read_journal_logs()

    def _read_journal_logs(self):
        """Read recent systemd journal entries for model-garden service."""

        try:
            # Get logs from the last 60 seconds for model-garden service
            result = subprocess.run(
                [
                    "journalctl",
                    "-u",
                    "model-garden.service",
                    "--since",
                    "-60s",
                    "--no-pager",
                    "-q",  # quiet, no metadata
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._parse_logs(result.stdout)
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not read journal: {e}[/yellow]")

    def _parse_logs(self, log_content: str):
        """Parse vLLM logs to extract memory values."""
        for line in log_content.split("\n"):
            # Model memory
            match = self.PATTERNS["model_memory"].search(line)
            if match:
                self.model_memory_gib = float(match.group(1))

            # KV cache memory
            match = self.PATTERNS["kv_cache_memory"].search(line)
            if match:
                self.kv_cache_memory_gib = float(match.group(1))

            # KV cache tokens
            match = self.PATTERNS["kv_cache_tokens"].search(line)
            if match:
                self.kv_cache_tokens = int(match.group(1).replace(",", ""))

            # CUDA graphs
            match = self.PATTERNS["cuda_graphs"].search(line)
            if match:
                self.cuda_graphs_gib = float(match.group(1))

            # Max concurrency
            match = self.PATTERNS["max_concurrency"].search(line)
            if match:
                self.max_concurrency = float(match.group(1))

    def get_stats(self) -> dict[str, Any]:
        """Get parsed memory statistics."""
        return {
            "model_memory_gib": self.model_memory_gib,
            "kv_cache_memory_gib": self.kv_cache_memory_gib,
            "kv_cache_tokens": self.kv_cache_tokens,
            "cuda_graphs_gib": self.cuda_graphs_gib,
            "max_concurrency": self.max_concurrency,
        }


@dataclass
class MemorySnapshot:
    """A snapshot of GPU memory at a specific point in time."""

    timestamp: float = 0.0
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    total_bytes: int = 0
    # System-wide GPU memory (from pynvml, captures all processes)
    system_used_bytes: int = 0
    label: str = ""

    @property
    def allocated_gb(self) -> float:
        return _bytes_to_gb(self.allocated_bytes)

    @property
    def reserved_gb(self) -> float:
        return _bytes_to_gb(self.reserved_bytes)

    @property
    def total_gb(self) -> float:
        return _bytes_to_gb(self.total_bytes)

    @property
    def free_gb(self) -> float:
        return _bytes_to_gb(self.total_bytes - self.reserved_bytes)

    @property
    def system_used_gb(self) -> float:
        """System-wide GPU memory usage (captures vLLM worker processes)."""
        return _bytes_to_gb(self.system_used_bytes)


@dataclass
class MemoryProfile:
    """Complete memory profile with breakdown of components."""

    # Hardware info
    gpu_name: str = ""
    total_memory_gb: float = 0.0

    # Snapshots at different phases
    baseline: MemorySnapshot | None = None
    final: MemorySnapshot | None = None

    # Calculated breakdown (in GB) - from vLLM logs
    weights_gb: float = 0.0
    kv_cache_gb: float = 0.0
    cuda_graphs_gb: float = 0.0
    other_gb: float = 0.0

    # Additional info
    model_path: str = ""
    max_model_len: int = 0
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9

    # KV cache details (from vLLM logs)
    kv_cache_tokens: int = 0
    max_concurrency: float = 0.0

    # Timing
    load_time_seconds: float = 0.0

    # Raw weight file size (from disk)
    weight_file_size_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "gpu": {
                "name": self.gpu_name,
                "total_memory_gb": round(self.total_memory_gb, 2),
            },
            "breakdown": {
                "weights_gb": round(self.weights_gb, 2),
                "kv_cache_gb": round(self.kv_cache_gb, 2),
                "cuda_graphs_gb": round(self.cuda_graphs_gb, 2),
                "other_gb": round(self.other_gb, 2),
                "total_used_gb": round(self.total_used_gb, 2),
                "available_gb": round(self.available_gb, 2),
            },
            "utilization": {
                "used_percent": round(self.used_percent, 1),
                "available_percent": round(100 - self.used_percent, 1),
            },
            "config": {
                "model_path": self.model_path,
                "max_model_len": self.max_model_len,
                "enforce_eager": self.enforce_eager,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            },
            "kv_cache": {
                "tokens": self.kv_cache_tokens,
                "max_concurrency": round(self.max_concurrency, 1),
            },
            "timing": {
                "load_time_seconds": round(self.load_time_seconds, 2),
            },
            "weight_file_size_gb": round(self.weight_file_size_gb, 2),
        }

    @property
    def total_used_gb(self) -> float:
        """Total GPU memory currently in use by model."""
        # Use system-wide GPU memory if available (captures vLLM workers)
        if self.final and self.baseline:
            model_memory = self.final.system_used_gb - self.baseline.system_used_gb
            if model_memory > 0:
                return model_memory
        # Fallback to component breakdown
        return self.weights_gb + self.kv_cache_gb + self.cuda_graphs_gb + self.other_gb

    @property
    def available_gb(self) -> float:
        """Available GPU memory."""
        return max(0, self.total_memory_gb - self.total_used_gb)

    @property
    def used_percent(self) -> float:
        """Percentage of GPU memory in use."""
        if self.total_memory_gb <= 0:
            return 0.0
        return (self.total_used_gb / self.total_memory_gb) * 100


class GPUMemoryProfiler:
    """Profiles GPU memory usage during model loading.

    Captures vLLM's logged memory statistics to get accurate breakdowns:
    - Model weights memory
    - KV cache memory allocation
    - CUDA graphs memory (if enabled)

    Usage:
        profiler = GPUMemoryProfiler()
        profiler.start()

        # Load model via vLLM (this logs memory stats)
        engine = AsyncLLM.from_engine_args(args)

        # Get final profile with real numbers from vLLM logs
        profile = profiler.get_profile(model_path=...)
        profiler.print_summary()
    """

    def __init__(self, device: int = 0):
        """Initialize the profiler.

        Args:
            device: CUDA device index to profile
        """
        self.device = device
        self.snapshots: dict[str, MemorySnapshot] = {}
        self.profile: MemoryProfile | None = None
        self._start_time: float | None = None
        self._log_capture: VLLMLogCapture | None = None

        # Get GPU info
        self._init_gpu_info()

    def _init_gpu_info(self) -> None:
        """Initialize GPU information."""
        try:
            import torch

            if torch.cuda.is_available():
                self.gpu_name = torch.cuda.get_device_name(self.device)
                props = torch.cuda.get_device_properties(self.device)
                self.total_memory_bytes = props.total_memory
            else:
                self.gpu_name = "No GPU"
                self.total_memory_bytes = 0
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not get GPU info: {e}[/yellow]")
            self.gpu_name = "Unknown"
            self.total_memory_bytes = 0

    def _get_memory_snapshot(self, label: str = "") -> MemorySnapshot:
        """Get current GPU memory state."""
        try:
            import torch

            # Force garbage collection for accurate measurements
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

            # PyTorch allocator stats (only shows memory in current process)
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)

            # System-wide GPU memory (captures vLLM worker processes)
            total, system_used, _ = _get_gpu_memory_pynvml(self.device)
            if total == 0:
                total = self.total_memory_bytes

            return MemorySnapshot(
                timestamp=time.time(),
                allocated_bytes=allocated,
                reserved_bytes=reserved,
                total_bytes=total,
                system_used_bytes=system_used,
                label=label,
            )
        except Exception as e:
            console.print(f"[yellow]⚠️ Could not get memory snapshot: {e}[/yellow]")
            return MemorySnapshot(label=label)

    def start(self) -> None:
        """Start profiling session and log capture."""
        self._start_time = time.time()
        self.snapshots.clear()
        self.profile = None

        # Start capturing vLLM logs
        self._log_capture = VLLMLogCapture()
        self._log_capture.start()

        # Reset peak stats for accurate tracking
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass

        self.snapshot("baseline")

    def snapshot(self, label: str) -> MemorySnapshot:
        """Take a memory snapshot with the given label.

        Args:
            label: Label for this snapshot (e.g., "baseline", "final")

        Returns:
            The memory snapshot
        """
        snap = self._get_memory_snapshot(label)
        self.snapshots[label] = snap
        return snap

    def get_profile(
        self,
        model_path: str = "",
        max_model_len: int = 0,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        weight_file_size_gb: float = 0.0,
    ) -> MemoryProfile:
        """Calculate and return the memory profile.

        Args:
            model_path: Path to the model (for metadata)
            max_model_len: Maximum model length configured
            enforce_eager: Whether CUDA graphs are disabled
            tensor_parallel_size: Number of GPUs used
            gpu_memory_utilization: vLLM's gpu_memory_utilization setting
            weight_file_size_gb: Size of weight files on disk

        Returns:
            Complete memory profile with breakdown from vLLM logs
        """
        # Take final snapshot
        self.snapshot("final")

        # Stop log capture and parse results
        vllm_stats: dict[str, Any] = {}
        if self._log_capture:
            self._log_capture.stop()
            vllm_stats = self._log_capture.get_stats()

        # Calculate load time
        load_time = 0.0
        if self._start_time:
            load_time = time.time() - self._start_time

        # Get snapshots
        baseline = self.snapshots.get("baseline")
        final = self.snapshots.get("final")

        # Get real memory values from vLLM logs (in GiB, convert to GB)
        weights_gb = _gib_to_gb(vllm_stats.get("model_memory_gib", 0.0))
        kv_cache_gb = _gib_to_gb(vllm_stats.get("kv_cache_memory_gib", 0.0))
        cuda_graphs_gb = _gib_to_gb(vllm_stats.get("cuda_graphs_gib", 0.0))
        kv_cache_tokens = vllm_stats.get("kv_cache_tokens", 0)
        max_concurrency = vllm_stats.get("max_concurrency", 0.0)

        # Calculate total model memory from system measurement
        total_model_memory = 0.0
        if final and baseline:
            total_model_memory = final.system_used_gb - baseline.system_used_gb

        # Calculate "other" memory (difference between measured total and logged components)
        component_total = weights_gb + kv_cache_gb + cuda_graphs_gb
        other_gb = max(0, total_model_memory - component_total) if total_model_memory > 0 else 0.0

        self.profile = MemoryProfile(
            gpu_name=self.gpu_name,
            total_memory_gb=_bytes_to_gb(self.total_memory_bytes),
            baseline=baseline,
            final=final,
            weights_gb=weights_gb,
            kv_cache_gb=kv_cache_gb,
            cuda_graphs_gb=cuda_graphs_gb,
            other_gb=other_gb,
            model_path=model_path,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_tokens=kv_cache_tokens,
            max_concurrency=max_concurrency,
            load_time_seconds=load_time,
            weight_file_size_gb=weight_file_size_gb,
        )

        return self.profile

    def print_summary(self) -> None:
        """Print a formatted summary of the memory profile."""
        if not self.profile:
            console.print("[yellow]No profile available. Call get_profile() first.[/yellow]")
            return

        p = self.profile

        console.print("\n[bold cyan]📊 GPU Memory Profile[/bold cyan]")
        console.print(f"[dim]{'─' * 50}[/dim]")

        # GPU info
        console.print(f"[bold]GPU:[/bold] {p.gpu_name}")
        console.print(f"[bold]Total Memory:[/bold] {_format_gb(p.total_memory_gb)}")
        console.print(f"[dim]{'─' * 50}[/dim]")

        # Breakdown
        console.print("[bold]Memory Breakdown:[/bold]")

        # Model weights
        weight_info = f"{_format_gb(p.weights_gb)}"
        if p.weight_file_size_gb > 0:
            weight_info += f" (files: {_format_gb(p.weight_file_size_gb)})"
        console.print(f"  ├── Model Weights:    {weight_info}")

        # KV Cache
        kv_info = f"{_format_gb(p.kv_cache_gb)}"
        if p.kv_cache_tokens > 0:
            kv_info += f" ({p.kv_cache_tokens:,} tokens)"
        console.print(f"  ├── KV Cache:         {kv_info}")

        # CUDA Graphs
        cuda_status = "(disabled)" if p.enforce_eager else ""
        console.print(f"  ├── CUDA Graphs:      {_format_gb(p.cuda_graphs_gb)} {cuda_status}")

        # Other
        console.print(f"  └── Other/Buffers:    {_format_gb(p.other_gb)}")

        console.print(f"[dim]{'─' * 50}[/dim]")

        # Totals
        console.print(
            f"[bold]Total Used:[/bold]      {_format_gb(p.total_used_gb)} ({p.used_percent:.1f}%)"
        )
        console.print(
            f"[bold]Available:[/bold]       {_format_gb(p.available_gb)} ({100 - p.used_percent:.1f}%)"
        )

        # KV cache info
        if p.max_concurrency > 0:
            console.print(
                f"[bold]Max Concurrency:[/bold] {p.max_concurrency:.1f}x (for {p.max_model_len:,} tokens)"
            )

        if p.load_time_seconds > 0:
            console.print(f"[bold]Load Time:[/bold]       {p.load_time_seconds:.2f}s")

        console.print(f"[dim]{'─' * 50}[/dim]\n")


# Global profiler instance for the current model
_current_profile: MemoryProfile | None = None


def get_current_memory_profile() -> MemoryProfile | None:
    """Get the memory profile from the last model load."""
    return _current_profile


def set_current_memory_profile(profile: MemoryProfile | None) -> None:
    """Set the current memory profile."""
    global _current_profile
    _current_profile = profile


def get_live_gpu_stats() -> dict[str, Any]:
    """Get live GPU memory statistics.

    Returns:
        Dictionary with current GPU memory usage stats
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"error": "No GPU available"}

        device = 0
        props = torch.cuda.get_device_properties(device)

        # Use pynvml for accurate system-wide memory (captures vLLM workers)
        total, used, free = _get_gpu_memory_pynvml(device)
        if total == 0:
            # Fallback to PyTorch stats
            total = props.total_memory
            used = torch.cuda.memory_allocated(device)
            free = total - used

        # Peak stats from PyTorch (only for current process)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)

        return {
            "gpu_name": torch.cuda.get_device_name(device),
            "total_gb": round(_bytes_to_gb(total), 2),
            "allocated_gb": round(_bytes_to_gb(used), 2),
            "reserved_gb": round(_bytes_to_gb(torch.cuda.memory_reserved(device)), 2),
            "free_gb": round(_bytes_to_gb(free), 2),
            "peak_allocated_gb": round(_bytes_to_gb(max_allocated), 2),
            "peak_reserved_gb": round(_bytes_to_gb(max_reserved), 2),
            "utilization_percent": round((used / total) * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}
