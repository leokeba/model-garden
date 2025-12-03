# System routes
"""
Routes for system management:
- GET /api/v1/system/status - Get system status (CPU, GPU, memory)
- GET /api/v1/system/settings - Get system settings including optional dependencies
- POST /api/v1/system/cleanup - Force GPU memory cleanup
- POST /api/v1/system/unsloth/install - Install Unsloth package
- POST /api/v1/system/unsloth/uninstall - Uninstall Unsloth package
- POST /api/v1/system/restart - Restart the Model Garden service
"""

import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

router = APIRouter(prefix="/api/v1/system", tags=["system"])

# Track package installation status
_package_operation_status: dict = {
    "in_progress": False,
    "operation": None,
    "output": [],
    "success": None,
    "error": None,
}


@router.get("/status")
async def system_status():
    """Get system status information."""
    import psutil
    import torch

    # GPU information with detailed metrics
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            import pynvml

            pynvml.nvmlInit()

            device_count = torch.cuda.device_count()
            gpus = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Get GPU info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Utilization info
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    mem_util = utilization.memory
                except Exception:
                    gpu_util = None
                    mem_util = None

                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temperature = None

                # Power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except Exception:
                    power_usage = None
                    power_limit = None

                gpus.append(
                    {
                        "id": i,
                        "name": name,
                        "memory": {
                            "total": mem_info.total,
                            "used": mem_info.used,
                            "free": mem_info.free,
                            "used_percent": round(
                                (float(mem_info.used) / float(mem_info.total)) * 100, 1
                            ),
                        },
                        "utilization": {
                            "gpu": gpu_util,
                            "memory": mem_util,
                        },
                        "temperature": temperature,
                        "power": {
                            "usage": power_usage,
                            "limit": power_limit,
                        }
                        if power_usage is not None
                        else None,
                    }
                )

            pynvml.nvmlShutdown()

            gpu_info = {
                "available": True,
                "device_count": device_count,
                "devices": gpus,
            }

        except Exception as e:
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

    from ..storage import get_storage_manager

    storage = get_storage_manager()
    models_storage = storage.load_models()
    training_jobs = storage.load_training_jobs()

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
            "active_jobs": len(
                [j for j in training_jobs.values() if j["status"] in ["running", "queued"]]
            ),
        },
    }


@router.get("/backends")
async def list_training_backends():
    """List all available training backends with their capabilities."""
    from model_garden.training.backends import list_backends

    backends = list_backends()

    return {
        "success": True,
        "data": backends,
        "total": len(backends),
        "message": f"{len(backends)} training backends available",
    }


@router.post("/cleanup")
async def cleanup_gpu_memory():
    """Force cleanup of GPU memory and Python garbage collection."""
    import gc

    result = {
        "success": True,
        "actions": [],
        "gpu_memory_before": None,
        "gpu_memory_after": None,
    }

    try:
        import torch

        mem_before = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024**3)
            result["gpu_memory_before"] = f"{mem_before:.2f} GB"

        # Force garbage collection
        collected = gc.collect()
        result["actions"].append(f"Garbage collection: {collected} objects collected")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result["actions"].append("CUDA cache cleared")

            mem_after = torch.cuda.memory_allocated() / (1024**3)
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


@router.get("/settings")
async def get_settings():
    """Get system settings including optional dependencies status."""
    from model_garden.utils.optional_deps import is_unsloth_installed

    # Clear the lru_cache to get fresh status
    is_unsloth_installed.cache_clear()

    # Get Unsloth version if installed
    unsloth_version = None
    if is_unsloth_installed():
        try:
            import unsloth

            unsloth_version = getattr(unsloth, "__version__", "unknown")
        except Exception:
            unsloth_version = "unknown"

    # Get Python and uv info
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Check if running as systemd service
    is_systemd_service = False
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "model-garden.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        is_systemd_service = result.returncode == 0
    except Exception:
        pass

    # Check if passwordless sudo is available for restart
    can_restart_service = False
    if is_systemd_service:
        try:
            result = subprocess.run(
                ["sudo", "-n", "systemctl", "restart", "model-garden.service", "--dry-run"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # If no password prompt and command exists, we can restart
            can_restart_service = result.returncode == 0 or "dry-run" in result.stderr.lower()
        except Exception:
            # Try simpler check - just see if sudo -n works for systemctl
            try:
                result = subprocess.run(
                    ["sudo", "-n", "true"],
                    capture_output=True,
                    timeout=5,
                )
                can_restart_service = result.returncode == 0
            except Exception:
                pass

    return {
        "success": True,
        "data": {
            "optional_dependencies": {
                "unsloth": {
                    "installed": is_unsloth_installed(),
                    "version": unsloth_version,
                    "description": "Optimized training backend (2x faster, 60% less memory)",
                },
            },
            "environment": {
                "python_version": python_version,
                "project_root": str(Path(__file__).parent.parent.parent.parent),
            },
            "service": {
                "is_systemd_service": is_systemd_service,
                "can_restart_service": can_restart_service,
            },
            "package_operation": _package_operation_status,
        },
    }


def _run_package_command(command: list[str], operation: str):
    """Run a package management command in the background."""
    global _package_operation_status

    _package_operation_status = {
        "in_progress": True,
        "operation": operation,
        "output": [],
        "success": None,
        "error": None,
    }

    try:
        # Get project root
        project_root = Path(__file__).parent.parent.parent.parent

        # Run the command
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Collect output
        output_lines = []
        if process.stdout:
            for line in process.stdout:
                output_lines.append(line.rstrip())
                # Keep last 100 lines
                if len(output_lines) > 100:
                    output_lines.pop(0)

        process.wait()

        _package_operation_status["output"] = output_lines
        _package_operation_status["success"] = process.returncode == 0
        if process.returncode != 0:
            _package_operation_status["error"] = f"Command exited with code {process.returncode}"

    except Exception as e:
        _package_operation_status["success"] = False
        _package_operation_status["error"] = str(e)
    finally:
        _package_operation_status["in_progress"] = False

    # Clear the unsloth cache so next check gets fresh status
    from model_garden.utils.optional_deps import is_unsloth_installed

    is_unsloth_installed.cache_clear()


@router.post("/unsloth/install")
async def install_unsloth(background_tasks: BackgroundTasks):
    """Install the Unsloth package."""
    from model_garden.utils.optional_deps import is_unsloth_installed

    # Clear cache to get fresh status
    is_unsloth_installed.cache_clear()

    if is_unsloth_installed():
        return {
            "success": False,
            "message": "Unsloth is already installed",
        }

    if _package_operation_status["in_progress"]:
        return {
            "success": False,
            "message": f"Another package operation is in progress: {_package_operation_status['operation']}",
        }

    # Use uv pip install to add unsloth without affecting other dependencies
    command = ["uv", "pip", "install", "unsloth"]

    background_tasks.add_task(_run_package_command, command, "install_unsloth")

    return {
        "success": True,
        "message": "Unsloth installation started. Check /api/v1/system/settings for progress.",
        "data": {
            "operation": "install_unsloth",
            "command": " ".join(command),
        },
    }


@router.post("/unsloth/uninstall")
async def uninstall_unsloth(background_tasks: BackgroundTasks):
    """Uninstall the Unsloth package."""
    from model_garden.utils.optional_deps import is_unsloth_installed

    # Clear cache to get fresh status
    is_unsloth_installed.cache_clear()

    if not is_unsloth_installed():
        return {
            "success": False,
            "message": "Unsloth is not installed",
        }

    if _package_operation_status["in_progress"]:
        return {
            "success": False,
            "message": f"Another package operation is in progress: {_package_operation_status['operation']}",
        }

    # Use uv pip uninstall
    command = ["uv", "pip", "uninstall", "unsloth", "-y"]

    background_tasks.add_task(_run_package_command, command, "uninstall_unsloth")

    return {
        "success": True,
        "message": "Unsloth uninstallation started. Check /api/v1/system/settings for progress.",
        "data": {
            "operation": "uninstall_unsloth",
            "command": " ".join(command),
        },
    }


@router.get("/unsloth/status")
async def get_unsloth_operation_status():
    """Get the status of the current or last package operation."""
    return {
        "success": True,
        "data": _package_operation_status,
    }


@router.post("/restart")
async def restart_service():
    """Restart the Model Garden service.

    This endpoint requires:
    1. The service to be running under systemd as 'model-garden.service'
    2. Passwordless sudo configured for the restart command

    To enable passwordless sudo, add to /etc/sudoers.d/model-garden:
        <username> ALL=(root) NOPASSWD: /bin/systemctl restart model-garden.service
    """
    # Check if running as systemd service
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "model-garden.service"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail="Model Garden is not running as a systemd service. Please restart manually.",
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Timeout checking service status",
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="systemctl not found. Service restart requires systemd.",
        )

    # Try to restart with passwordless sudo
    try:
        # Use -n flag to prevent password prompt
        result = subprocess.run(
            ["sudo", "-n", "systemctl", "restart", "model-garden.service"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            if "password is required" in result.stderr.lower() or "sudo:" in result.stderr.lower():
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Passwordless sudo not configured for service restart. "
                        "Add to /etc/sudoers.d/model-garden:\n"
                        "  <username> ALL=(root) NOPASSWD: /bin/systemctl restart model-garden.service"
                    ),
                )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to restart service: {result.stderr}",
            )

        # If we get here, restart was initiated successfully
        # The response may not reach the client since the service is restarting
        return {
            "success": True,
            "message": "Service restart initiated. The connection will be lost momentarily.",
        }

    except subprocess.TimeoutExpired:
        # Timeout might mean restart is happening
        return {
            "success": True,
            "message": "Service restart initiated (response timed out, which is expected).",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error restarting service: {str(e)}",
        )
