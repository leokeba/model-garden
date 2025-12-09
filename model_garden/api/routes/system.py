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

    # Get transformers version
    transformers_version = None
    try:
        import transformers

        transformers_version = getattr(transformers, "__version__", "unknown")
    except Exception:
        transformers_version = "unknown"

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
    # Use sudo -l to list allowed commands for the user
    can_restart_service = False
    if is_systemd_service:
        try:
            result = subprocess.run(
                ["sudo", "-n", "-l"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Check if the output contains our specific restart command
            # The sudo -l output shows full path: /usr/bin/systemctl restart model-garden.service
            if result.returncode == 0:
                can_restart_service = "systemctl restart model-garden.service" in result.stdout
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
                "transformers_version": transformers_version,
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


def _run_package_commands_sequence(commands: list[list[str]], operation: str):
    """Run multiple package management commands in sequence."""
    global _package_operation_status

    _package_operation_status = {
        "in_progress": True,
        "operation": operation,
        "output": [],
        "success": None,
        "error": None,
    }

    try:
        project_root = Path(__file__).parent.parent.parent.parent
        all_output = []

        for i, command in enumerate(commands):
            all_output.append(f">>> Running: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if process.stdout:
                for line in process.stdout:
                    all_output.append(line.rstrip())
                    # Keep last 100 lines
                    if len(all_output) > 100:
                        all_output.pop(0)

            process.wait()

            if process.returncode != 0:
                _package_operation_status["output"] = all_output
                _package_operation_status["success"] = False
                _package_operation_status["error"] = (
                    f"Command {i + 1} exited with code {process.returncode}"
                )
                return

        _package_operation_status["output"] = all_output
        _package_operation_status["success"] = True

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

    # Use uv pip install to add unsloth (this will bring in compatible transformers version)
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

    # Uninstall unsloth and unsloth-zoo, then upgrade transformers to latest
    # unsloth-zoo is a dependency that also pins transformers to older versions
    # Note: We use `uv pip install` instead of `uv sync` because the lockfile
    # includes unsloth-zoo constraints (from the unsloth optional dependency)
    # which would prevent upgrading transformers to the latest version.
    uninstall_command = ["uv", "pip", "uninstall", "unsloth", "unsloth-zoo"]
    upgrade_command = ["uv", "pip", "install", "--upgrade", "transformers"]

    background_tasks.add_task(
        _run_package_commands_sequence,
        [uninstall_command, upgrade_command],
        "uninstall_unsloth",
    )

    return {
        "success": True,
        "message": "Unsloth uninstallation started. This will also upgrade transformers to the latest version. Check /api/v1/system/settings for progress.",
        "data": {
            "operation": "uninstall_unsloth",
            "commands": [
                " ".join(uninstall_command),
                " ".join(upgrade_command),
            ],
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
        <username> ALL=(root) NOPASSWD: /usr/bin/systemctl restart model-garden.service
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
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=500,
            detail="Timeout checking service status",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=400,
            detail="systemctl not found. Service restart requires systemd.",
        ) from exc

    # Try to restart with passwordless sudo
    try:
        # First, verify sudo access with a quick test (this won't restart anything)
        test_result = subprocess.run(
            ["sudo", "-n", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if (
            test_result.returncode != 0
            or "systemctl restart model-garden.service" not in test_result.stdout
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    "Passwordless sudo not configured for service restart. "
                    "Add to /etc/sudoers.d/model-garden:\n"
                    "  <username> ALL=(root) NOPASSWD: /usr/bin/systemctl restart model-garden.service"
                ),
            )

        # Now run the actual restart in a detached process
        # Use nohup and shell to ensure the command survives the service shutdown
        subprocess.Popen(
            "nohup sudo -n systemctl restart model-garden.service > /dev/null 2>&1 &",
            shell=True,
            start_new_session=True,
        )

        # Return success immediately - the service will restart momentarily
        return {
            "success": True,
            "message": "Service restart initiated. The connection will be lost momentarily.",
        }

    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=500,
            detail="Timeout checking sudo permissions",
        ) from exc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error restarting service: {str(e)}",
        ) from e


@router.get("/gpu/memory")
async def get_gpu_memory_stats():
    """Get detailed GPU memory statistics.

    Returns live GPU memory stats and the memory profile from the last model load.
    """
    from model_garden.inference import get_current_memory_profile, get_live_gpu_stats

    live_stats = get_live_gpu_stats()
    memory_profile = get_current_memory_profile()

    return {
        "success": True,
        "data": {
            "live": live_stats,
            "profile": memory_profile.to_dict() if memory_profile else None,
        },
    }


@router.get("/gpu/memory/profile")
async def get_gpu_memory_profile():
    """Get the GPU memory profile from the last model load.

    Returns detailed breakdown of memory usage by component (weights, KV cache, CUDA graphs, etc.)
    """
    from model_garden.inference import get_current_memory_profile

    profile = get_current_memory_profile()

    if not profile:
        return {
            "success": False,
            "message": "No memory profile available. Load a model first.",
            "data": None,
        }

    return {
        "success": True,
        "data": profile.to_dict(),
    }
