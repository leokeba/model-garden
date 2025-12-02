# System routes
"""
Routes for system management:
- GET /api/v1/system/status - Get system status (CPU, GPU, memory)
- POST /api/v1/system/cleanup - Force GPU memory cleanup
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/system", tags=["system"])


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
