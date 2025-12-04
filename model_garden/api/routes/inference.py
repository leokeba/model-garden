# Inference routes
"""
Routes for inference and model serving:
- POST /api/v1/inference/load - Load a model
- POST /api/v1/inference/unload - Unload the current model
- GET /api/v1/inference/status - Get inference status
- GET /api/v1/inference/queue - Get model loading queue
- POST /api/v1/inference/generate - Generate text
- POST /api/v1/chat/completions - Chat completions (OpenAI-compatible)
- POST /v1/chat/completions - Chat completions (OpenAI standard path)
"""

import json
import re
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import cast

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models import APIResponse

router = APIRouter(tags=["inference"])


# Request/Response Models
class InferenceRequest(BaseModel):
    """Request for text generation."""

    prompt: str
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | None = None
    stream: bool = False


class ChatMessage(BaseModel):
    """Chat message with support for multimodal content."""

    role: str
    content: str | list[dict] | dict
    name: str | None = None

    model_config = {"extra": "allow"}


class ResponseFormat(BaseModel):
    """OpenAI-compatible response format for structured outputs."""

    type: str
    json_schema: dict | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = 0.7
    top_p: float | None = 0.95
    top_k: int | None = -1
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    stream: bool | None = False
    stop: str | list[str] | None = None
    n: int | None = 1
    best_of: int | None = None
    logprobs: int | None = None
    echo: bool | None = False
    user: str | None = None
    response_format: ResponseFormat | None = None

    model_config = {"extra": "allow"}


class LoadModelRequest(BaseModel):
    """Request to load a model for inference."""

    model_path: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.0
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool | None = None
    limit_mm_per_prompt: dict[str, int] | None = None
    dtype: str = "auto"
    quantization: str | None = None


def resolve_path(path: str) -> str:
    """Resolve a path to an absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p.resolve())


def resolve_model_path(path: str) -> str:
    """Resolve a model path, handling simple names and HuggingFace IDs."""
    # Check if it's a HuggingFace ID (contains /)
    if "/" in path and not path.startswith("./") and not path.startswith("/"):
        # Likely a HuggingFace model ID
        return path

    # Check if it's a simple name
    if "/" not in path and not Path(path).exists():
        models_path = Path("./models") / path
        if models_path.exists():
            return str(models_path.resolve())

    return resolve_path(path)


def convert_response_format_to_structured_outputs(
    response_format: ResponseFormat | None,
) -> dict | None:
    """Convert OpenAI response_format to vLLM structured_outputs parameters."""
    if not response_format:
        return None

    if response_format.type == "text":
        return None

    elif response_format.type == "json_object":
        return {"json": {"type": "object", "properties": {}, "additionalProperties": True}}

    elif response_format.type == "json_schema":
        if not response_format.json_schema:
            raise ValueError("json_schema must be provided when type is 'json_schema'")

        schema = response_format.json_schema
        if isinstance(schema, dict):
            if "schema" in schema:
                actual_schema = schema["schema"]
            else:
                actual_schema = schema
            return {"json": actual_schema}

        return {"json": schema}

    return None


def ensure_openai_chat_response_format(response: dict, model_name: str | None) -> dict:
    """Ensure chat responses include OpenAI-compatible choice structure."""
    if not isinstance(response, dict):
        return response

    if "choices" in response and isinstance(response["choices"], list):
        return response

    text = response.get("text")
    if text is None:
        return response

    usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
    timestamp = int(time.time())

    converted = {
        "id": response.get("id") or f"chatcmpl-local-{timestamp}",
        "object": "chat.completion",
        "created": timestamp,
        "model": model_name or "unknown-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": response.get("finish_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }

    if "x_carbon_trace" in response:
        converted["x_carbon_trace"] = response["x_carbon_trace"]

    return converted


@router.post("/api/v1/inference/load", response_model=APIResponse)
async def load_inference_model(request: LoadModelRequest, background_tasks: BackgroundTasks):
    """Load a model for inference."""
    from model_garden.inference import get_inference_service
    from model_garden.model_registry import get_model
    from model_garden.queue import JobType, get_job_queue

    queue = get_job_queue()

    # Resolve model path
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
    max_num_seqs = request.max_num_seqs
    enforce_eager = request.enforce_eager
    limit_mm_per_prompt = request.limit_mm_per_prompt
    dtype = request.dtype
    gpu_memory_utilization = request.gpu_memory_utilization
    quantization = request.quantization
    tensor_parallel_size = request.tensor_parallel_size

    if model_info:
        if max_model_len is None:
            max_model_len = model_info.inference_defaults.max_model_len
        if max_num_seqs is None:
            max_num_seqs = model_info.inference_defaults.max_num_seqs
        if enforce_eager is None:
            enforce_eager = model_info.inference_defaults.enforce_eager
        if limit_mm_per_prompt is None and model_info.inference_defaults.limit_mm_per_prompt:
            limit_mm_per_prompt = model_info.inference_defaults.limit_mm_per_prompt
        if dtype == "auto":
            dtype = model_info.inference_defaults.dtype
        if gpu_memory_utilization == 0.0:
            gpu_memory_utilization = model_info.inference_defaults.gpu_memory_utilization
        if quantization is None and model_info.inference_defaults.quantization:
            quantization = model_info.inference_defaults.quantization
        if tensor_parallel_size == 1 and model_info.inference_defaults.tensor_parallel_size > 1:
            tensor_parallel_size = model_info.inference_defaults.tensor_parallel_size

    # Apply final defaults
    if max_num_seqs is None:
        max_num_seqs = 16
    if enforce_eager is None:
        enforce_eager = False

    # Check if a model is already loaded
    current_service = get_inference_service()
    if current_service and current_service.is_loaded:
        return APIResponse(
            success=False,
            message=f"Model already loaded: {current_service.model_path}. Unload it first.",
        )

    # Check for loading jobs
    loading_job = await queue.get_running_job(JobType.MODEL_LOADING)
    if loading_job:
        return APIResponse(
            success=False, message="A model is already loading. Please wait or cancel it first."
        )

    # Start loading
    job_id = f"model-loading-{int(datetime.now().timestamp())}"

    await queue.add_job(
        job_id=job_id,
        job_type=JobType.MODEL_LOADING,
        job_config={
            "model_path": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "enforce_eager": enforce_eager,
            "limit_mm_per_prompt": limit_mm_per_prompt,
            "dtype": dtype,
            "quantization": quantization,
        },
        priority=0,
    )

    # Import and schedule the loading task
    from model_garden.api import run_model_loading

    background_tasks.add_task(
        run_model_loading,
        job_id,
        model_path,
        tensor_parallel_size,
        gpu_memory_utilization,
        max_model_len,
        max_num_seqs,
        enforce_eager,
        limit_mm_per_prompt,
        dtype,
        quantization,
    )

    return APIResponse(
        success=True,
        data={"job_id": job_id, "status": "loading"},
        message=f"Model loading started: {request.model_path}",
    )


@router.post("/api/v1/inference/unload", response_model=APIResponse)
async def unload_inference_model():
    """Unload the currently loaded model."""
    from model_garden.carbon import stop_inference_tracker
    from model_garden.inference import get_inference_service, set_inference_service

    service = get_inference_service()
    if not service or not service.is_loaded:
        return APIResponse(success=False, message="No model currently loaded")

    try:
        # Stop carbon tracking
        try:
            emissions_data = stop_inference_tracker()
            if emissions_data:
                print(
                    f"✅ Inference emissions saved: {emissions_data['emissions_kg_co2']:.6f} kg CO2"
                )
        except Exception as e:
            print(f"⚠️  Failed to stop inference carbon tracking: {e}")

        await service.unload_model()
        set_inference_service(None)

        return APIResponse(success=True, message="Model unloaded successfully")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}",
        ) from None


@router.get("/api/v1/inference/status")
async def get_inference_status():
    """Get inference service status."""
    from model_garden.inference import get_inference_service
    from model_garden.queue import JobStatus, JobType, get_job_queue

    queue = get_job_queue()

    loading_job = await queue.get_running_job(JobType.MODEL_LOADING)
    queued_jobs = await queue.list_jobs(
        status=JobStatus.QUEUED, job_type=JobType.MODEL_LOADING.value
    )

    service = get_inference_service()

    response = {
        "loaded": service is not None and service.is_loaded,
        "model_info": service.get_model_info() if service and service.is_loaded else None,
    }

    if loading_job:
        response["loading"] = {
            "job_id": loading_job["job_id"],
            "model_path": loading_job["job_config"].get("model_path"),
            "started_at": loading_job["started_at"],
            "status_message": loading_job["status_message"],
        }

    if queued_jobs:
        response["queue"] = {
            "count": len(queued_jobs),
            "jobs": [
                {
                    "job_id": job["job_id"],
                    "model_path": job["job_config"].get("model_path"),
                    "queued_at": job["queued_at"],
                    "position": i + 1,
                }
                for i, job in enumerate(queued_jobs)
            ],
        }

    return response


@router.get("/api/v1/inference/queue")
async def get_model_loading_queue():
    """Get model loading queue status."""
    from model_garden.queue import JobStatus, JobType, get_job_queue

    queue = get_job_queue()
    all_jobs = await queue.list_jobs(job_type=JobType.MODEL_LOADING.value)

    running = [j for j in all_jobs if j["status"] == JobStatus.RUNNING]
    queued = [j for j in all_jobs if j["status"] == JobStatus.QUEUED]
    completed = [j for j in all_jobs if j["status"] == JobStatus.COMPLETED][:10]
    failed = [j for j in all_jobs if j["status"] == JobStatus.FAILED][:10]

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
        },
    }


@router.post("/api/v1/inference/generate")
async def generate_text(request: InferenceRequest):
    """Generate text from a prompt."""
    from model_garden.inference import get_inference_service

    service = get_inference_service()
    if not service or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded. Load a model first using /api/v1/inference/load",
        )

    try:
        if request.stream:

            async def generate_stream():
                stream = cast(
                    AsyncIterator[str],
                    await service.generate(
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        frequency_penalty=request.frequency_penalty,
                        presence_penalty=request.presence_penalty,
                        stop=request.stop,
                        stream=True,
                    ),
                )

                async for chunk in stream:
                    yield f"data: {json.dumps({'text': chunk})}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Generation failed: {str(e)}"
        ) from None


@router.post("/api/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with vision support."""
    import time

    from model_garden.inference import get_inference_service

    print(
        f"📨 Received chat completion request: model={request.model}, messages={len(request.messages)}, stream={request.stream}"
    )

    service = get_inference_service()
    if not service or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model loaded. Load a model first using /api/v1/inference/load",
        )

    try:
        # Process messages and extract multimodal content
        processed_messages = []
        image_data: str | dict | None = None

        for msg in request.messages:
            msg_dict = msg.model_dump()
            content = msg_dict.get("content", "")

            # Handle multimodal content (OpenAI format)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                            else:
                                url = image_url

                            if url.startswith("data:image/"):
                                match = re.match(r"data:(image/[^;]+);base64,(.+)", url)
                                if match:
                                    image_data = {
                                        "data": match.group(2),
                                        "mime": match.group(1),
                                    }
                            else:
                                image_data = url
                    else:
                        text_parts.append(str(part))

                msg_dict["content"] = " ".join(text_parts)
            elif isinstance(content, dict):
                if content.get("type") == "text":
                    msg_dict["content"] = content.get("text", "")
                elif content.get("type") == "image_url":
                    image_url = content.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url

                    if url.startswith("data:image/"):
                        match = re.match(r"data:(image/[^;]+);base64,(.+)", url)
                        if match:
                            image_data = {
                                "data": match.group(2),
                                "mime": match.group(1),
                            }
                    else:
                        image_data = url

            # Check for custom 'image' field
            if "image" in msg_dict and msg_dict["image"]:
                image_data = msg_dict["image"]

            processed_messages.append(msg_dict)

        # Prepare generation parameters
        gen_params = {
            "messages": processed_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream or False,
        }

        if image_data:
            gen_params["image"] = image_data
            print("✅ Added image to generation parameters")

        if request.stop:
            gen_params["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]

        # Add structured output parameters
        if request.response_format:
            structured_outputs = convert_response_format_to_structured_outputs(
                request.response_format
            )
            if structured_outputs:
                gen_params["structured_outputs"] = structured_outputs
                print("✅ Added structured output parameters")

        if request.stream:

            async def generate_stream():
                total_tokens = 0
                stream = cast(AsyncIterator[dict], await service.chat_completion(**gen_params))

                async for chunk in stream:
                    if isinstance(chunk, dict) and "usage" in chunk:
                        total_tokens = chunk["usage"].get("completion_tokens", 0)

                    yield f"data: {json.dumps(chunk)}\n\n"

                # Record in carbon tracker
                try:
                    from model_garden.carbon import get_inference_tracker

                    tracker = get_inference_tracker()
                    if tracker:
                        tracker.record_request(tokens_generated=total_tokens)
                except Exception:
                    pass

                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Track carbon emissions
            carbon_data = None
            before_emissions = None
            request_start_time = None
            session_tracker = None

            try:
                from model_garden.carbon import get_inference_tracker

                session_tracker = get_inference_tracker()
                if session_tracker:
                    before_emissions = session_tracker.get_request_emissions()
                    request_start_time = time.time()
            except Exception:
                pass

            response = cast(dict, await service.chat_completion(**gen_params))
            response = ensure_openai_chat_response_format(response, request.model or service.model_path)

            # Calculate carbon emissions
            try:
                if session_tracker and before_emissions:
                    after_emissions = session_tracker.get_request_emissions()
                    request_duration = time.time() - request_start_time if request_start_time else 0

                    if after_emissions:
                        delta_emissions_kg = after_emissions.get(
                            "emissions_kg_co2", 0.0
                        ) - before_emissions.get("emissions_kg_co2", 0.0)
                        delta_energy_kwh = after_emissions.get(
                            "energy_consumed_kwh", 0.0
                        ) - before_emissions.get("energy_consumed_kwh", 0.0)

                        tokens = 0
                        if isinstance(response, dict) and "usage" in response:
                            tokens = response["usage"].get("completion_tokens", 0)

                        carbon_data = {
                            "emissions_g_co2": delta_emissions_kg * 1000,
                            "energy_consumed_wh": delta_energy_kwh * 1000,
                            "duration_seconds": request_duration,
                            "completion_tokens": tokens,
                            "measured": True,
                        }

                        session_tracker.record_request(tokens_generated=tokens)
            except Exception:
                pass

            if carbon_data and isinstance(response, dict):
                response["x_carbon_trace"] = carbon_data

            return response

    except Exception as e:
        import traceback

        print(f"❌ Chat completion error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}",
        ) from None


# OpenAI-compatible endpoint (standard path)
@router.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint (standard path)."""
    return await chat_completions(request)
