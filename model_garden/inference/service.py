"""vLLM-powered inference service for Model Garden.

This module provides a simplified inference service that uses vLLM's OpenAI-compatible
serving layer internally. This approach:
- Delegates chat/vision/multimodal handling to vLLM's battle-tested code
- Reduces complexity from 1300+ lines to ~300 lines
- Automatically supports new model types as vLLM adds them
- Keeps our LoRA adapter detection and vision model merging logic
"""

import asyncio
import gc
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from model_garden.utils.console import console

FALLBACK_CHAT_TEMPLATE = """
{% set system_message = 'You are a helpful assistant.' %}
{% for message in messages %}
{% if message['role'] == 'system' %}
{% set system_message = message['content'] %}
{% endif %}
{% endfor %}
{% if bos_token is defined and bos_token %}{{ bos_token }} {% endif %}System: {{ system_message.strip() }}
{% for message in messages %}
{% if message['role'] != 'system' %}
{{ message['role'].title() }}: {{ message['content'] | default('') }}
{% endif %}
{% endfor %}
Assistant:
"""

from .utils import (
    calculate_gpu_memory_utilization,
    detect_quantization_method,
    get_base_model_from_adapter,
    is_lora_adapter,
    is_vision_model,
)


class InferenceService:
    """Manages model inference using vLLM's OpenAI-compatible serving layer.

    This service wraps vLLM's AsyncLLM engine and exposes it through
    OpenAIServingChat for standardized chat completions handling.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.0,
        max_model_len: int | None = None,
        max_num_seqs: int = 16,
        enforce_eager: bool = False,
        limit_mm_per_prompt: dict[str, int] | None = None,
        dtype: str = "auto",
        quantization: str | None = "auto",
        trust_remote_code: bool = False,
        enable_lora: bool = True,
        max_loras: int = 1,
        max_lora_rank: int = 64,
    ):
        """Initialize the inference service.

        Args:
            model_path: Path to the model or HuggingFace model ID (can be LoRA adapter)
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0, 0 = auto)
            max_model_len: Maximum sequence length
            max_num_seqs: Maximum number of concurrent sequences
            enforce_eager: Disable CUDA graphs (saves ~2GB memory but slower)
            limit_mm_per_prompt: Limit multimodal inputs per prompt
            dtype: Data type (auto, float16, bfloat16, float32)
            quantization: Quantization method (auto, awq, gptq, etc.)
            trust_remote_code: Whether to trust remote code
            enable_lora: Enable LoRA adapter support
            max_loras: Maximum number of LoRA adapters
            max_lora_rank: Maximum LoRA rank
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.dtype = dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.enable_lora = enable_lora
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.chat_template: str | None = None

        # Engine state
        self.engine = None
        self.is_loaded = False

        # vLLM OpenAI serving layer
        self.openai_serving_chat = None
        self.openai_serving_models = None

        # LoRA adapter tracking
        self.is_adapter = False
        self.base_model_path: str | None = None
        self.adapter_path: str | None = None

        # Vision model tracking
        self.is_vision_lora_adapter = False
        self.merged_vision_model_path: str | None = None
        self.original_base_model: str | None = None

    async def load_model(self) -> None:
        """Load the model into vLLM engine with OpenAI serving layer."""
        if self.is_loaded:
            console.print("[yellow]Model already loaded[/yellow]")
            return

        console.print(f"[cyan]Loading model: {self.model_path}[/cyan]")

        # Handle LoRA adapters
        await self._handle_lora_adapter()

        # Pre-load GPU cleanup
        self._cleanup_gpu()

        try:
            from vllm import AsyncEngineArgs
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
            from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
            from vllm.v1.engine.async_llm import AsyncLLM

            # Build engine args
            engine_args_dict = self._build_engine_args()
            override_template = self._should_override_chat_template(engine_args_dict)
            engine_args = AsyncEngineArgs(**engine_args_dict)

            # Create async engine
            console.print("[cyan]🚀 Starting vLLM engine...[/cyan]")
            self.engine = AsyncLLM.from_engine_args(engine_args)

            # Wait for engine to be ready
            await self._wait_for_engine_ready()

            # Create OpenAI serving layer on top
            console.print("[cyan]🔧 Setting up OpenAI-compatible serving layer...[/cyan]")

            model_name = Path(self.model_path).name
            chat_template = None
            if override_template:
                if self.chat_template is None:
                    self.chat_template = self._resolve_chat_template()
                chat_template = self.chat_template
            else:
                console.print(
                    "[yellow]Skipping custom chat template (model provides native formatting)[/yellow]"
                )
            base_model_paths = [
                BaseModelPath(name=model_name, model_path=self.base_model_path or self.model_path)
            ]

            self.openai_serving_models = OpenAIServingModels(
                engine_client=self.engine,
                base_model_paths=base_model_paths,
                lora_modules=None,  # We handle LoRA separately
            )

            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                models=self.openai_serving_models,
                response_role="assistant",
                request_logger=None,
                chat_template=chat_template,
                chat_template_content_format="auto",
            )

            self.is_loaded = True
            console.print("[green]✓[/green] Model loaded successfully with OpenAI-compatible API")

        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            import traceback

            console.print(traceback.format_exc())
            raise

    async def _wait_for_engine_ready(self, timeout: float = 120.0) -> None:
        """Wait for the vLLM engine to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to get model config - this will work when engine is ready
                if self.engine and hasattr(self.engine, "model_config"):
                    console.print("[green]✓[/green] Engine ready")
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)

        # If we get here, engine should be ready anyway
        console.print("[yellow]⚠️  Engine startup check timed out, proceeding anyway[/yellow]")

    async def _handle_lora_adapter(self) -> None:
        """Handle LoRA adapter detection and merging for vision models."""
        if not is_lora_adapter(self.model_path):
            self.base_model_path = self.model_path
            return

        self.is_adapter = True
        self.adapter_path = self.model_path

        # Get base model from adapter config
        base_model = get_base_model_from_adapter(self.model_path)
        if not base_model:
            raise ValueError(
                f"Could not determine base model for adapter {self.model_path}. "
                "Please ensure adapter_config.json contains 'base_model_name_or_path'."
            )

        self.base_model_path = base_model

        # Check if this is a vision model adapter
        adapter_is_vision = is_vision_model(self.adapter_path)
        base_is_vision = is_vision_model(base_model)

        if adapter_is_vision or base_is_vision:
            console.print("[yellow]⚠️  Detected vision-language model adapter[/yellow]")
            console.print(
                "[yellow]   Merging adapter with base model (vLLM doesn't support vision LoRAs)[/yellow]"
            )

            self.is_vision_lora_adapter = True
            self.original_base_model = base_model

            # Create temp directory for merged model
            hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
            temp_base = Path(hf_home) / "temp_merges"
            temp_base.mkdir(parents=True, exist_ok=True)
            temp_dir = temp_base / f"model-garden-merged-{int(time.time())}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.merged_vision_model_path = str(temp_dir)

            # Merge the adapter (backend-agnostic, uses Unsloth save if available)
            from model_garden.training.merge import merge_vision_lora_adapter

            merged_path = merge_vision_lora_adapter(
                adapter_path=self.adapter_path,
                output_dir=self.merged_vision_model_path,
                base_model=base_model,
                load_in_4bit=True,
            )

            # Verify merge
            if not (Path(merged_path) / "config.json").exists():
                raise FileNotFoundError(f"Merge failed - config.json not found in {merged_path}")

            self.base_model_path = merged_path
            self.enable_lora = False
            console.print("[green]✓[/green] Vision adapter merged successfully")
        else:
            # Text model adapter - can use vLLM's LoRA support
            console.print(f"[cyan]📦 Loading base model: {base_model}[/cyan]")
            console.print(f"[cyan]🔧 Will apply LoRA adapter: {self.adapter_path}[/cyan]")
            self.enable_lora = True

    def _build_engine_args(self) -> dict[str, Any]:
        """Build the engine arguments dictionary."""
        # Auto-calculate GPU memory if needed
        gpu_memory_utilization = self.gpu_memory_utilization
        if gpu_memory_utilization == 0.0:
            gpu_memory_utilization = calculate_gpu_memory_utilization(
                model_path=self.model_path,
                max_model_len=self.max_model_len,
                tensor_parallel_size=self.tensor_parallel_size,
            )
            console.print(f"[green]✓[/green] Auto GPU memory utilization: {gpu_memory_utilization}")

        # Detect quantization
        quantization = self.quantization
        load_format = "auto"
        is_hf_model = "/" in self.model_path and not Path(self.model_path).exists()

        if quantization == "auto" or quantization is None:
            if is_hf_model:
                quantization = None
            else:
                quantization = detect_quantization_method(self.base_model_path or self.model_path)
                if quantization is None:
                    load_format = "safetensors"

        # Validate quantization value
        valid_quantization = [
            "awq",
            "gptq",
            "squeezellm",
            "fp8",
            "bitsandbytes",
            "compressed-tensors",
            "marlin",
            "ggml",
        ]
        quantization_param = quantization if quantization in valid_quantization else None

        # Build args dict
        engine_args = {
            "model": self.base_model_path or self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "dtype": self.dtype
            if self.dtype in ["auto", "half", "float16", "bfloat16", "float", "float32"]
            else "auto",
            "quantization": quantization_param,
            "load_format": load_format,
            "trust_remote_code": self.trust_remote_code,
            "enforce_eager": self.enforce_eager,
            "disable_log_stats": False,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        }

        # Add multimodal limits
        if self.limit_mm_per_prompt:
            engine_args["limit_mm_per_prompt"] = self.limit_mm_per_prompt

        # Handle Mistral models with native tokenizer
        model_path_lower = self.model_path.lower()
        if any(x in model_path_lower for x in ["mistral", "ministral", "pixtral"]):
            console.print("[cyan]🔧 Detected Mistral model - using native tokenizer mode[/cyan]")
            engine_args["tokenizer_mode"] = "mistral"
            engine_args["config_format"] = "mistral"
            engine_args["load_format"] = "mistral"

        # LoRA support for text models
        if self.enable_lora and self.is_adapter and not self.is_vision_lora_adapter:
            engine_args["enable_lora"] = True
            engine_args["max_loras"] = self.max_loras
            engine_args["max_lora_rank"] = self.max_lora_rank

        return engine_args

    def _cleanup_gpu(self) -> None:
        """Clean up GPU memory before loading."""
        try:
            import torch

            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            console.print("[cyan]✓ Pre-load cleanup completed[/cyan]")
        except Exception as e:
            console.print(f"[yellow]⚠️  Pre-load cleanup warning: {e}[/yellow]")

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if not self.is_loaded:
            console.print("[yellow]No model loaded[/yellow]")
            return

        console.print("[cyan]Unloading model...[/cyan]")

        # Clear serving layer
        self.openai_serving_chat = None
        self.openai_serving_models = None

        # Delete engine
        if self.engine:
            del self.engine
            self.engine = None

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        # Clean up temp merged model
        if self.merged_vision_model_path and Path(self.merged_vision_model_path).exists():
            import shutil

            shutil.rmtree(self.merged_vision_model_path, ignore_errors=True)
            self.merged_vision_model_path = None

        self.is_loaded = False
        console.print("[green]✓[/green] Model unloaded successfully")

    async def close(self) -> None:
        """Close the inference service."""
        await self.unload_model()

    async def chat_completions(
        self,
        request,  # ChatCompletionRequest from vLLM
        raw_request=None,
    ) -> AsyncGenerator[str, None] | dict | Any:
        """
        Process a chat completion request using vLLM's OpenAI-compatible layer.

        This method directly delegates to vLLM's OpenAIServingChat, which handles:
        - Chat template application
        - Multimodal (vision) inputs
        - Structured outputs
        - Streaming
        - All model-specific quirks (Mistral, Qwen, Llama, etc.)

        Args:
            request: ChatCompletionRequest (from vllm.entrypoints.openai.protocol)
            raw_request: Optional FastAPI Request object for SSE streaming

        Returns:
            ChatCompletionResponse, AsyncGenerator for streaming, or ErrorResponse
        """
        if not self.is_loaded or not self.openai_serving_chat:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return await self.openai_serving_chat.create_chat_completion(request, raw_request)

    def _resolve_chat_template(self) -> str | None:
        """Attempt to load a chat template from tokenizer with a safe fallback."""
        model_id = self.base_model_path or self.model_path
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=self.trust_remote_code,
            )
            template = getattr(tokenizer, "chat_template", None)
            if template and template.strip():
                console.print("[green]✓[/green] Chat template loaded from tokenizer")
                return template
        except Exception as exc:
            console.print(f"[yellow]⚠️  Could not load chat template from tokenizer: {exc}[/yellow]")

        console.print("[yellow]⚠️  Falling back to generic chat template[/yellow]")
        return FALLBACK_CHAT_TEMPLATE

    def _normalize_message_content(self, content: Any) -> list[dict[str, Any]]:
        """Convert message content to OpenAI v1 list-based format."""
        normalized: list[dict[str, Any]] = []

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    normalized.append({"type": "text", "text": str(item)})
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text")
                    if text is not None:
                        normalized.append({"type": "text", "text": str(text)})
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    else:
                        url = image_url
                    if url:
                        normalized.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    # Preserve unknown multimodal parts as text to avoid validation errors
                    normalized.append({"type": "text", "text": item.get("text", "")})
        elif isinstance(content, dict):
            item_type = content.get("type")
            if item_type == "image_url":
                image_url = content.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                else:
                    url = image_url
                if url:
                    normalized.append({"type": "image_url", "image_url": {"url": url}})
            else:
                text_value = content.get("text") if item_type == "text" else str(content)
                normalized.append({"type": "text", "text": text_value})
        else:
            text_value = "" if content is None else str(content)
            normalized.append({"type": "text", "text": text_value})

        if not normalized:
            normalized.append({"type": "text", "text": ""})

        return normalized

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize chat messages for vLLM's ChatCompletionRequest schema."""
        normalized_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role") or "user"
            normalized_msg: dict[str, Any] = {"role": role}

            name = msg.get("name")
            if isinstance(name, str) and name.strip():
                normalized_msg["name"] = name.strip()

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    normalized_msg["tool_call_id"] = tool_call_id
                tool_content = msg.get("content")
                normalized_msg["content"] = "" if tool_content is None else str(tool_content)
            else:
                normalized_msg["content"] = self._normalize_message_content(msg.get("content"))

                if role == "assistant":
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        normalized_msg["tool_calls"] = tool_calls
                function_call = msg.get("function_call")
                if function_call:
                    normalized_msg["function_call"] = function_call

            normalized_messages.append(normalized_msg)

        return normalized_messages

    def _should_override_chat_template(self, engine_args: dict[str, Any]) -> bool:
        """Determine if we should pass a custom chat template to vLLM."""

        def _marker(value: Any) -> str:
            return str(value).lower() if value else ""

        markers = {
            _marker(engine_args.get("tokenizer_mode")),
            _marker(engine_args.get("config_format")),
            _marker(engine_args.get("load_format")),
        }

        # Mistral-native tokenizers reject chat_template overrides. Let vLLM handle it.
        if any(marker.startswith("mistral") for marker in markers):
            return False

        return True

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        images: list[str] | None = None,
        structured_outputs: dict | None = None,
    ) -> dict | AsyncGenerator[str, None]:
        """
        Generate text from a prompt (legacy API, wraps chat_completions).

        For new code, prefer using chat_completions() directly with
        ChatCompletionRequest for full OpenAI API compatibility.
        """
        from vllm.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
            ErrorResponse,
        )

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Handle images by converting to OpenAI multimodal format
        if images:
            content = []
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

        # Build request
        request_dict: dict[str, Any] = {
            "model": Path(self.model_path).name,
            "messages": messages,
            "max_tokens": max_tokens or (16384 if structured_outputs else 512),
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stop:
            request_dict["stop"] = stop
        if frequency_penalty is not None:
            request_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_dict["presence_penalty"] = presence_penalty

        # Handle structured outputs
        if structured_outputs:
            request_dict["response_format"] = structured_outputs

        request = ChatCompletionRequest(**request_dict)
        result = await self.chat_completions(request)

        if stream:
            return result

        if isinstance(result, AsyncGenerator):
            raise RuntimeError("Streaming generator returned for non-streaming request")

        if isinstance(result, ErrorResponse):
            return result.model_dump()

        if isinstance(result, ChatCompletionResponse):
            text_content = ""
            if result.choices:
                first_choice = result.choices[0]
                if first_choice.message and first_choice.message.content:
                    text_content = first_choice.message.content

            usage = result.usage
            return {
                "text": text_content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
            }

        if isinstance(result, dict):
            return result

        if hasattr(result, "model_dump"):
            return result.model_dump()

        return {
            "text": str(getattr(result, "text", "")),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def generate_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        stop: list[str] | None = None,
        stream: bool = False,
        image: str | dict[str, Any] | None = None,
        structured_outputs: dict | None = None,
        **kwargs,
    ) -> dict | AsyncGenerator[dict, None]:
        """
        Generate a chat completion (legacy API, wraps chat_completions).

        For new code, prefer using chat_completions() directly.

        Returns:
            For non-streaming: dict with 'text' and 'usage' keys
            For streaming: AsyncGenerator yielding dicts (OpenAI chunk format)
        """
        # Deep copy messages to avoid mutating input
        import copy
        import json

        from vllm.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
            ErrorResponse,
        )

        messages = self._normalize_messages(copy.deepcopy(messages))

        # Handle image by injecting into last user message
        if image:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    if not isinstance(content, list):
                        content = self._normalize_message_content(content)

                    image_url: str = ""
                    mime_type = "image/jpeg"

                    if isinstance(image, dict):
                        url_value = image.get("url")
                        if isinstance(url_value, str) and url_value:
                            image_url = url_value
                        else:
                            base64_data = image.get("data") or image.get("base64")
                            if isinstance(base64_data, str):
                                mime_type = image.get("mime", mime_type)
                                image_url = f"data:{mime_type};base64,{base64_data}"
                    elif isinstance(image, str):
                        if image.startswith("data:"):
                            image_url = image
                        elif not image.startswith(("http://", "https://", "file://")):
                            image_url = f"data:{mime_type};base64,{image}"
                        else:
                            image_url = image

                    msg["content"] = [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        *content,
                    ]
                    break

        request_dict: dict[str, Any] = {
            "model": Path(self.model_path).name,
            "messages": messages,
            "max_tokens": max_tokens or (16384 if structured_outputs else 512),
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

        if stop:
            request_dict["stop"] = stop
        if frequency_penalty is not None:
            request_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_dict["presence_penalty"] = presence_penalty
        if structured_outputs:
            request_dict["response_format"] = structured_outputs

        request = ChatCompletionRequest(**request_dict)
        result = await self.chat_completions(request)

        # Handle streaming - convert SSE strings to dicts
        if stream:
            if isinstance(result, ErrorResponse):

                async def error_stream() -> AsyncGenerator[dict[str, Any], None]:
                    yield result.model_dump()

                return error_stream()

            if not isinstance(result, AsyncGenerator):

                async def fallback_stream() -> AsyncGenerator[dict[str, Any], None]:
                    yield {"error": "Streaming response unavailable"}

                return fallback_stream()

            stream_result = result

            async def parse_sse_stream() -> AsyncGenerator[dict[str, Any], None]:
                """Parse vLLM's SSE stream into dict chunks."""
                async for sse_line in stream_result:
                    # vLLM yields "data: {...}\n\n" strings
                    if sse_line.startswith("data: "):
                        data = sse_line[6:].strip()
                        if data and data != "[DONE]":
                            try:
                                yield json.loads(data)
                            except json.JSONDecodeError:
                                pass

            return parse_sse_stream()

        if isinstance(result, AsyncGenerator):
            raise RuntimeError("Streaming generator returned for non-streaming request")

        if isinstance(result, ErrorResponse):
            return result.model_dump()

        # Non-streaming - convert response to legacy format
        if isinstance(result, ChatCompletionResponse) and result.choices:
            first_choice = result.choices[0]
            text_content = ""
            if first_choice.message and first_choice.message.content:
                text_content = first_choice.message.content

            usage = result.usage
            return {
                "text": text_content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
            }

        if isinstance(result, ChatCompletionResponse):
            # No choices returned
            usage = result.usage
            return {
                "text": "",
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                },
            }

        if isinstance(result, dict):
            return result

        if hasattr(result, "model_dump"):
            return result.model_dump()

        return {"error": "Unexpected response type"}

    # Alias for backwards compatibility with routes
    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        image: str | None = None,
        structured_outputs: dict | None = None,
        **kwargs,
    ) -> dict | AsyncGenerator[dict, None]:
        """
        OpenAI-compatible chat completion (legacy API).

        This is an alias for generate_chat() with OpenAI-style response format.
        """
        return await self.generate_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            image=image,
            structured_outputs=structured_outputs,
            **kwargs,
        )

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        model_display_path = self.model_path
        if Path(self.model_path).is_absolute():
            model_display_path = Path(self.model_path).name

        info = {
            "model_path": model_display_path,
            "is_loaded": self.is_loaded,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "quantization": self.quantization,
        }

        if self.is_adapter:
            info["is_lora_adapter"] = True
            info["base_model"] = self.base_model_path
            info["adapter_path"] = self.adapter_path
            info["lora_enabled"] = self.enable_lora

            if self.is_vision_lora_adapter:
                info["is_vision_adapter"] = True
                info["merged_automatically"] = True
                info["note"] = "Vision LoRA was automatically merged"

        return info


# Global service instance management
_inference_service: InferenceService | None = None


def get_inference_service() -> InferenceService | None:
    """Get the global inference service instance."""
    return _inference_service


def set_inference_service(service: InferenceService | None) -> None:
    """Set the global inference service instance."""
    global _inference_service
    _inference_service = service
