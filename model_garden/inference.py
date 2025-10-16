"""vLLM-powered inference service for Model Garden."""

import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional, Union

from rich.console import Console

console = Console()


class InferenceService:
    """Manages model inference using vLLM."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """Initialize the inference service.

        Args:
            model_path: Path to the model or HuggingFace model ID
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length
            dtype: Data type (auto, float16, bfloat16, float32)
            quantization: Quantization method (awq, gptq, squeezellm, fp8)
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        
        self.engine = None
        self.is_loaded = False

    async def load_model(self) -> None:
        """Load the model into vLLM engine."""
        if self.is_loaded:
            console.print("[yellow]Model already loaded[/yellow]")
            return

        console.print(f"[cyan]Loading model: {self.model_path}[/cyan]")
        
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            
            # Configure engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                quantization=self.quantization,
                trust_remote_code=self.trust_remote_code,
                enforce_eager=False,  # Use CUDA graphs for better performance
                disable_log_stats=False,
            )
            
            # Create async engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.is_loaded = True
            
            console.print("[green]✓[/green] Model loaded successfully")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        if not self.is_loaded:
            console.print("[yellow]No model loaded[/yellow]")
            return

        console.print("[cyan]Unloading model...[/cyan]")
        
        # vLLM doesn't have explicit unload, but we can delete the engine
        if self.engine:
            del self.engine
            self.engine = None
        
        self.is_loaded = False
        console.print("[green]✓[/green] Model unloaded")

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        images: Optional[List[str]] = None,
        structured_outputs: Optional[Dict] = None,
    ) -> Union[Dict, AsyncIterator[str]]:
        """Generate text from a prompt with optional multimodal inputs.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate (None = auto, default 512)
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling probability
            top_k: Top-k sampling (-1 to disable)
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            stop: List of stop sequences
            stream: Whether to stream the response
            images: List of image URLs or file paths (for vision models)
            structured_outputs: Optional structured output parameters (json, regex, choice, grammar, structural_tag)

        Returns:
            Dict with text and usage, or async iterator of text chunks if streaming
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from vllm import SamplingParams
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            if structured_outputs:
                max_tokens = 2048  # Higher default for structured outputs
            else:
                max_tokens = 512  # Standard default
        
        # Create structured outputs params if provided
        structured_outputs_params = None
        if structured_outputs:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                structured_outputs_params = StructuredOutputsParams(**structured_outputs)
            except ImportError:
                console.print("[yellow]Warning: StructuredOutputsParams not available in this vLLM version[/yellow]")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            structured_outputs=structured_outputs_params,
        )
        
        # Prepare inputs (text + optional images)
        inputs = self._prepare_inputs(prompt, images)
        
        # Generate unique request ID
        request_id = f"req-{id(prompt)}-{asyncio.get_event_loop().time()}"
        
        if stream:
            return self._generate_stream(inputs, sampling_params, request_id)
        else:
            return await self._generate_complete(inputs, sampling_params, request_id)

    def _prepare_inputs(self, prompt: str, images: Optional[List[str]] = None):
        """Prepare inputs for generation, handling multimodal data if images are provided."""
        if images is None or len(images) == 0:
            return prompt
        
        try:
            from vllm.inputs import TextPrompt
            import requests
            from io import BytesIO
            from PIL import Image
            import base64
            import tempfile
            
            # Load images from URLs, file paths, or base64 data
            # For vLLM multiprocessing compatibility, we'll convert base64 to temp files
            loaded_images = []
            for img_data in images:
                if img_data.startswith('data:image/'):
                    # It's a data URL with base64 - this shouldn't happen as we extract it in the API
                    # but handle it just in case
                    import re
                    match = re.match(r"data:image/[^;]+;base64,(.+)", img_data)
                    if match:
                        img_data = match.group(1)
                    # Fall through to base64 handling
                
                if img_data.startswith(('http://', 'https://')):
                    # Download the image from URL and save to temp file
                    # vLLM's Qwen2.5-VL processor doesn't handle URL downloading
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        response = requests.get(img_data, timeout=10, headers=headers)
                        response.raise_for_status()
                        img = Image.open(BytesIO(response.content))
                        # Ensure image is in RGB mode for vLLM compatibility
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb') as tmp_file:
                            img.save(tmp_file, format='PNG')
                            img_path = tmp_file.name
                            print(f"✅ Downloaded image from URL: {img.size} {img.mode}, saved to {img_path}")
                            loaded_images.append(img_path)
                    except Exception as e:
                        print(f"❌ Failed to download image from URL {img_data}: {e}")
                        raise
                elif '/' not in img_data or len(img_data) > 200:
                    # Likely base64 data (no path separators, or long string)
                    # For vLLM multiprocessing, we need to save to a temp file instead of passing PIL objects
                    try:
                        # Decode base64 to image
                        img_bytes = base64.b64decode(img_data)
                        img = Image.open(BytesIO(img_bytes))
                        # Ensure image is in RGB mode for vLLM compatibility
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Save to temporary file and pass the path instead of PIL object
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', mode='wb') as tmp_file:
                            img.save(tmp_file, format='PNG')
                            img_path = tmp_file.name
                            print(f"✅ Loaded image from base64 data: {img.size} {img.mode}, saved to {img_path}")
                            loaded_images.append(img_path)
                    except Exception as e:
                        print(f"❌ Failed to decode base64 image: {e}")
                        raise
                else:
                    # File path - verify it exists and pass the path
                    img_file = Path(img_data)
                    if not img_file.exists():
                        raise FileNotFoundError(f"Image file not found: {img_data}")
                    loaded_images.append(str(img_file))
            
            # For Qwen2.5-VL and similar models, ensure prompt includes vision tokens
            # If the prompt doesn't have vision tokens, add them
            if '<|vision_start|>' not in prompt and '<|image_pad|>' not in prompt:
                # Wrap the prompt with Qwen2.5-VL vision tokens
                # Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n
                formatted_prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    f"{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                prompt = formatted_prompt
            
            # Create multimodal input
            # For Qwen2-VL models, vLLM expects "image" (singular) key with a LIST of images
            multi_modal_data = {"image": loaded_images}  # Always pass as list
            
            return TextPrompt(
                prompt=prompt,
                multi_modal_data=multi_modal_data
            )
        except ImportError:
            # Fall back to text-only if multimodal not available
            logger.warning("Multimodal imports not available, falling back to text-only mode")
            return prompt

    async def _generate_complete(
        self,
        inputs,  # Can be str or TextPrompt
        sampling_params,
        request_id: str,
    ) -> Dict:
        """Generate complete response (non-streaming)."""
        try:
            results_generator = self.engine.generate(inputs, sampling_params, request_id)
            
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
        except Exception as e:
            import traceback
            console.print(f"[red]❌ vLLM EngineCore error details:[/red]")
            console.print(f"[red]Error type: {type(e).__name__}[/red]")
            console.print(f"[red]Error message: {str(e)}[/red]")
            console.print(f"[yellow]Full traceback:[/yellow]")
            traceback.print_exc()
            raise
        
        if final_output is None:
            return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        
        # Return the generated text with usage stats
        generated_text = final_output.outputs[0].text
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(final_output.outputs[0].token_ids)
        
        return {
            "text": generated_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    async def _generate_stream(
        self,
        inputs,  # Can be str or TextPrompt
        sampling_params,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        results_generator = self.engine.generate(inputs, sampling_params, request_id)
        
        previous_text = ""
        async for request_output in results_generator:
            text = request_output.outputs[0].text
            # Yield only the new tokens
            new_text = text[len(previous_text):]
            if new_text:
                yield new_text
            previous_text = text

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        image: Optional[str] = None,
        structured_outputs: Optional[Dict] = None,
        **kwargs
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """OpenAI-compatible chat completion with vision support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate (None = auto-determine, default 512)
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stream: Whether to stream the response
            image: Optional image URL or base64 data for vision models
            structured_outputs: Optional structured output parameters
            **kwargs: Additional generation parameters

        Returns:
            Chat completion response in OpenAI format
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            if structured_outputs:
                max_tokens = 2048  # Higher default for structured outputs
            else:
                max_tokens = 512  # Standard default

        # Format messages into a single prompt
        # This is a simple implementation - you may need to customize for specific models
        prompt = self._format_chat_messages(messages)
        
        if stream:
            return self._chat_completion_stream(messages, prompt, max_tokens, temperature, top_p, image=image, structured_outputs=structured_outputs, **kwargs)
        else:
            return await self._chat_completion_complete(messages, prompt, max_tokens, temperature, top_p, image=image, structured_outputs=structured_outputs, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt string.
        
        This is a simple implementation. For specific models, you should use their
        official chat template from the tokenizer.
        """
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        
        # Add final "Assistant:" to prompt the model to respond
        formatted_parts.append("Assistant:")
        
        return "\n".join(formatted_parts)

    async def _chat_completion_complete(
        self,
        messages: List[Dict[str, str]],
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str] = None,
        structured_outputs: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """Generate complete chat completion response."""
        # Convert single image to list format
        images = [image] if image else None
        
        # Keep original prompt string for token counting
        prompt_str = prompt
        
        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            images=images,
            structured_outputs=structured_outputs,
            **kwargs
        )
        
        # Extract text from result
        response_text = result.get('text', '') if isinstance(result, dict) else str(result)
        usage_info = result.get('usage', {}) if isinstance(result, dict) else {}
        
        # Format as OpenAI-compatible response
        return {
            "id": f"chatcmpl-{id(result)}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": self.model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_info if usage_info else {
                "prompt_tokens": len(prompt_str.split()),  # Rough estimate
                "completion_tokens": len(response_text.split()),  # Rough estimate
                "total_tokens": len(prompt_str.split()) + len(response_text.split()),
            },
        }

    async def _chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str] = None,
        structured_outputs: Optional[Dict] = None,
        **kwargs
    ) -> AsyncIterator[Dict]:
        """Generate streaming chat completion response."""
        # Convert single image to list format
        images = [image] if image else None
        
        stream_generator = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            images=images,
            structured_outputs=structured_outputs,
            **kwargs
        )
        
        async for chunk in stream_generator:
            # Format as OpenAI-compatible streaming response
            yield {
                "id": f"chatcmpl-{id(chunk)}",
                "object": "chat.completion.chunk",
                "created": int(asyncio.get_event_loop().time()),
                "model": self.model_path,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": chunk,
                        },
                        "finish_reason": None,
                    }
                ],
            }
        
        # Send final chunk with finish_reason
        yield {
            "id": f"chatcmpl-final",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_event_loop().time()),
            "model": self.model_path,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "quantization": self.quantization,
        }


# Global inference service instance (will be managed by FastAPI lifespan)
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> Optional[InferenceService]:
    """Get the global inference service instance."""
    return _inference_service


def set_inference_service(service: Optional[InferenceService]) -> None:
    """Set the global inference service instance."""
    global _inference_service
    _inference_service = service
