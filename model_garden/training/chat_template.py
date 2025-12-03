"""Chat template detection utilities.

This module provides utilities for automatically detecting chat template markers
from tokenizers. Extracted from VisionLanguageTrainer to improve modularity.

The ChatTemplateDetector works with any chat model (Qwen, Llama, Phi, Mistral, etc.)
by analyzing the tokenizer's chat template to extract the markers used for
user and assistant messages.
"""

from typing import Any

from model_garden.utils.console import console

# Common chat template patterns for known model families
FALLBACK_MARKERS: dict[str, tuple[str, str]] = {
    "qwen": ("<|im_start|>user", "<|im_start|>assistant"),
    "llama": ("[INST]", "[/INST]"),
    "phi": ("<|user|>", "<|assistant|>"),
    "mistral": ("[INST]", "[/INST]"),
    "gemma": ("<start_of_turn>user", "<start_of_turn>model"),
    "vicuna": ("USER:", "ASSISTANT:"),
    "chatml": ("<|im_start|>user", "<|im_start|>assistant"),
}


class ChatTemplateDetector:
    """Auto-detect chat markers from tokenizer templates.

    This class analyzes a model's tokenizer to extract the chat markers used
    for user and assistant messages. This enables training code to work with
    any chat model without hardcoding model-specific templates.

    Example:
        >>> detector = ChatTemplateDetector()
        >>> instruction, response = detector.detect(processor)
        >>> print(f"User marker: {instruction}")
        >>> print(f"Assistant marker: {response}")
    """

    def __init__(self, verbose: bool = True):
        """Initialize the detector.

        Args:
            verbose: Whether to print detection results to console
        """
        self.verbose = verbose

    def detect(self, processor: Any) -> tuple[str, str]:
        """Detect instruction and response markers from tokenizer's chat template.

        This method automatically extracts the chat markers used by the model's
        tokenizer, making the code work with any chat model without hardcoding
        model-specific templates.

        Args:
            processor: The model's processor (must have apply_chat_template method)

        Returns:
            Tuple of (instruction_marker, response_marker)
            - instruction_marker: The marker before user messages
            - response_marker: The marker before assistant messages

        Example:
            >>> detector = ChatTemplateDetector()
            >>> instruction, response = detector.detect(processor)
            >>> print(f"User: {instruction}, Assistant: {response}")
            User: <|im_start|>user, Assistant: <|im_start|>assistant
        """
        try:
            # Apply template to sample messages with placeholders
            sample = [
                {"role": "user", "content": "__USER_PLACEHOLDER__"},
                {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
            ]

            formatted = processor.apply_chat_template(
                sample, tokenize=False, add_generation_prompt=False
            )

            # Find placeholder positions
            user_idx = formatted.find("__USER_PLACEHOLDER__")
            assistant_idx = formatted.find("__ASSISTANT_PLACEHOLDER__")

            if user_idx > 0 and assistant_idx > 0:
                # Extract the line before user content
                lines_before_user = formatted[:user_idx].split("\n")
                instruction_marker = None
                for line in reversed(lines_before_user):
                    if line.strip() and not line.strip().endswith("_PLACEHOLDER__"):
                        instruction_marker = line.strip()
                        break

                # Extract the line before assistant content
                lines_before_assistant = formatted[:assistant_idx].split("\n")
                response_marker = None
                for line in reversed(lines_before_assistant):
                    if line.strip() and not line.strip().endswith("_PLACEHOLDER__"):
                        response_marker = line.strip()
                        break

                if instruction_marker and response_marker:
                    if self.verbose:
                        console.print("[green]✓ Auto-detected chat markers:[/green]")
                        console.print(f"  instruction_part: {repr(instruction_marker)}")
                        console.print(f"  response_part: {repr(response_marker)}")
                    return instruction_marker, response_marker

        except Exception as e:
            if self.verbose:
                console.print(f"[yellow]⚠️  Could not auto-detect chat markers: {e}[/yellow]")

        # Fallback to model-specific markers
        return self.get_fallback_markers(processor)

    def get_fallback_markers(self, processor: Any) -> tuple[str, str]:
        """Get fallback chat markers based on model type.

        Uses model type to select appropriate markers from known patterns.

        Args:
            processor: The model's processor

        Returns:
            Tuple of (instruction_marker, response_marker)
        """
        model_type = self._get_model_type(processor)

        # Try to find a matching model family
        for family, markers in FALLBACK_MARKERS.items():
            if family in model_type:
                if self.verbose:
                    console.print(f"[cyan]Using fallback markers for {family}:[/cyan]")
                    console.print(f"  instruction_part: {repr(markers[0])}")
                    console.print(f"  response_part: {repr(markers[1])}")
                return markers

        # Generic fallback
        if self.verbose:
            console.print(
                "[yellow]⚠️  Using generic markers - training may not work optimally[/yellow]"
            )
            console.print(
                "[yellow]    Consider adding model-specific markers to FALLBACK_MARKERS[/yellow]"
            )

        markers = ("User:", "Assistant:")
        if self.verbose:
            console.print("[cyan]Using generic markers:[/cyan]")
            console.print(f"  instruction_part: {repr(markers[0])}")
            console.print(f"  response_part: {repr(markers[1])}")

        return markers

    def _get_model_type(self, processor: Any) -> str:
        """Extract model type string from processor.

        Args:
            processor: The model's processor

        Returns:
            Lowercase model type string, or empty string if not found
        """
        try:
            # Try tokenizer config
            if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "config"):
                return processor.tokenizer.config.model_type.lower()
        except Exception:
            pass

        try:
            # Try processor config directly
            if hasattr(processor, "config") and hasattr(processor.config, "model_type"):
                return processor.config.model_type.lower()
        except Exception:
            pass

        try:
            # Try to get from name
            if hasattr(processor, "name_or_path"):
                name = processor.name_or_path.lower()
                for family in FALLBACK_MARKERS.keys():
                    if family in name:
                        return family
        except Exception:
            pass

        return ""



