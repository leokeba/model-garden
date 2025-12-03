"""Dataset format conversion utilities.

This module provides converters between different dataset formats used in
vision-language training. Extracted from VisionLanguageTrainer to improve
modularity and testability.

Supported formats:
- Simple: {text, image, response}
- OpenAI messages: {messages: [{role, content: [{type, text/image}]}]}
- VQA: {question, answer/answers, image} (ScienceQA, DocVQA, etc.)
"""

from typing import Any

from model_garden.utils.console import console


class DatasetFormatConverter:
    """Handles conversion between different dataset formats.

    This class provides methods to detect and convert between various dataset
    formats commonly used in vision-language training.

    Supported formats:
    - Simple format: {text, image, response}
    - OpenAI messages format: {messages: [{role, content}]}
    - VQA formats: {question, answer/answers, image}

    Example:
        >>> converter = DatasetFormatConverter()
        >>> if converter.detect_format(example) == "vqa":
        ...     simple = converter.convert_vqa_to_simple(example)
    """

    @staticmethod
    def detect_format(example: dict) -> str:
        """Detect the format of a dataset example.

        Args:
            example: A single dataset example

        Returns:
            Format string: "simple", "messages", "vqa", or "unknown"
        """
        if not isinstance(example, dict):
            return "unknown"

        # Check for OpenAI messages format
        if "messages" in example:
            messages = example["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                return "messages"

        # Check for VQA format
        has_question = "question" in example
        has_answer = "answer" in example or "answers" in example
        has_image = "image" in example
        if has_question and has_answer and has_image:
            return "vqa"

        # Check for simple format
        has_text = "text" in example
        has_response = "response" in example or "output" in example
        if (has_text or has_image) and has_response:
            return "simple"

        return "unknown"

    @staticmethod
    def detect_vqa_format(example: dict) -> bool:
        """Detect if example uses VQA format (question + answer/answers).

        Args:
            example: A single dataset example to check

        Returns:
            True if the example appears to be in VQA format
        """
        if not isinstance(example, dict):
            return False
        has_question = "question" in example
        has_answer = "answer" in example or "answers" in example
        has_image = "image" in example
        return has_question and has_answer and has_image

    @staticmethod
    def convert_vqa_to_simple(example: dict) -> dict[str, Any]:
        """Convert VQA-style formats to simple format.

        Handles formats like:
        - ScienceQA: {question, choices, answer (index), solution, image}
        - VQA: {question, answers (list), image}
        - DocVQA: {question, answers (list), image}

        Args:
            example: A VQA-style dataset example

        Returns:
            Dict with 'text', 'image', and 'response' keys

        Raises:
            ValueError: If example is not a dict
        """
        if not isinstance(example, dict):
            raise ValueError(f"Expected VQA example to be a dict, got {type(example).__name__}")

        result: dict[str, Any] = {
            "text": example.get("question", ""),
            "image": example.get("image"),
            "response": "",
        }

        # Handle different answer formats
        if "choices" in example and "answer" in example:
            # ScienceQA format - answer is index into choices
            answer_idx = example.get("answer", 0)
            choices = example.get("choices", [])
            if isinstance(answer_idx, int) and answer_idx < len(choices):
                result["response"] = choices[answer_idx]

                # Add solution if available
                solution = example.get("solution", "")
                if solution:
                    result["response"] = f"{result['response']}. {solution}"

        elif "answers" in example:
            # Generic VQA format - answers is a list
            answers = example.get("answers", [])
            if isinstance(answers, list) and answers:
                # Get first answer
                if isinstance(answers[0], str):
                    result["response"] = answers[0]
                elif isinstance(answers[0], dict):
                    result["response"] = answers[0].get("answer", "")
            elif isinstance(answers, str):
                result["response"] = answers

        elif "answer" in example:
            # Simple answer field
            answer = example.get("answer")
            if isinstance(answer, str):
                result["response"] = answer
            elif isinstance(answer, int) and "choices" in example:
                # Answer is index
                choices = example.get("choices", [])
                if answer < len(choices):
                    result["response"] = choices[answer]

        return result

    @staticmethod
    def convert_messages_to_simple(messages: list[dict]) -> dict[str, str | None]:
        """Convert OpenAI messages format to simple format.

        Extracts the system message, first image and text from user message,
        and assistant's response. This ensures compatibility with
        UnslothVisionDataCollator while preserving the original system prompt.

        Args:
            messages: List of OpenAI-style messages

        Returns:
            Dict with 'text', 'image', 'response', and 'system' keys

        Raises:
            ValueError: If messages is not a list or contains invalid entries
        """
        # Validate input
        if not isinstance(messages, list):
            raise ValueError(
                f"Expected 'messages' to be a list, got {type(messages).__name__}. "
                f"Check your dataset format - it should have a 'messages' field containing a list."
            )

        if len(messages) == 0:
            console.print("[yellow]⚠️  Empty messages list in dataset example[/yellow]")
            return {"text": "", "image": None, "response": "", "system": ""}

        result: dict[str, str | None] = {"text": "", "image": None, "response": "", "system": ""}

        for idx, msg in enumerate(messages):
            # Validate each message is a dict
            if not isinstance(msg, dict):
                console.print(
                    f"[yellow]⚠️  Message at index {idx} is not a dict "
                    f"(got {type(msg).__name__}), skipping[/yellow]"
                )
                continue

            role = msg.get("role", "")
            content = msg.get("content", [])

            # Validate content is iterable (list or similar)
            if not isinstance(content, (list, tuple)):
                # Content might be a plain string (simplified format)
                if isinstance(content, str):
                    if role == "system" and not result["system"]:
                        result["system"] = content
                    elif role == "user" and not result["text"]:
                        result["text"] = content
                    elif role == "assistant" and not result["response"]:
                        result["response"] = content
                    continue
                else:
                    console.print(
                        f"[yellow]⚠️  Message content at index {idx} is not a list or string "
                        f"(got {type(content).__name__}), skipping[/yellow]"
                    )
                    continue

            if role == "system":
                # Extract system message
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and not result["system"]:
                        result["system"] = item.get("text", "")

            elif role == "user":
                # Extract text and image from user message
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type", "")
                    if item_type == "text" and not result["text"]:
                        result["text"] = item.get("text", "")
                    elif item_type in ("image", "image_url") and not result["image"]:
                        # Handle both old format (type: "image", image: "...")
                        # and new format (type: "image_url", image_url: {url: "..."})
                        image_data = item.get("image", item.get("image_url", {}))
                        if isinstance(image_data, dict):
                            image_data = image_data.get("url", "")
                        result["image"] = image_data

            elif role == "assistant":
                # Extract response from assistant message
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and not result["response"]:
                        result["response"] = item.get("text", "")

        # Warn if essential fields are missing
        if not result["text"] and not result["image"]:
            console.print(
                "[yellow]⚠️  No text or image found in user message - check dataset format[/yellow]"
            )
        if not result["response"]:
            console.print(
                "[yellow]⚠️  No response found in assistant message - check dataset format[/yellow]"
            )

        return result

    @staticmethod
    def to_openai_messages(
        text: str,
        image: Any,
        response: str,
        system_message: str = "You are a helpful assistant that can analyze images.",
    ) -> dict:
        """Convert simple format to OpenAI messages format.

        Args:
            text: User's text/question
            image: PIL Image or image reference
            response: Assistant's response
            system_message: System message to use

        Returns:
            Dict with 'messages' key containing OpenAI-style messages
        """
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}],
                },
            ],
        }



