"""Unit tests for chat template handling in InferenceService."""

from model_garden.inference import InferenceService


def test_should_override_chat_template_by_default():
    """Default engine args should allow custom chat templates."""
    service = InferenceService(model_path="dummy-model")
    engine_args = {"model": "dummy-model"}

    assert service._should_override_chat_template(engine_args) is True


def test_should_not_override_chat_template_for_mistral():
    """Mistral tokenizer mode should rely on native templates."""
    service = InferenceService(model_path="dummy-model")
    engine_args = {
        "model": "dummy-model",
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
    }

    assert service._should_override_chat_template(engine_args) is False
