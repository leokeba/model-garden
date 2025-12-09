"""Inference-specific carbon tracking."""

import time
from pathlib import Path
from typing import Any

from .tracker import CarbonTracker


class InferenceEmissionsTracker:
    """
    Tracks carbon emissions for inference operations.

    Can track at different granularities:
    - Per-request (fine-grained, higher overhead)
    - Per-session (aggregate multiple requests)
    - Per-model (lifetime emissions for a model)
    """

    def __init__(self, model_name: str):
        """
        Initialize inference emissions tracker.

        Args:
            model_name: Name of the model being served
        """
        self.model_name = model_name
        self.session_tracker: CarbonTracker | None = None
        self.session_start_time: float | None = None
        self.request_count = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0

    def start_session(self) -> None:
        """Start tracking a session (e.g., when model is loaded)."""
        if self.session_tracker is not None:
            return  # Already tracking

        # Generate session ID
        session_id = f"inference-{self.model_name.replace('/', '-')}-{int(time.time())}"

        self.session_tracker = CarbonTracker(
            job_id=session_id,
            job_type="inference",
            output_dir=Path(f"storage/logs/{session_id}"),
            model_name=self.model_name,
        )
        self.session_tracker.start()
        self.session_start_time = time.time()
        self.request_count = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0

    def record_request(self, tokens_generated: int = 0, prompt_tokens: int = 0) -> None:
        """Record an inference request."""
        self.request_count += 1
        self.total_tokens += tokens_generated
        self.total_prompt_tokens += prompt_tokens

    def get_request_emissions(self) -> dict[str, Any] | None:
        """
        Get emissions data for the current request using delta measurement.
        Call this before and after a request to get per-request emissions.

        Returns:
            Dictionary with current emissions snapshot
        """
        if self.session_tracker is None:
            return None

        return self.session_tracker.get_live_emissions()

    def stop_session(self) -> dict[str, Any] | None:
        """Stop tracking and save emissions data."""
        if self.session_tracker is None:
            return None

        emissions_data = self.session_tracker.stop()

        if emissions_data:
            # Add inference-specific metrics
            duration = time.time() - (self.session_start_time or time.time())
            emissions_data["model_name"] = self.model_name
            emissions_data["request_count"] = self.request_count
            emissions_data["total_tokens"] = self.total_tokens
            emissions_data["prompt_tokens"] = self.total_prompt_tokens
            emissions_data["completion_tokens"] = (
                self.total_tokens
            )  # total_tokens tracks generated tokens
            emissions_data["requests_per_second"] = (
                self.request_count / duration if duration > 0 else 0
            )
            emissions_data["tokens_per_second"] = (
                self.total_tokens / duration if duration > 0 else 0
            )

            # Calculate per-request metrics
            if self.request_count > 0:
                emissions_data["emissions_per_request_g"] = (
                    emissions_data["emissions_kg_co2"] * 1000 / self.request_count
                )
                emissions_data["energy_per_request_wh"] = (
                    emissions_data.get("energy_consumed_kwh", 0) * 1000 / self.request_count
                )

            # Calculate per-token metrics
            if self.total_tokens > 0:
                emissions_data["emissions_per_1k_tokens_g"] = (
                    emissions_data["emissions_kg_co2"] * 1000000 / self.total_tokens
                )

        self.session_tracker = None
        self.session_start_time = None

        return emissions_data

    def get_current_stats(self) -> dict[str, Any]:
        """Get current tracking statistics without stopping."""
        if self.session_tracker is None:
            return {
                "tracking": False,
                "request_count": 0,
                "total_tokens": 0,
            }

        duration = time.time() - (self.session_start_time or time.time())

        # Get live emissions from CodeCarbon
        live_emissions_data = self.session_tracker.get_live_emissions()

        stats = {
            "tracking": True,
            "model_name": self.model_name,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "duration_seconds": duration,
            "requests_per_second": self.request_count / duration if duration > 0 else 0,
            "tokens_per_second": self.total_tokens / duration if duration > 0 else 0,
        }

        # Add emissions data if available
        if live_emissions_data is not None:
            live_emissions_kg = live_emissions_data.get("emissions_kg_co2", 0.0)
            stats["emissions_kg_co2"] = live_emissions_kg
            stats["emissions_g_co2"] = live_emissions_kg * 1000
            stats["energy_consumed_kwh"] = live_emissions_data.get("energy_consumed_kwh", 0.0)

            # Calculate per-request metrics
            if self.request_count > 0:
                stats["emissions_per_request_g"] = (live_emissions_kg * 1000) / self.request_count
                stats["energy_per_request_wh"] = (
                    live_emissions_data.get("energy_consumed_kwh", 0.0) * 1000
                ) / self.request_count

            # Calculate per-token metrics
            if self.total_tokens > 0:
                stats["emissions_per_1k_tokens_g"] = (
                    live_emissions_kg * 1000000
                ) / self.total_tokens

        return stats


# Global inference tracker instance
_inference_tracker: InferenceEmissionsTracker | None = None


def get_inference_tracker() -> InferenceEmissionsTracker | None:
    """Get the global inference emissions tracker."""
    return _inference_tracker


def init_inference_tracker(model_name: str) -> InferenceEmissionsTracker:
    """
    Initialize the global inference emissions tracker.

    Args:
        model_name: Name of the model being served

    Returns:
        Inference tracker instance
    """
    global _inference_tracker

    # Stop existing tracker if running
    if _inference_tracker is not None:
        _inference_tracker.stop_session()

    _inference_tracker = InferenceEmissionsTracker(model_name)
    _inference_tracker.start_session()

    return _inference_tracker


def stop_inference_tracker() -> dict[str, Any] | None:
    """Stop the global inference emissions tracker."""
    global _inference_tracker

    if _inference_tracker is None:
        return None

    emissions_data = _inference_tracker.stop_session()
    _inference_tracker = None

    return emissions_data
