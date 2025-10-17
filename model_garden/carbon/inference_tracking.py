"""Inference-specific carbon tracking."""

import asyncio
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from .tracker import CarbonTracker
from .database import get_emissions_db


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
        self.session_tracker: Optional[CarbonTracker] = None
        self.session_start_time: Optional[float] = None
        self.request_count = 0
        self.total_tokens = 0
        
    def start_session(self) -> None:
        """Start tracking a session (e.g., when model is loaded)."""
        if self.session_tracker is not None:
            return  # Already tracking
        
        # Generate session ID
        session_id = f"inference-{self.model_name.replace('/', '-')}-{int(time.time())}"
        
        self.session_tracker = CarbonTracker(
            job_id=session_id,
            job_type="inference",
            output_dir=Path(f"storage/logs/{session_id}")
        )
        self.session_tracker.start()
        self.session_start_time = time.time()
        self.request_count = 0
        self.total_tokens = 0
        
    def record_request(self, tokens_generated: int = 0) -> None:
        """Record an inference request."""
        self.request_count += 1
        self.total_tokens += tokens_generated
    
    def stop_session(self) -> Optional[Dict[str, Any]]:
        """Stop tracking and save emissions data."""
        if self.session_tracker is None:
            return None
        
        emissions_data = self.session_tracker.stop()
        
        if emissions_data:
            # Add inference-specific metrics
            duration = time.time() - (self.session_start_time or time.time())
            emissions_data['model_name'] = self.model_name
            emissions_data['request_count'] = self.request_count
            emissions_data['total_tokens'] = self.total_tokens
            emissions_data['requests_per_second'] = self.request_count / duration if duration > 0 else 0
            emissions_data['tokens_per_second'] = self.total_tokens / duration if duration > 0 else 0
            
            # Calculate per-request metrics
            if self.request_count > 0:
                emissions_data['emissions_per_request_g'] = (
                    emissions_data['emissions_kg_co2'] * 1000 / self.request_count
                )
                emissions_data['energy_per_request_wh'] = (
                    emissions_data.get('energy_consumed_kwh', 0) * 1000 / self.request_count
                )
            
            # Calculate per-token metrics
            if self.total_tokens > 0:
                emissions_data['emissions_per_1k_tokens_g'] = (
                    emissions_data['emissions_kg_co2'] * 1000000 / self.total_tokens
                )
        
        self.session_tracker = None
        self.session_start_time = None
        
        return emissions_data
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current tracking statistics without stopping."""
        if self.session_tracker is None:
            return {
                'tracking': False,
                'request_count': 0,
                'total_tokens': 0,
            }
        
        duration = time.time() - (self.session_start_time or time.time())
        
        # Get live emissions from CodeCarbon
        live_emissions_kg = self.session_tracker.get_live_emissions()
        
        stats = {
            'tracking': True,
            'model_name': self.model_name,
            'request_count': self.request_count,
            'total_tokens': self.total_tokens,
            'duration_seconds': duration,
            'requests_per_second': self.request_count / duration if duration > 0 else 0,
            'tokens_per_second': self.total_tokens / duration if duration > 0 else 0,
        }
        
        # Add emissions data if available
        if live_emissions_kg is not None:
            stats['emissions_kg_co2'] = live_emissions_kg
            stats['emissions_g_co2'] = live_emissions_kg * 1000
            
            # Calculate per-request metrics
            if self.request_count > 0:
                stats['emissions_per_request_g'] = (live_emissions_kg * 1000) / self.request_count
            
            # Calculate per-token metrics
            if self.total_tokens > 0:
                stats['emissions_per_1k_tokens_g'] = (live_emissions_kg * 1000000) / self.total_tokens
        
        return stats


# Global inference tracker instance
_inference_tracker: Optional[InferenceEmissionsTracker] = None


def get_inference_tracker() -> Optional[InferenceEmissionsTracker]:
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


def stop_inference_tracker() -> Optional[Dict[str, Any]]:
    """Stop the global inference emissions tracker."""
    global _inference_tracker
    
    if _inference_tracker is None:
        return None
    
    emissions_data = _inference_tracker.stop_session()
    _inference_tracker = None
    
    return emissions_data
