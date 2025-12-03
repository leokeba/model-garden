"""Centralized Rich console instance.

This module provides a single shared Console instance for consistent
output formatting across Model Garden.
"""

from rich.console import Console

# Shared console instance for consistent output
console = Console()

__all__ = ["console"]
