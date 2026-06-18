"""
Toulouse - High-Performance Card Game Library for Reinforcement Learning

A modern, high-performance card library for RL/MCTS applications.
"""

from importlib.metadata import PackageNotFoundError, version

from .core import (
    Card,
    Deck,
    get_card,
    get_card_system,
    register_card_system,
)

try:
    __version__ = version("toulouse")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "Card",
    "Deck",
    "get_card",
    "get_card_system",
    "register_card_system",
    "__version__",
]
