"""
Arena module for tournament-style bargaining matches.

Provides a standardized interface compatible with Google DeepMind's game arena pattern.
"""

from .adapter import (
    ArenaEnvironment,
    ArenaMatchRunner,
    ArenaMatchResult,
    ArenaMove,
    create_arena_match,
)

__all__ = [
    "ArenaEnvironment",
    "ArenaMatchRunner",
    "ArenaMatchResult",
    "ArenaMove",
    "create_arena_match",
]
