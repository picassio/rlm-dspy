"""
Language server configurations for solidlsp.

Each module provides a SolidLanguageServer subclass configured for a specific
language server (e.g., pyright for Python, gopls for Go).
"""
from .common import RuntimeDependency, RuntimeDependencyCollection

__all__ = [
    "RuntimeDependency",
    "RuntimeDependencyCollection",
]
