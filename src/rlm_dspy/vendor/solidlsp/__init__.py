"""
solidlsp - Unified Language Server Protocol client library.

Vendored from the Serena project (https://github.com/Oraios/serena)
MIT License - Copyright (c) 2025 Oraios AI

This library provides a unified interface for interacting with various
Language Server Protocol (LSP) servers across different programming languages.
"""

from rlm_dspy.vendor.solidlsp.ls import SolidLanguageServer
from rlm_dspy.vendor.solidlsp.ls_config import Language, LanguageServerConfig
from rlm_dspy.vendor.solidlsp.ls_exceptions import SolidLSPException
from rlm_dspy.vendor.solidlsp.settings import SolidLSPSettings

__all__ = [
    "SolidLanguageServer",
    "Language",
    "LanguageServerConfig",
    "SolidLSPException",
    "SolidLSPSettings",
]

__version__ = "0.1.0"
__upstream_version__ = "2025-01"  # Serena version this was vendored from
