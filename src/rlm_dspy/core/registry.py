"""Registry pattern for extensible components.

Learned from modaic: plugin system with freezing support.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RegistryError(Exception):
    """Raised for registry-related errors."""

    pass


class Registry(Generic[T]):
    """
    A registry for named components.

    Learned from modaic's Registry pattern:
    - Register components by name
    - Support freezing to prevent modifications
    - Decorator-based registration
    - Fast cached lookups

    Usage:
        strategies = Registry[Strategy]("strategies")

        @strategies.register("chunked")
        class ChunkedStrategy(Strategy):
            ...

        strategies.freeze()
        strategy = strategies.get("chunked")
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type[T] | T] = {}
        self._frozen = False
        self._cache: dict[str, T] = {}

    def register(
        self,
        name: str,
        override: bool = False,
    ) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a component.

        Args:
            name: Name to register under
            override: Allow overriding existing registration
        """

        def decorator(cls: type[T]) -> type[T]:
            self.add(name, cls, override=override)
            return cls

        return decorator

    def add(
        self,
        name: str,
        component: type[T] | T,
        override: bool = False,
    ) -> None:
        """
        Add a component to the registry.

        Args:
            name: Name to register under
            component: Class or instance to register
            override: Allow overriding existing registration
        """
        if self._frozen:
            raise RegistryError(f"Registry '{self.name}' is frozen, cannot add '{name}'")

        if name in self._registry and not override:
            raise RegistryError(f"'{name}' already registered in '{self.name}'. Use override=True to replace.")

        self._registry[name] = component
        self._cache.pop(name, None)  # Invalidate cache
        logger.debug("Registered %s in %s", name, self.name)

    def get(self, name: str, default: T | None = None) -> T | None:
        """
        Get a component by name.

        Returns instance (instantiates class if needed).
        """
        if name not in self._registry:
            return default

        # Check cache first (fast path from modaic)
        if name in self._cache:
            return self._cache[name]

        component = self._registry[name]

        # If it's a class, instantiate it
        if isinstance(component, type):
            instance = component()
            self._cache[name] = instance
            return instance

        return component

    def get_class(self, name: str) -> type[T] | None:
        """Get the registered class without instantiating."""
        component = self._registry.get(name)
        if isinstance(component, type):
            return component
        return None

    def freeze(self) -> None:
        """Freeze the registry to prevent further modifications."""
        self._frozen = True
        logger.debug("Registry '%s' frozen with %d entries", self.name, len(self._registry))

    def unfreeze(self) -> None:
        """Unfreeze the registry (use with caution)."""
        self._frozen = False

    def list(self) -> list[str]:
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


# Global registries for RLM-DSPy components
# Using Any since concrete types are defined at runtime
strategies: Registry[Any] = Registry("strategies")
processors: Registry[Any] = Registry("processors")
models: Registry[Any] = Registry("models")


def builtin_strategy(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a builtin strategy."""
    return strategies.register(name)


def builtin_processor(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a builtin processor."""
    return processors.register(name)


@lru_cache(maxsize=128)
def load_class(class_path: str) -> type:
    """
    Dynamically load a class from a dotted path.

    Learned from modaic's auto-loading pattern with LRU cache.

    Args:
        class_path: Dotted path like "rlm_dspy.core.RLM"

    Returns:
        The loaded class
    """
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid class path: {class_path}")

    module_path, class_name = parts

    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)
