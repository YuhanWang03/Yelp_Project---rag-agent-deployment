"""
Backend factory.

Usage:
    from yelp_rag_agent.backends import load_backend

    backend = load_backend("configs/ollama.yaml")
    backend = load_backend("configs/lmdeploy.yaml",
                           overrides={"model": "Qwen/Qwen2.5-7B-AWQ"})
"""

import yaml

from yelp_rag_agent.backends.base import BaseBackend
from yelp_rag_agent.backends.ollama import OllamaBackend
from yelp_rag_agent.backends.lmdeploy import LMDeployBackend

_REGISTRY: dict[str, type[BaseBackend]] = {
    "ollama"   : OllamaBackend,
    "lmdeploy" : LMDeployBackend,
}


def load_backend(config_path: str,
                 overrides: dict | None = None) -> BaseBackend:
    """
    Load a backend from a YAML config file.

    Args:
        config_path: Path to a YAML file (e.g. "configs/ollama.yaml").
        overrides:   Optional dict that overrides specific YAML fields.
                     Useful for CLI --model / --base-url arguments.

    Returns:
        A concrete BaseBackend instance.

    Raises:
        ValueError if the backend type is not registered.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg.update(overrides)

    backend_type = cfg.get("backend")
    cls = _REGISTRY.get(backend_type)
    if cls is None:
        raise ValueError(
            f"Unknown backend: {backend_type!r}. "
            f"Available: {list(_REGISTRY)}"
        )
    return cls.from_config(cfg)


__all__ = ["load_backend", "BaseBackend", "OllamaBackend", "LMDeployBackend"]
