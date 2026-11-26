import importlib
import importlib.metadata
from types import ModuleType


def sdk_httpx(package: str) -> ModuleType:
    """Return the HTTP library (`httpx` or its successor `httpx2`) that the installed SDK is built on."""
    requirements = [req.replace(" ", "") for req in importlib.metadata.requires(package) or []]
    uses_httpx2 = any(req.startswith("httpx2") and "extra==" not in req for req in requirements)
    return importlib.import_module("httpx2" if uses_httpx2 else "httpx")
