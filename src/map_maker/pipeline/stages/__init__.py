"""Pipeline stage registrations."""

import importlib

from . import geometry  # noqa: F401
from . import tectonics  # noqa: F401
from . import world_age  # noqa: F401
from . import geology  # noqa: F401
from . import erosion  # noqa: F401


def ensure_builtin_stages() -> None:
    """Restore built-in registrations if an embedding process cleared the registry."""

    from ..registry import registry

    modules = {
        "geometry": geometry,
        "tectonics": tectonics,
        "world_age": world_age,
        "geology": geology,
        "erosion": erosion,
    }
    for stage_name, module in modules.items():
        if stage_name not in registry():
            importlib.reload(module)


__all__ = ["ensure_builtin_stages"]
