"""Pipeline stage registrations."""

import importlib

from . import geometry  # noqa: F401
from . import planet  # noqa: F401
from . import climate  # noqa: F401
from . import hydrology  # noqa: F401
from . import basin_refinement  # noqa: F401
from . import basin_erosion  # noqa: F401
from . import hydrology_pass2  # noqa: F401
from . import surface_water  # noqa: F401
from . import outlet_incision  # noqa: F401
from . import tectonics  # noqa: F401
from . import world_age  # noqa: F401
from . import geology  # noqa: F401
from . import elevation  # noqa: F401
from . import erosion  # noqa: F401


def ensure_builtin_stages() -> None:
    """Restore built-in registrations if an embedding process cleared the registry."""

    from ..registry import registry

    modules = {
        "geometry": geometry,
        "planet": planet,
        "climate": climate,
        "hydrology": hydrology,
        "basin_refinement": basin_refinement,
        "basin_erosion": basin_erosion,
        "hydrology_pass2": hydrology_pass2,
        "surface_water": surface_water,
        "outlet_incision": outlet_incision,
        "tectonics": tectonics,
        "world_age": world_age,
        "geology": geology,
        "elevation": elevation,
        "erosion": erosion,
    }
    for stage_name, module in modules.items():
        if stage_name not in registry():
            importlib.reload(module)


__all__ = ["ensure_builtin_stages"]
