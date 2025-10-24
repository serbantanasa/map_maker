"""Legacy world-generation modules preserved for backwards compatibility."""

from .generate import export_dataset, generate_world, render_png
from .cli import main as legacy_cli_main
from .core.tectonics import generate_tectonic_field, save_tectonic_field

__all__ = [
    "generate_world",
    "render_png",
    "export_dataset",
    "legacy_cli_main",
    "generate_tectonic_field",
    "save_tectonic_field",
]
