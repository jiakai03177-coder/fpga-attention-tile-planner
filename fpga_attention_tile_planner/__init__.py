"""Plan self-attention tile shapes for FPGA and accelerator SRAM budgets."""

from .core import (
    AttentionShape,
    FPGADevice,
    FPGA_DEVICES,
    ResourceEstimate,
    TileCandidate,
    estimate_resources,
    plan_tiles,
    tile_candidate,
)

__all__ = [
    "AttentionShape",
    "FPGADevice",
    "FPGA_DEVICES",
    "ResourceEstimate",
    "TileCandidate",
    "estimate_resources",
    "plan_tiles",
    "tile_candidate",
]

