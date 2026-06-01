"""Plan self-attention tile shapes for FPGA and accelerator SRAM budgets."""

from .core import AttentionShape, TileCandidate, plan_tiles, tile_candidate

__all__ = ["AttentionShape", "TileCandidate", "plan_tiles", "tile_candidate"]

