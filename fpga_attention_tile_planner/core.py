from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "int8": 1,
    "fp8": 1,
}


PRESETS = {
    "llama2-7b": {
        "hidden_size": 4096,
        "heads": 32,
        "kv_heads": 32,
    },
    "llama3-8b": {
        "hidden_size": 4096,
        "heads": 32,
        "kv_heads": 8,
    },
    "mistral-7b": {
        "hidden_size": 4096,
        "heads": 32,
        "kv_heads": 8,
    },
}


@dataclass(frozen=True)
class AttentionShape:
    hidden_size: int
    heads: int
    seq_len: int
    kv_heads: int | None = None
    batch_size: int = 1
    dtype: str = "fp16"

    @property
    def normalized_kv_heads(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    @property
    def dtype_bytes(self) -> int:
        try:
            return DTYPE_BYTES[self.dtype.lower()]
        except KeyError as exc:
            known = ", ".join(sorted(DTYPE_BYTES))
            raise ValueError(f"unknown dtype {self.dtype!r}; expected one of: {known}") from exc

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        return self.hidden_size // self.heads

    @property
    def group_size(self) -> int:
        return self.heads // self.normalized_kv_heads

    def validate(self) -> None:
        values = {
            "hidden_size": self.hidden_size,
            "heads": self.heads,
            "seq_len": self.seq_len,
            "kv_heads": self.normalized_kv_heads,
            "batch_size": self.batch_size,
        }
        for name, value in values.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.heads % self.normalized_kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        _ = self.head_dim
        _ = self.dtype_bytes


@dataclass(frozen=True)
class TileCandidate:
    query_tile: int
    key_tile: int
    total_sram_bytes: int
    sram_utilization: float
    q_bytes: int
    k_bytes: int
    v_bytes: int
    score_bytes: int
    output_bytes: int
    tile_hbm_bytes: int
    tile_flops: int
    arithmetic_intensity_flops_per_byte: float
    hbm_bytes_per_query_block: int
    hbm_bytes_per_layer: int
    q_blocks_per_layer: int
    fits: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def tile_candidate(
    shape: AttentionShape,
    query_tile: int,
    key_tile: int,
    sram_bytes: int,
    double_buffer_kv: bool = False,
) -> TileCandidate:
    shape.validate()
    if query_tile < 1:
        raise ValueError("query_tile must be >= 1")
    if key_tile < 1:
        raise ValueError("key_tile must be >= 1")
    if sram_bytes < 1:
        raise ValueError("sram_bytes must be >= 1")

    query_tile = min(query_tile, shape.seq_len)
    key_tile = min(key_tile, shape.seq_len)
    dtype_bytes = shape.dtype_bytes
    head_dim = shape.head_dim

    q_bytes = shape.batch_size * query_tile * shape.heads * head_dim * dtype_bytes
    k_bytes = shape.batch_size * key_tile * shape.normalized_kv_heads * head_dim * dtype_bytes
    v_bytes = k_bytes
    score_bytes = shape.batch_size * shape.heads * query_tile * key_tile * dtype_bytes
    output_bytes = shape.batch_size * query_tile * shape.heads * head_dim * dtype_bytes

    kv_buffer_multiplier = 2 if double_buffer_kv else 1
    total_sram_bytes = q_bytes + (k_bytes + v_bytes) * kv_buffer_multiplier + score_bytes + output_bytes

    tile_hbm_bytes = q_bytes + k_bytes + v_bytes + output_bytes
    tile_flops = 4 * shape.batch_size * shape.heads * query_tile * key_tile * head_dim
    arithmetic_intensity = tile_flops / tile_hbm_bytes if tile_hbm_bytes else 0.0

    q_blocks = ceil(shape.seq_len / query_tile)
    kv_full_sequence_bytes = (
        shape.batch_size * shape.seq_len * shape.normalized_kv_heads * head_dim * 2 * dtype_bytes
    )
    hbm_bytes_per_query_block = q_bytes + kv_full_sequence_bytes + output_bytes
    hbm_bytes_per_layer = q_blocks * hbm_bytes_per_query_block

    return TileCandidate(
        query_tile=query_tile,
        key_tile=key_tile,
        total_sram_bytes=total_sram_bytes,
        sram_utilization=total_sram_bytes / sram_bytes,
        q_bytes=q_bytes,
        k_bytes=k_bytes,
        v_bytes=v_bytes,
        score_bytes=score_bytes,
        output_bytes=output_bytes,
        tile_hbm_bytes=tile_hbm_bytes,
        tile_flops=tile_flops,
        arithmetic_intensity_flops_per_byte=arithmetic_intensity,
        hbm_bytes_per_query_block=hbm_bytes_per_query_block,
        hbm_bytes_per_layer=hbm_bytes_per_layer,
        q_blocks_per_layer=q_blocks,
        fits=total_sram_bytes <= sram_bytes,
    )


def plan_tiles(
    shape: AttentionShape,
    sram_kib: int,
    query_tiles: list[int],
    key_tiles: list[int],
    top: int = 5,
    double_buffer_kv: bool = False,
) -> list[TileCandidate]:
    if sram_kib < 1:
        raise ValueError("sram_kib must be >= 1")
    if top < 1:
        raise ValueError("top must be >= 1")

    sram_bytes = sram_kib * 1024
    candidates = [
        tile_candidate(shape, query_tile, key_tile, sram_bytes, double_buffer_kv)
        for query_tile in sorted(set(query_tiles))
        for key_tile in sorted(set(key_tiles))
    ]
    fitting = [candidate for candidate in candidates if candidate.fits]
    return sorted(
        fitting,
        key=lambda item: (
            item.query_tile * item.key_tile,
            -item.hbm_bytes_per_layer,
            item.query_tile,
            item.key_tile,
            item.arithmetic_intensity_flops_per_byte,
        ),
        reverse=True,
    )[:top]


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def format_number(value: int | float) -> str:
    units = ["", "K", "M", "B", "T", "P"]
    amount = float(value)
    for unit in units:
        if abs(amount) < 1000 or unit == units[-1]:
            return f"{amount:.2f}{unit}" if unit else str(int(amount))
        amount /= 1000
    return str(value)
