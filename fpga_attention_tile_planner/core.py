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

DTYPE_DSPS_PER_MAC = {
    "fp32": 5,
    "float32": 5,
    "bf16": 3,
    "bfloat16": 3,
    "fp16": 3,
    "float16": 3,
    "int8": 1,
    "fp8": 1,
}

BRAM18K_BITS = 18 * 1024
BRAM36K_BITS = 36 * 1024
URAM_BITS = 288 * 1024


@dataclass(frozen=True)
class FPGADevice:
    name: str
    bram36k: int
    uram: int
    dsp: int

    @property
    def bram18k(self) -> int:
        return self.bram36k * 2

    @property
    def total_bram_bytes(self) -> int:
        return self.bram36k * BRAM36K_BITS // 8

    @property
    def total_uram_bytes(self) -> int:
        return self.uram * URAM_BITS // 8


FPGA_DEVICES = {
    "vu9p": FPGADevice(name="VU9P (Alveo U200/U250)", bram36k=2160, uram=960, dsp=6840),
    "vu13p": FPGADevice(name="VU13P (Alveo U55C)", bram36k=2688, uram=1280, dsp=12288),
    "zcu102": FPGADevice(name="ZCU102 (ZU9EG)", bram36k=912, uram=0, dsp=2520),
    "zcu104": FPGADevice(name="ZCU104 (ZU7EV)", bram36k=312, uram=96, dsp=1728),
    "vck190": FPGADevice(name="VCK190 (VC1902)", bram36k=967, uram=463, dsp=1968),
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
class ResourceEstimate:
    bram18k: int
    bram36k: int
    uram: int
    dsp: int
    bram18k_pct: float | None = None
    bram36k_pct: float | None = None
    uram_pct: float | None = None
    dsp_pct: float | None = None
    device_name: str | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


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
    resources: ResourceEstimate | None = None

    def to_dict(self) -> dict[str, int | float | bool]:
        d = asdict(self)
        if self.resources is not None:
            d["resources"] = self.resources.to_dict()
        else:
            d["resources"] = None
        return d


def estimate_resources(
    total_sram_bytes: int,
    tile_flops: int,
    dtype: str,
    device: FPGADevice | None = None,
    head_dim: int = 128,
    heads: int = 1,
) -> ResourceEstimate:
    total_bits = total_sram_bytes * 8

    bram36k_count = ceil(total_bits / BRAM36K_BITS) if total_bits > 0 else 0
    bram18k_count = bram36k_count * 2
    uram_count = ceil(total_bits / URAM_BITS) if total_bits > 0 else 0

    if device is not None and device.uram == 0:
        uram_count = 0

    dsps_per_mac = DTYPE_DSPS_PER_MAC.get(dtype.lower(), 3)
    parallel_macs = heads * head_dim
    dsp_count = parallel_macs * dsps_per_mac

    bram18k_pct = None
    bram36k_pct = None
    uram_pct = None
    dsp_pct = None
    device_name = None

    if device is not None:
        device_name = device.name
        bram18k_pct = (bram18k_count / device.bram18k * 100) if device.bram18k > 0 else 0.0
        bram36k_pct = (bram36k_count / device.bram36k * 100) if device.bram36k > 0 else 0.0
        uram_pct = (uram_count / device.uram * 100) if device.uram > 0 else 0.0
        dsp_pct = (dsp_count / device.dsp * 100) if device.dsp > 0 else 0.0

    return ResourceEstimate(
        bram18k=bram18k_count,
        bram36k=bram36k_count,
        uram=uram_count,
        dsp=dsp_count,
        bram18k_pct=bram18k_pct,
        bram36k_pct=bram36k_pct,
        uram_pct=uram_pct,
        dsp_pct=dsp_pct,
        device_name=device_name,
    )


def tile_candidate(
    shape: AttentionShape,
    query_tile: int,
    key_tile: int,
    sram_bytes: int,
    double_buffer_kv: bool = False,
    device: FPGADevice | None = None,
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

    resources = estimate_resources(
        total_sram_bytes, tile_flops, shape.dtype, device, head_dim, shape.heads,
    )

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
        resources=resources,
    )


def plan_tiles(
    shape: AttentionShape,
    sram_kib: int,
    query_tiles: list[int],
    key_tiles: list[int],
    top: int = 5,
    double_buffer_kv: bool = False,
    device: FPGADevice | None = None,
) -> list[TileCandidate]:
    if sram_kib < 1:
        raise ValueError("sram_kib must be >= 1")
    if top < 1:
        raise ValueError("top must be >= 1")

    sram_bytes = sram_kib * 1024
    candidates = [
        tile_candidate(shape, query_tile, key_tile, sram_bytes, double_buffer_kv, device)
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
