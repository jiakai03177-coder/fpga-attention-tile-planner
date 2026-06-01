from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .core import AttentionShape, FPGA_DEVICES, PRESETS, format_bytes, format_number, plan_tiles


DEFAULT_QUERY_TILES = "8,16,32,64,128"
DEFAULT_KEY_TILES = "16,32,64,128,256"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="attention-tiles",
        description="Plan self-attention query/key tile sizes for FPGA and accelerator SRAM budgets.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Load a common model shape.")
    parser.add_argument("--hidden-size", type=int, help="Model hidden size.")
    parser.add_argument("--heads", type=int, help="Number of query attention heads.")
    parser.add_argument("--kv-heads", type=int, help="Number of key/value heads for GQA or MQA.")
    parser.add_argument("--seq-len", type=int, required=True, help="Context length in tokens.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Defaults to 1.")
    parser.add_argument("--dtype", default="fp16", help="Activation/KV dtype. Defaults to fp16.")
    parser.add_argument("--sram-kib", type=int, default=512, help="On-chip SRAM budget in KiB. Defaults to 512.")
    parser.add_argument("--query-tiles", default=DEFAULT_QUERY_TILES, help="Comma-separated query tile candidates.")
    parser.add_argument("--key-tiles", default=DEFAULT_KEY_TILES, help="Comma-separated key tile candidates.")
    parser.add_argument("--top", type=int, default=5, help="Number of fitting candidates to print. Defaults to 5.")
    parser.add_argument("--double-buffer-kv", action="store_true", help="Reserve extra SRAM for double-buffered K/V tiles.")
    parser.add_argument("--device", choices=sorted(FPGA_DEVICES), help="FPGA device for resource estimates.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--csv", help="Write candidates to a CSV file.")
    return parser


def parse_tile_list(value: str, option_name: str) -> list[int]:
    tiles = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            tile = int(stripped)
        except ValueError as exc:
            raise SystemExit(f"invalid {option_name} value: {stripped}") from exc
        if tile < 1:
            raise SystemExit(f"{option_name} values must be >= 1")
        tiles.append(tile)
    if not tiles:
        raise SystemExit(f"{option_name} must include at least one tile")
    return tiles


def shape_from_args(args: argparse.Namespace) -> AttentionShape:
    values = {}
    if args.preset:
        values.update(PRESETS[args.preset])

    for key, arg_name in [
        ("hidden_size", "hidden_size"),
        ("heads", "heads"),
        ("kv_heads", "kv_heads"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            values[key] = value

    missing = [key for key in ["hidden_size", "heads"] if key not in values]
    if missing:
        missing_args = ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        raise SystemExit(f"missing required model shape: {missing_args} or --preset")

    shape = AttentionShape(
        hidden_size=values["hidden_size"],
        heads=values["heads"],
        kv_heads=values.get("kv_heads"),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    try:
        shape.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return shape


def candidate_row(candidate) -> dict[str, int | float | str | bool]:
    row = candidate.to_dict()
    row.update(
        {
            "total_sram": format_bytes(candidate.total_sram_bytes),
            "tile_flops_formatted": format_number(candidate.tile_flops),
            "hbm_per_query_block": format_bytes(candidate.hbm_bytes_per_query_block),
            "hbm_per_layer": format_bytes(candidate.hbm_bytes_per_layer),
        }
    )
    return row


def write_csv(path: str, candidates) -> Path:
    output_path = Path(path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [candidate_row(candidate) for candidate in candidates]
    if not rows:
        rows = [{"message": "no candidates fit the SRAM budget"}]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_candidates(shape: AttentionShape, candidates, sram_kib: int, device_key: str | None = None) -> None:
    print("FPGA Attention Tile Planner")
    print("===========================")
    print(f"hidden_size={shape.hidden_size} heads={shape.heads} kv_heads={shape.normalized_kv_heads}")
    header_line = f"seq_len={shape.seq_len} batch_size={shape.batch_size} dtype={shape.dtype} sram={sram_kib} KiB"
    if device_key:
        device = FPGA_DEVICES[device_key]
        header_line += f" device={device.name}"
    print(header_line)
    print()

    if not candidates:
        print("No tile candidates fit the SRAM budget.")
        return

    has_device = device_key is not None

    headers = [
        "q_tile",
        "k_tile",
        "SRAM",
        "util",
        "BRAM18K",
        "URAM",
        "DSP",
        "FLOPs/tile",
        "HBM/layer",
        "AI",
    ]
    rows = []
    for candidate in candidates:
        res = candidate.resources
        if has_device and res:
            bram_str = f"{res.bram18k} ({res.bram18k_pct:.1f}%)"
            uram_str = f"{res.uram} ({res.uram_pct:.1f}%)" if res.uram > 0 else "0"
            dsp_str = f"{format_number(res.dsp)} ({res.dsp_pct:.1f}%)"
        elif res:
            bram_str = str(res.bram18k)
            uram_str = str(res.uram)
            dsp_str = format_number(res.dsp)
        else:
            bram_str = "-"
            uram_str = "-"
            dsp_str = "-"

        rows.append(
            [
                str(candidate.query_tile),
                str(candidate.key_tile),
                format_bytes(candidate.total_sram_bytes),
                f"{candidate.sram_utilization:.1%}",
                bram_str,
                uram_str,
                dsp_str,
                format_number(candidate.tile_flops),
                format_bytes(candidate.hbm_bytes_per_layer),
                f"{candidate.arithmetic_intensity_flops_per_byte:.2f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("Top fitting tile candidates")
    print("---------------------------")
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    shape = shape_from_args(args)
    query_tiles = parse_tile_list(args.query_tiles, "--query-tiles")
    key_tiles = parse_tile_list(args.key_tiles, "--key-tiles")

    device = FPGA_DEVICES.get(args.device) if args.device else None

    try:
        candidates = plan_tiles(
            shape,
            sram_kib=args.sram_kib,
            query_tiles=query_tiles,
            key_tiles=key_tiles,
            top=args.top,
            double_buffer_kv=args.double_buffer_kv,
            device=device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    csv_path = write_csv(args.csv, candidates) if args.csv else None

    if args.json:
        print(
            json.dumps(
                {
                    "shape": asdict(shape),
                    "sram_kib": args.sram_kib,
                    "double_buffer_kv": args.double_buffer_kv,
                    "device": args.device,
                    "candidates": [candidate.to_dict() for candidate in candidates],
                },
                indent=2,
            )
        )
        return 0 if candidates else 1

    print_candidates(shape, candidates, args.sram_kib, args.device)
    if csv_path:
        print()
        print(f"CSV written to {csv_path}")
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
