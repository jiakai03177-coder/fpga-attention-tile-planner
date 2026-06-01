# FPGA Attention Tile Planner

Plan self-attention query/key tile sizes for FPGA and accelerator SRAM budgets.

This small CLI helps answer early hardware mapping questions:

- Which query/key tile pairs fit a local SRAM budget?
- How much on-chip memory is used by Q, K, V, score, and output tiles?
- How does GQA reduce K/V tile storage and HBM traffic?
- What rough per-layer HBM traffic and tile arithmetic intensity should I expect?
- How many BRAM18K / URAM / DSP48 slices does a tile configuration need?
- Does the design fit a target FPGA device (VU9P, ZCU102, VCK190, …)?

## Demo

```bash
python -m fpga_attention_tile_planner.cli --preset llama3-8b --seq-len 4096 --sram-kib 512 --device vu9p
```

Example output:

```text
FPGA Attention Tile Planner
===========================
hidden_size=4096 heads=32 kv_heads=8
seq_len=4096 batch_size=1 dtype=fp16 sram=512 KiB device=VU9P (Alveo U200/U250)

Top fitting tile candidates
---------------------------
q_tile  k_tile  SRAM        util   BRAM18K     URAM       DSP              FLOPs/tile  HBM/layer  AI
------  ------  ----------  -----  ----------  ---------  ---------------  ----------  ---------  -----
16      32      416.00 KiB  81.2%  186 (4.3%)  12 (1.2%)  12.29K (179.6%)  8.39M       4.06 GiB   21.33
8       64      416.00 KiB  81.2%  186 (4.3%)  12 (1.2%)  12.29K (179.6%)  8.39M       8.06 GiB   21.33
16      16      336.00 KiB  65.6%  150 (3.5%)  10 (1.0%)  12.29K (179.6%)  4.19M       4.06 GiB   12.80
```

## Install From GitHub

```bash
pip install git+https://github.com/jiakai03177-coder/fpga-attention-tile-planner.git
```

Local development:

```bash
git clone https://github.com/jiakai03177-coder/fpga-attention-tile-planner.git
cd fpga-attention-tile-planner
python -m pip install -e .
```

## Usage

Use a preset:

```bash
attention-tiles --preset llama3-8b --seq-len 8192 --sram-kib 1024
```

Or pass a custom shape:

```bash
attention-tiles \
  --hidden-size 4096 \
  --heads 32 \
  --kv-heads 8 \
  --seq-len 8192 \
  --sram-kib 512 \
  --dtype fp16
```

Try custom tile candidates:

```bash
attention-tiles --preset mistral-7b --seq-len 4096 --query-tiles 16,32,64 --key-tiles 128,256,512
```

Target a specific FPGA device for resource utilization:

```bash
attention-tiles --preset llama3-8b --seq-len 4096 --sram-kib 512 --device vu9p
```

Reserve SRAM for double-buffered K/V tiles:

```bash
attention-tiles --preset llama3-8b --seq-len 4096 --sram-kib 768 --double-buffer-kv
```

JSON output:

```bash
attention-tiles --preset llama3-8b --seq-len 4096 --sram-kib 512 --json
```

CSV export:

```bash
attention-tiles --preset llama3-8b --seq-len 4096 --sram-kib 512 --csv tiles.csv
```

## Model Presets

The presets are common shape shortcuts, not performance claims:

- `llama2-7b`
- `llama3-8b`
- `mistral-7b`

## FPGA Device Presets

Use `--device` to see BRAM/URAM/DSP utilization against a target device:

- `vu9p` — VU9P (Alveo U200/U250): 2160 BRAM36K, 960 URAM, 6840 DSP
- `vu13p` — VU13P (Alveo U55C): 2688 BRAM36K, 1280 URAM, 12288 DSP
- `zcu102` — ZCU102 (ZU9EG): 912 BRAM36K, 0 URAM, 2520 DSP
- `zcu104` — ZCU104 (ZU7EV): 312 BRAM36K, 96 URAM, 1728 DSP
- `vck190` — VCK190 (VC1902): 967 BRAM36K, 463 URAM, 1968 DSP

## Development

```bash
python -m unittest discover -s tests
python -m fpga_attention_tile_planner.cli --preset llama3-8b --seq-len 4096 --sram-kib 512
```

## Roadmap

- Add Markdown/SVG report export.
- ~~Add DSP and BRAM/URAM resource estimates.~~ *(done in v0.2.0)*
- Add grouped-query and multi-batch schedule comparisons.
- Add streaming softmax workspace modeling.

## License

MIT
