# FPGA Attention Tile Planner

Plan self-attention query/key tile sizes for FPGA and accelerator SRAM budgets.

This small CLI helps answer early hardware mapping questions:

- Which query/key tile pairs fit a local SRAM budget?
- How much on-chip memory is used by Q, K, V, score, and output tiles?
- How does GQA reduce K/V tile storage and HBM traffic?
- What rough per-layer HBM traffic and tile arithmetic intensity should I expect?

## Demo

```bash
python -m fpga_attention_tile_planner.cli --preset llama3-8b --seq-len 4096 --sram-kib 512
```

Example output:

```text
FPGA Attention Tile Planner
===========================
hidden_size=4096 heads=32 kv_heads=8
seq_len=4096 batch_size=1 dtype=fp16 sram=512 KiB

Top fitting tile candidates
---------------------------
q_tile  k_tile  SRAM        util   FLOPs/tile  HBM/q-block  HBM/layer  AI
------  ------  ----------  -----  ----------  -----------  ---------  -----
16      32      416.00 KiB  81.2%  8.39M       16.25 MiB    4.06 GiB   21.33
8       64      416.00 KiB  81.2%  8.39M       16.12 MiB    8.06 GiB   21.33
16      16      336.00 KiB  65.6%  4.19M       16.25 MiB    4.06 GiB   12.80
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

## Development

```bash
python -m unittest discover -s tests
python -m fpga_attention_tile_planner.cli --preset llama3-8b --seq-len 4096 --sram-kib 512
```

## Roadmap

- Add Markdown/SVG report export.
- Add DSP and BRAM/URAM resource estimates.
- Add grouped-query and multi-batch schedule comparisons.
- Add streaming softmax workspace modeling.

## License

MIT
