import json
import tempfile
import unittest
from pathlib import Path

from fpga_attention_tile_planner.cli import main


class CliTests(unittest.TestCase):
    def test_cli_json_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            code = main([
                "--preset",
                "llama3-8b",
                "--seq-len",
                "2048",
                "--sram-kib",
                "512",
                "--query-tiles",
                "8,16",
                "--key-tiles",
                "16,32,64",
                "--json",
            ])

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["shape"]["hidden_size"], 4096)
        self.assertEqual(payload["shape"]["kv_heads"], 8)
        self.assertTrue(payload["candidates"])

    def test_cli_table_output(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            code = main([
                "--hidden-size",
                "4096",
                "--heads",
                "32",
                "--kv-heads",
                "8",
                "--seq-len",
                "1024",
                "--sram-kib",
                "512",
            ])

        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("FPGA Attention Tile Planner", text)
        self.assertIn("Top fitting tile candidates", text)
        self.assertIn("q_tile", text)

    def test_cli_writes_csv(self):
        from contextlib import redirect_stdout
        from io import StringIO

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "tiles.csv"
            output = StringIO()
            with redirect_stdout(output):
                code = main([
                    "--preset",
                    "llama3-8b",
                    "--seq-len",
                    "1024",
                    "--sram-kib",
                    "512",
                    "--csv",
                    str(csv_path),
                ])

            self.assertEqual(code, 0)
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("query_tile,key_tile,total_sram_bytes", text)
            self.assertIn("arithmetic_intensity_flops_per_byte", text)


if __name__ == "__main__":
    unittest.main()
