import unittest

from fpga_attention_tile_planner.core import (
    AttentionShape,
    FPGA_DEVICES,
    FPGADevice,
    estimate_resources,
    plan_tiles,
    tile_candidate,
)


class CoreTests(unittest.TestCase):
    def test_tile_candidate_fits_sram_budget(self):
        shape = AttentionShape(hidden_size=16, heads=2, kv_heads=1, seq_len=128)

        candidate = tile_candidate(shape, query_tile=16, key_tile=32, sram_bytes=64 * 1024)

        self.assertTrue(candidate.fits)
        self.assertEqual(candidate.query_tile, 16)
        self.assertEqual(candidate.key_tile, 32)
        self.assertEqual(shape.group_size, 2)
        self.assertLessEqual(candidate.total_sram_bytes, 64 * 1024)
        self.assertGreater(candidate.arithmetic_intensity_flops_per_byte, 0)

    def test_plan_tiles_returns_largest_fitting_candidates_first(self):
        shape = AttentionShape(hidden_size=64, heads=4, kv_heads=2, seq_len=256)

        candidates = plan_tiles(
            shape,
            sram_kib=128,
            query_tiles=[16, 32, 64],
            key_tiles=[64, 128],
            top=2,
        )

        self.assertEqual(len(candidates), 2)
        self.assertGreaterEqual(
            candidates[0].query_tile * candidates[0].key_tile,
            candidates[1].query_tile * candidates[1].key_tile,
        )

    def test_gqa_reduces_kv_tile_bytes(self):
        mha = AttentionShape(hidden_size=64, heads=4, kv_heads=4, seq_len=256)
        gqa = AttentionShape(hidden_size=64, heads=4, kv_heads=1, seq_len=256)

        mha_candidate = tile_candidate(mha, query_tile=16, key_tile=64, sram_bytes=128 * 1024)
        gqa_candidate = tile_candidate(gqa, query_tile=16, key_tile=64, sram_bytes=128 * 1024)

        self.assertLess(gqa_candidate.k_bytes + gqa_candidate.v_bytes, mha_candidate.k_bytes + mha_candidate.v_bytes)
        self.assertLess(gqa_candidate.hbm_bytes_per_layer, mha_candidate.hbm_bytes_per_layer)

    def test_invalid_shape_is_rejected(self):
        shape = AttentionShape(hidden_size=63, heads=4, seq_len=128)

        with self.assertRaises(ValueError):
            tile_candidate(shape, query_tile=16, key_tile=64, sram_bytes=128 * 1024)


class ResourceEstimateTests(unittest.TestCase):
    def test_tile_candidate_includes_resources(self):
        shape = AttentionShape(hidden_size=16, heads=2, kv_heads=1, seq_len=128)
        candidate = tile_candidate(shape, query_tile=16, key_tile=32, sram_bytes=64 * 1024)

        self.assertIsNotNone(candidate.resources)
        self.assertGreater(candidate.resources.bram18k, 0)
        self.assertGreater(candidate.resources.dsp, 0)

    def test_resource_estimate_without_device(self):
        res = estimate_resources(total_sram_bytes=4096, tile_flops=1024, dtype="fp16")

        self.assertGreater(res.bram18k, 0)
        self.assertGreater(res.uram, 0)
        self.assertGreater(res.dsp, 0)
        self.assertIsNone(res.bram18k_pct)
        self.assertIsNone(res.device_name)

    def test_resource_estimate_with_device(self):
        device = FPGA_DEVICES["vu9p"]
        res = estimate_resources(total_sram_bytes=4096, tile_flops=1024, dtype="fp16", device=device)

        self.assertIsNotNone(res.bram18k_pct)
        self.assertIsNotNone(res.dsp_pct)
        self.assertEqual(res.device_name, device.name)
        self.assertGreaterEqual(res.bram18k_pct, 0)
        self.assertGreaterEqual(res.dsp_pct, 0)

    def test_int8_uses_fewer_dsps_than_fp16(self):
        res_fp16 = estimate_resources(total_sram_bytes=4096, tile_flops=1024, dtype="fp16")
        res_int8 = estimate_resources(total_sram_bytes=4096, tile_flops=1024, dtype="int8")

        self.assertLess(res_int8.dsp, res_fp16.dsp)

    def test_device_without_uram_uses_only_bram(self):
        device = FPGA_DEVICES["zcu102"]
        self.assertEqual(device.uram, 0)

        res = estimate_resources(total_sram_bytes=4096, tile_flops=1024, dtype="fp16", device=device)
        self.assertEqual(res.uram, 0)
        self.assertGreater(res.bram36k, 0)

    def test_plan_tiles_with_device(self):
        shape = AttentionShape(hidden_size=64, heads=4, kv_heads=2, seq_len=256)
        device = FPGA_DEVICES["vu9p"]

        candidates = plan_tiles(
            shape,
            sram_kib=128,
            query_tiles=[16, 32],
            key_tiles=[64, 128],
            top=2,
            device=device,
        )

        self.assertTrue(candidates)
        for c in candidates:
            self.assertIsNotNone(c.resources)
            self.assertEqual(c.resources.device_name, device.name)
            self.assertIsNotNone(c.resources.bram18k_pct)

    def test_fpga_device_bram18k_is_double_bram36k(self):
        device = FPGA_DEVICES["vu9p"]
        self.assertEqual(device.bram18k, device.bram36k * 2)

    def test_to_dict_includes_resources(self):
        shape = AttentionShape(hidden_size=16, heads=2, kv_heads=1, seq_len=128)
        candidate = tile_candidate(shape, query_tile=16, key_tile=32, sram_bytes=64 * 1024)

        d = candidate.to_dict()
        self.assertIn("resources", d)
        self.assertIn("bram18k", d["resources"])
        self.assertIn("dsp", d["resources"])


if __name__ == "__main__":
    unittest.main()

