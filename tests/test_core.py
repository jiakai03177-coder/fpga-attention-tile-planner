import unittest

from fpga_attention_tile_planner.core import AttentionShape, plan_tiles, tile_candidate


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


if __name__ == "__main__":
    unittest.main()

