import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("convert_qwen36", HERE / "tools" / "convert_qwen36.py")
Q = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = Q
SPEC.loader.exec_module(Q)


class QwenConverterTests(unittest.TestCase):
    def test_classification_and_renaming(self):
        p = "model.language_model.layers.0."
        self.assertEqual(Q.normalized_name(p + "mlp.gate.weight"), "model.layers.0.mlp.gate.weight")
        self.assertEqual(Q.classify(p + "mlp.gate.weight"), "f32")
        self.assertEqual(Q.classify(p + "mlp.experts.7.up_proj.weight"), "expert")
        self.assertEqual(Q.classify(p + "linear_attn.in_proj_qkv.weight"), "dense")
        self.assertEqual(Q.classify("model.visual.blocks.0.attn.qkv.weight"), "skip")
        self.assertEqual(Q.classify("mtp.layers.0.self_attn.q_proj.weight"), "skip")

    def test_fp8_known_values(self):
        lut = Q.FP8_LUT
        self.assertEqual(float(lut[0x00]), 0.0)
        self.assertEqual(float(lut[0x38]), 1.0)
        self.assertEqual(float(lut[0x7E]), 448.0)
        self.assertTrue(np.isnan(lut[0x7F]))
        self.assertEqual(float(lut[0xB8]), -1.0)

    def test_q4_packing_matches_colibri(self):
        w = np.array([[-8.0, -7.0, 0.0, 7.0, 3.0]], dtype=np.float32)
        q, s = Q.quant_rows(w, 4)
        self.assertAlmostEqual(float(s[0]), 8.0 / 7.0, places=6)
        # Decode exactly as qwen36.c / glm.c: low nibble first, both biased by 8.
        got = []
        for i in range(w.shape[1]):
            nibble = (int(q[0, i // 2]) >> (4 * (i & 1))) & 15
            got.append(nibble - 8)
        expected = np.clip(np.rint(w[0] / s[0]), -8, 7).astype(int).tolist()
        self.assertEqual(got, expected)

    def test_safetensors_writer_roundtrip_header(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "tiny.safetensors"
            w = Q.SafeTensorWriter(path, [
                {"name": "x", "dtype": "F32", "shape": [2]},
                {"name": "q", "dtype": "U8", "shape": [3]},
            ], {"format": "test"})
            w.write("x", np.array([1.25, -2.0], np.float32))
            w.write("q", np.array([1, 2, 3], np.uint8))
            w.close()
            with Q.SafeTensorReader(path) as r:
                np.testing.assert_array_equal(r.f32("x"), [1.25, -2.0])
                np.testing.assert_array_equal(np.asarray(r.raw("q")), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
