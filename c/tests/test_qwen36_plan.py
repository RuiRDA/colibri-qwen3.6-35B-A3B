import json
import struct
import tempfile
import unittest
from pathlib import Path

from resource_plan import GB, analyze_model, build_plan


def write_shard(path):
    tensors = [
        ("model.embed_tokens.weight", 100),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", 30),
        ("model.layers.0.mlp.experts.0.up_proj.weight", 30),
    ]
    offset, header, payload = 0, {}, b""
    for name, size in tensors:
        header[name] = {"dtype": "U8", "shape": [size],
                        "data_offsets": [offset, offset + size]}
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class QwenResourcePlanTest(unittest.TestCase):
    def test_nested_config_accounts_for_delta_state_and_full_kv(self):
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            text = {
                "num_hidden_layers": 4, "num_experts": 4,
                "full_attention_interval": 4,
                "num_key_value_heads": 1, "head_dim": 4,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 4, "linear_value_head_dim": 4,
            }
            (model / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5_moe", "text_config": text,
            }))
            write_shard(model / "out.safetensors")
            info = analyze_model(model)
            self.assertEqual(info["config"]["num_experts"], 4)
            plan = build_plan(model, ram_gb=8, context=32,
                              available_memory=16 * GB, available_disk=20 * GB,
                              gpus=[], physical_cpus=4)
            expected_kv = 1 * 32 * 2 * 1 * 4 * 4
            expected_delta = 3 * 2 * 4 * 4 * 4
            expected = int(3.7 * GB + 64 * 60 + expected_kv + expected_delta)
            self.assertEqual(plan["tiers"]["ram"]["runtime_bytes"], expected)


if __name__ == "__main__":
    unittest.main()
