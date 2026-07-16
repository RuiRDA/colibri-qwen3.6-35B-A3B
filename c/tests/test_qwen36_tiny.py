import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np


from openai_server import Engine, render_qwen_chat
HERE = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("convert_qwen36_tiny", HERE / "tools" / "convert_qwen36.py")
Q = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = Q
SPEC.loader.exec_module(Q)


class QwenTinyEndToEnd(unittest.TestCase):
    """Exercise every Qwen block type through the real quantized C loader."""

    def test_quantized_hybrid_model_matches_residual_oracle(self):
        engine = HERE / ("qwen36.exe" if os.name == "nt" else "qwen36")
        if not engine.exists():
            self.skipTest("build qwen36 first")

        rng = np.random.default_rng(36035)
        D, L, V = 8, 4, 300
        E, TOP, I, SI = 4, 2, 6, 6
        QH, KVH, HD = 2, 1, 4
        LKH, LVH, KD, VD, CK = 1, 2, 4, 4, 2
        key, value, conv = LKH * KD, LVH * VD, 2 * LKH * KD + LVH * VD

        config = {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": D, "num_hidden_layers": L, "vocab_size": V,
                "num_experts": E, "num_experts_per_tok": TOP,
                "moe_intermediate_size": I, "shared_expert_intermediate_size": SI,
                "num_attention_heads": QH, "num_key_value_heads": KVH,
                "head_dim": HD, "partial_rotary_factor": 0.5,
                "full_attention_interval": 4, "linear_num_key_heads": LKH,
                "linear_num_value_heads": LVH, "linear_key_head_dim": KD,
                "linear_value_head_dim": VD, "linear_conv_kernel_dim": CK,
                "rms_norm_eps": 1e-6, "eos_token_id": 257,
                "rope_parameters": {"rope_theta": 10000000.0},
            },
        }

        tensors = {}
        emb = rng.normal(0, 0.5, (V, D)).astype(np.float32)
        tensors["model.language_model.embed_tokens.weight"] = emb
        tensors["lm_head.weight"] = emb.copy()
        tensors["model.language_model.norm.weight"] = np.zeros(D, np.float32)

        def zero(name, shape):
            tensors[name] = np.zeros(shape, np.float32)

        for li in range(L):
            p = f"model.language_model.layers.{li}."
            zero(p + "input_layernorm.weight", (D,))
            zero(p + "post_attention_layernorm.weight", (D,))
            if (li + 1) % 4 == 0:
                zero(p + "self_attn.q_proj.weight", (QH * HD * 2, D))
                zero(p + "self_attn.k_proj.weight", (KVH * HD, D))
                zero(p + "self_attn.v_proj.weight", (KVH * HD, D))
                zero(p + "self_attn.o_proj.weight", (D, QH * HD))
                zero(p + "self_attn.q_norm.weight", (HD,))
                zero(p + "self_attn.k_norm.weight", (HD,))
            else:
                zero(p + "linear_attn.in_proj_qkv.weight", (conv, D))
                zero(p + "linear_attn.in_proj_z.weight", (value, D))
                zero(p + "linear_attn.in_proj_b.weight", (LVH, D))
                zero(p + "linear_attn.in_proj_a.weight", (LVH, D))
                zero(p + "linear_attn.out_proj.weight", (D, value))
                zero(p + "linear_attn.conv1d.weight", (conv, 1, CK))
                zero(p + "linear_attn.A_log", (LVH,))
                zero(p + "linear_attn.dt_bias", (LVH,))
                tensors[p + "linear_attn.norm.weight"] = np.ones(VD, np.float32)
            zero(p + "mlp.gate.weight", (E, D))
            zero(p + "mlp.shared_expert.gate_proj.weight", (SI, D))
            zero(p + "mlp.shared_expert.up_proj.weight", (SI, D))
            zero(p + "mlp.shared_expert.down_proj.weight", (D, SI))
            zero(p + "mlp.shared_expert_gate.weight", (1, D))
            for eid in range(E):
                ep = p + f"mlp.experts.{eid}."
                zero(ep + "gate_proj.weight", (I, D))
                zero(ep + "up_proj.weight", (I, D))
                zero(ep + "down_proj.weight", (D, I))

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            src, model = root / "source.safetensors", root / "model"
            model.mkdir()
            specs = [{"name": name, "dtype": "F32", "shape": list(value.shape)}
                     for name, value in tensors.items()]
            writer = Q.SafeTensorWriter(src, specs, {"format": "tiny-qwen36"})
            for name, value in tensors.items():
                writer.write(name, value)
            writer.close()
            Q.convert_shard(src, model / "out-00001-of-00001.safetensors", 4, 4, 8, 16)
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
            vocab, extra = {}, 0
            for byte in range(256):
                if byte in direct:
                    cp = byte
                else:
                    cp = 256 + extra
                    extra += 1
                vocab[chr(cp)] = byte
            added = [
                {"id": 256, "content": "<|im_start|>"},
                {"id": 257, "content": "<|im_end|>"},
            ]
            added += [{"id": i, "content": f"<extra_{i}>"} for i in range(258, V)]
            tokenizer = {"model": {"type": "BPE", "vocab": vocab,
                                    "merges": ["x y", ["z", "q"]]},
                         "added_tokens": added}
            (model / "tokenizer.json").write_text(
                json.dumps(tokenizer, ensure_ascii=False), encoding="utf-8")


            q, scales = Q.quant_rows(emb, 8)
            deq = q.view(np.int8).astype(np.float32) * scales[:, None]
            expected = []
            for token in (1, 2, 3):
                x = deq[token]
                x = x / np.sqrt(np.mean(x * x) + 1e-6)
                expected.append(int(np.argmax(deq @ x)))

            env = dict(os.environ, SNAP=str(model), IDS="1,2,3", CTX="8",
                       OMP_NUM_THREADS="1", TEMP="0")
            run = subprocess.run([str(engine), "2"], env=env, text=True,
                                 capture_output=True, timeout=30, check=True)
            got = [int(line.split()[1]) for line in run.stdout.splitlines()
                   if line.startswith("PRED ")]
            self.assertEqual(got, expected, run.stderr)

            prompt = render_qwen_chat([{"role": "user", "content": "a"}])
            self.assertEqual(prompt,
                             "<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n")
            runtime = Engine(engine, model, cap=2, max_tokens=2,
                             env=dict(os.environ, CTX="64", OMP_NUM_THREADS="1", TEMP="0"),
                             kv_slots=1)
            pieces = []
            try:
                stats = runtime.generate(prompt, 1, 0.0, 1.0, pieces.append)
            finally:
                runtime.close()
            self.assertGreater(stats["prompt_tokens"], 0)
            self.assertLessEqual(stats["completion_tokens"], 1)
            self.assertIsInstance("".join(pieces), str)


if __name__ == "__main__":
    unittest.main()
