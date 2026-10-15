"""
test_gem_sublayer_hooks.py — hook-based sublayer capture on TINY random-init
transformers built from config (hidden 64, 3 layers, CPU, no downloads).
Checks Delta == attn + mlp numerically, that block outputs equal
hidden_states[l+1] except at the last layer (post-final-norm), and that
post-LN / training mode / left padding are refused. Skipped without torch
and transformers. Split from test_gem_schema.py so the schema tests run
without torch.
"""

import numpy as np
import pytest

from rosetta_tools.gem import ConceptGEM, GEMExtractionConfig, SettledSubspace, load_gem, save_gem
from rosetta_tools.gem_sublayer import attach_layer_profile, decompose_settled_subspace

# ---------------------------------------------------------------------------
# Tiny random-init transformers (CPU, no downloads)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class _Tok:
    """Minimal right-padding character tokenizer (avoids any download)."""
    pad_token = 0
    eos_token = 0
    padding_side = "right"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        ids = [[(ord(ch) % 90) + 1 for ch in t][:max_length] for t in texts]
        m = max(map(len, ids))
        return {
            "input_ids": torch.tensor([i + [0] * (m - len(i)) for i in ids]),
            "attention_mask": torch.tensor([[1] * len(i) + [0] * (m - len(i)) for i in ids]),
        }


_KW = dict(num_hidden_layers=3, num_attention_heads=4, vocab_size=100)


def _cfg(name):
    T = transformers
    return {
        "qwen2": lambda: T.Qwen2Config(hidden_size=64, intermediate_size=128, num_key_value_heads=2, **_KW),
        "llama": lambda: T.LlamaConfig(hidden_size=64, intermediate_size=128, num_key_value_heads=2, **_KW),
        "gemma2": lambda: T.Gemma2Config(hidden_size=64, intermediate_size=128, num_key_value_heads=2,
                                         head_dim=16, **_KW),
        "gpt2": lambda: T.GPT2Config(n_embd=64, n_layer=3, n_head=4, vocab_size=100),
        "neox_parallel": lambda: T.GPTNeoXConfig(hidden_size=64, intermediate_size=128,
                                                 use_parallel_residual=True, **_KW),
        "opt_pre_ln": lambda: T.OPTConfig(hidden_size=64, ffn_dim=128, word_embed_proj_dim=64,
                                          do_layer_norm_before=True, **_KW),
        "mixtral": lambda: T.MixtralConfig(hidden_size=64, intermediate_size=64, num_key_value_heads=2,
                                           num_local_experts=4, num_experts_per_tok=2, **_KW),
        "qwen2_moe": lambda: T.Qwen2MoeConfig(hidden_size=64, intermediate_size=64, num_key_value_heads=2,
                                              moe_intermediate_size=32, shared_expert_intermediate_size=32,
                                              num_experts=4, num_experts_per_tok=2, **_KW),
        "opt_post_ln": lambda: T.OPTConfig(hidden_size=64, ffn_dim=128, word_embed_proj_dim=64,
                                           do_layer_norm_before=False, **_KW),
    }[name]()


def _tiny(name, seed=0):
    torch.manual_seed(seed)
    return transformers.AutoModel.from_config(_cfg(name)).eval()


TEXTS = ["the cat sat", "a considerably longer sentence for padding", "no", "it is not so"]


@pytest.mark.parametrize("name", ["qwen2", "llama", "gemma2", "gpt2", "neox_parallel", "opt_pre_ln",
                                  "mixtral", "qwen2_moe"])
def test_tiny_model_additivity(name):
    from rosetta_tools.gem_sublayer import capture_sublayer_activations
    model = _tiny(name)
    cap = capture_sublayer_activations(model, _Tok(), TEXTS, batch_size=3)
    delta = cap.stream - np.concatenate([cap.embed[None], cap.stream[:-1]], 0)
    np.testing.assert_allclose(delta, cap.attn + cap.mlp, atol=1e-5)
    assert cap.max_recon_rel_stream < 1e-5
    assert cap.max_continuity_gap == 0.0
    assert cap.architecture.sublayer_post_norm == (name == "gemma2")
    assert cap.architecture.parallel_block == (name == "neox_parallel")
    assert cap.architecture.moe == (name in ("mixtral", "qwen2_moe"))


@pytest.mark.parametrize("name", ["qwen2", "gpt2", "neox_parallel", "gemma2", "opt_pre_ln"])
def test_block_outputs_match_hidden_states_except_last(name):
    from rosetta_tools.gem_sublayer import capture_sublayer_activations
    model, tok = _tiny(name), _Tok()
    cap = capture_sublayer_activations(model, tok, TEXTS[:3], batch_size=3)
    enc = tok(TEXTS[:3])
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    last = enc["attention_mask"].sum(1) - 1
    rows = torch.arange(3)
    for l in range(cap.stream.shape[0]):
        ref = hs[l + 1][rows, last].float().numpy()
        if l < cap.stream.shape[0] - 1:
            np.testing.assert_allclose(cap.stream[l], ref, atol=1e-6)
        else:  # hidden_states[-1] is post-final-norm; captured separately (D2)
            assert np.abs(cap.stream[l] - ref).max() > 1e-3
            np.testing.assert_allclose(cap.final_norm, ref, atol=1e-6)


def test_post_ln_refused_and_detected_from_data():
    from rosetta_tools.gem_sublayer import capture_sublayer_activations
    model = _tiny("opt_post_ln")
    with pytest.raises(ValueError, match="post-LN"):
        capture_sublayer_activations(model, _Tok(), TEXTS)
    # Even if the config lies, the numeric check catches it.
    model.config.do_layer_norm_before = True
    with pytest.raises(ValueError, match="additivity check failed"):
        capture_sublayer_activations(model, _Tok(), TEXTS)
    cap = capture_sublayer_activations(model, _Tok(), TEXTS, allow_inexact=True)
    assert cap.architecture.additivity_expected_exact is False
    assert cap.max_recon_rel_stream > 0.1


def test_training_mode_and_left_padding_refused():
    from rosetta_tools.gem_sublayer import capture_sublayer_activations
    with pytest.raises(RuntimeError, match="eval"):
        capture_sublayer_activations(_tiny("qwen2").train(), _Tok(), TEXTS)
    tok = _Tok(); tok.padding_side = "left"
    with pytest.raises(ValueError, match="right padding"):
        capture_sublayer_activations(_tiny("qwen2"), tok, TEXTS)


def test_bf16_within_tolerance():
    from rosetta_tools.gem_sublayer import ADDITIVITY_TOL, capture_sublayer_activations
    model = _tiny("qwen2").to(torch.bfloat16)
    cap = capture_sublayer_activations(model, _Tok(), TEXTS)
    assert cap.model_dtype == "bfloat16"
    assert 0 < cap.max_recon_rel_stream < ADDITIVITY_TOL["bfloat16"]


def test_end_to_end_tiny_qwen2(tmp_path):
    from rosetta_tools.gem import extract_concept_layer_profile  # lazy re-export
    model = _tiny("qwen2")
    pos = [f"it is true that item {i} works" for i in range(12)]
    neg = [f"it is not true that item {i} works" for i in range(12)]
    prof, arrays, docs = extract_concept_layer_profile(
        model, _Tok(), pos, neg, model_id="tiny-qwen2-random",
        config=GEMExtractionConfig(batch_size=5, n_permutations=10),
    )
    assert arrays["profile.attn_acts"].shape == (3, 24, 64)  # D3 default on
    assert prof.final_norm is not None and prof.architecture.final_norm_hook == "norm"
    assert len(prof.layers) == 3 and prof.hidden_dim == 64
    assert prof.architecture.attn_hook == "self_attn"
    assert prof.provenance.model_dtype == "float32" and prof.provenance.n_pos == 12
    assert "self_attn" in prof.provenance.hook_points["attn"]
    for rec in prof.layers:
        assert rec.recon_residual_max_rel_stream < 1e-5
    sub = SettledSubspace(layer=1, directions=[arrays["profile.directions"][1].tolist()], k_effective=1)
    dec = decompose_settled_subspace(sub, arrays).sublayer_decomposition
    assert abs(dec.reconstruction_residual[0]) < 1e-3 * max(1.0, abs(dec.stream_diff_proj[0]))
    gem = ConceptGEM(concept="tiny", model_id="tiny-qwen2-random", n_nodes=0, nodes=[])
    out = save_gem(attach_layer_profile(gem, prof), tmp_path / "gem_tiny.json", arrays, docs)
    g2 = load_gem(tmp_path / "gem_tiny.json")
    assert g2.layer_profile.architecture.decoder_class == "Qwen2DecoderLayer"
    assert out.sidecar.sha256
