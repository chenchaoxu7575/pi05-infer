"""Isolation proof: the action expert must be OURS, the PaliGemma prefix STOCK.

Builds the real model exactly as ``bench/standalone_infer_bench.py`` does, then
walks the two towers and asserts where every module class was defined:

  * expert  (``paligemma_with_expert.gemma_expert``)  -> ``pi05_infer.gemma.*``
  * prefix  (``paligemma_with_expert.paligemma``)     -> ``transformers.*``

Fails loudly if either side leaks into the other. Run with no arguments:

    python tools/isolation_check.py
"""

import argparse
import os
import sys

import torch

EXPERT_PREFIX = "pi05_infer.gemma"
PREFIX_PREFIX = "transformers."


def _mod(obj) -> str:
    return type(obj).__module__


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default=os.environ.get("PI05_MODEL_PATH"),
        required="PI05_MODEL_PATH" not in os.environ,
    )
    p.add_argument("--config-name", default="pi05_turtle")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--action-chunk", type=int, default=50)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--num-images", type=int, default=3)
    args = p.parse_args()

    from pi05_infer import build_model

    # Same construction as bench/standalone_infer_bench.py:build_model.
    model = build_model(
        model_path=args.model_path,
        config_name=args.config_name,
        num_images_in_input=args.num_images,
        noise_level=0.5,
        action_chunk=args.action_chunk,
        num_steps=args.num_steps,
        train_expert_only=True,
        action_env_dim=args.action_dim,
        noise_method="flow_sde",
    )

    pwe = model.paligemma_with_expert
    expert = pwe.gemma_expert
    prefix = pwe.paligemma.language_model

    rows = [
        ("engine PI0Pytorch base", type(model).__mro__[1].__module__),
        ("PaliGemmaWithExpertModel", _mod(pwe)),
        ("--- EXPERT (must be pi05_infer.gemma) ---", ""),
        ("expert ForCausalLM", _mod(expert)),
        ("expert GemmaModel", _mod(expert.model)),
        ("expert decoder layer", _mod(expert.model.layers[0])),
        ("expert attention", _mod(expert.model.layers[0].self_attn)),
        ("expert MLP", _mod(expert.model.layers[0].mlp)),
        ("expert RMSNorm", _mod(expert.model.layers[0].input_layernorm)),
        ("--- PREFIX (must be transformers) ---", ""),
        ("prefix PaliGemma", _mod(pwe.paligemma)),
        ("prefix GemmaModel", _mod(prefix)),
        ("prefix decoder layer", _mod(prefix.layers[0])),
        ("prefix attention", _mod(prefix.layers[0].self_attn)),
        ("prefix MLP", _mod(prefix.layers[0].mlp)),
        ("prefix RMSNorm", _mod(prefix.layers[0].input_layernorm)),
        ("prefix vision tower", _mod(pwe.paligemma.model.vision_tower)),
    ]
    for k, v in rows:
        print(f"{k:42s} {v}")

    ok = True

    def need(label, got, want_prefix):
        nonlocal ok
        if not got.startswith(want_prefix):
            print(f"FAIL: {label} is {got!r}, expected a {want_prefix!r} module")
            ok = False

    for label, obj in [
        ("expert GemmaModel", expert.model),
        ("expert decoder layer", expert.model.layers[0]),
        ("expert attention", expert.model.layers[0].self_attn),
        ("expert MLP", expert.model.layers[0].mlp),
        ("expert RMSNorm", expert.model.layers[0].input_layernorm),
    ]:
        need(label, _mod(obj), EXPERT_PREFIX)

    for label, obj in [
        ("prefix GemmaModel", prefix),
        ("prefix decoder layer", prefix.layers[0]),
        ("prefix attention", prefix.layers[0].self_attn),
        ("prefix MLP", prefix.layers[0].mlp),
        ("prefix RMSNorm", prefix.layers[0].input_layernorm),
    ]:
        need(label, _mod(obj), PREFIX_PREFIX)

    # The expert's optimizations must actually be reachable through the vendored
    # class, not merely present in some other copy of the file.
    for attr in (
        "build_adarms_stack",
        "build_qkv_fused",
        "prime_kv_static",
        "clear_kv_static",
        "refresh_derived_weights",
    ):
        if not hasattr(expert.model, attr):
            print(f"FAIL: expert GemmaModel has no {attr}()")
            ok = False

    # Fused kernels: the expert's MLP must see our module, and the custom ops must
    # be registered under this package's namespace.
    import pi05_infer.gemma.modeling_gemma as V

    print(f"{'expert _FUSED_OPS':42s} {getattr(V._FUSED_OPS, '__file__', None)}")
    if V._FUSED_OPS is None:
        print("FAIL: the vendored gemma has no fused ops (triton import failed?)")
        ok = False
    elif not V._FUSED_OPS.__file__.startswith(__import__("os").path.dirname(V.__file__)):
        print(f"FAIL: fused ops came from {V._FUSED_OPS.__file__}")
        ok = False
    for op in ("gate_up_geglu", "qkv_rope_kv"):
        have = hasattr(torch.ops.pi05_infer, op)
        print(f"{'torch.ops.pi05_infer.' + op:42s} {have}")
        if not have:
            ok = False

    # Informational: is the container still carrying the old global overwrite?
    import transformers.models.gemma.modeling_gemma as T

    print(f"{'transformers gemma file':42s} {T.__file__}")
    print(
        f"{'transformers gemma _FUSED_OPS':42s} "
        f"{getattr(getattr(T, '_FUSED_OPS', None), '__file__', 'absent (pristine)')}"
    )

    print("\nISOLATION_OK" if ok else "\nISOLATION_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
