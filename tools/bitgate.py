"""Bit-exactness gate for the two fusions, against the code they actually replace.

The reference is *inductor's compiled output* for the same op sequence at the
same shapes (max-autotune-no-cudagraphs, i.e. the exact templates the denoise
graph uses), not cuBLAS -- comparing against cuBLAS would only measure "my
Triton GEMM vs cuBLAS", which is a different question.
"""

import os
import sys

import torch
import torch.nn.functional as F

# Which copy of ``rlinf_fused_denoise.py`` to gate. Default: this package's
# vendored copy -- the one the action expert actually runs. Pass a directory to
# gate a different copy, e.g. the container's old global overwrite:
#     python tools/bitgate.py \
#         /opt/venv/openpi/lib/python3.11/site-packages/transformers/models/gemma
_GEMMA_DIR = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "pi05_infer", "gemma")
)
sys.path.insert(0, _GEMMA_DIR)
import rlinf_fused_denoise as R  # noqa: E402

print(f"gating: {R.__file__}")

dev = "cuda:0"
torch.manual_seed(0)
B, M, K, I = 1, 50, 1024, 4096
HQ, HKV, D, P = 8, 1, 256, 968
MODE = "max-autotune-no-cudagraphs"

x = (torch.randn(B, M, K, device=dev, dtype=torch.bfloat16) * 0.1).contiguous()
wg = (torch.randn(I, K, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
wu = (torch.randn(I, K, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
N = (HQ + 2 * HKV) * D
w = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
cos = torch.randn(B, M, D, device=dev, dtype=torch.bfloat16)
sin = torch.randn(B, M, D, device=dev, dtype=torch.bfloat16)


def report(tag, ref, got):
    ref = ref.contiguous()
    got = got.contiguous()
    same = torch.equal(ref, got)
    d = (ref.float() - got.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    nd = (ref != got).sum().item()
    print(
        f"{tag:28s} bitwise={str(same):5s}  max|d|={d:.3e}  ref_absmax={scale:.3e}  "
        f"elems_differing={nd}/{ref.numel()}"
    )
    return same


# ---------------------------------------------------------------- swiglu
def swiglu_eager(x, wg, wu):
    return F.gelu(F.linear(x, wg), approximate="tanh") * F.linear(x, wu)


swiglu_c = torch.compile(swiglu_eager, mode=MODE, fullgraph=True)
ref = swiglu_c(x, wg, wu)
got = R.fused_gate_up_swiglu(x, wg, wu)
ok1 = report("swiglu vs inductor", ref, got)


# ---------------------------------------------------------------- qkv+rope
def rotate_half(t):
    return torch.cat((-t[..., t.shape[-1] // 2 :], t[..., : t.shape[-1] // 2]), dim=-1)


def qkv_eager(x, w, cos, sin, kc, vc):
    qkv = F.linear(x, w)
    q, k, v = torch.split(qkv, [HQ * D, HKV * D, HKV * D], dim=-1)
    q = q.view(B, M, HQ, D).transpose(1, 2)
    k = k.view(B, M, HKV, D).transpose(1, 2)
    v = v.view(B, M, HKV, D).transpose(1, 2)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    q = (q * c) + (rotate_half(q) * s)
    k = (k * c) + (rotate_half(k) * s)
    kc[:, :, P:, :].copy_(k)
    vc[:, :, P:, :].copy_(v)
    return q


kc0 = torch.zeros(B, HKV, P + M, D, device=dev, dtype=torch.bfloat16)
vc0 = torch.zeros(B, HKV, P + M, D, device=dev, dtype=torch.bfloat16)
kc1 = torch.zeros(B, HKV, P + M, D, device=dev, dtype=torch.bfloat16)
vc1 = torch.zeros(B, HKV, P + M, D, device=dev, dtype=torch.bfloat16)

qkv_c = torch.compile(qkv_eager, mode=MODE, fullgraph=True)
qref = qkv_c(x, w, cos, sin, kc0, vc0)
qgot = R.fused_qkv_rope_kv(x, w, cos, sin, kc1, vc1, P, HQ, HKV, D).transpose(1, 2)
ok2 = report("qkv+rope q vs inductor", qref, qgot)
ok3 = report("qkv+rope k vs inductor", kc0[:, :, P:, :], kc1[:, :, P:, :])
ok4 = report("qkv+rope v vs inductor", vc0[:, :, P:, :], vc1[:, :, P:, :])
print("q strides: ref", qref.stride(), " fused", qgot.stride())

print("\nBITGATE_ALL_EXACT =", bool(ok1 and ok2 and ok3 and ok4))
print("BITGATE_DONE")
