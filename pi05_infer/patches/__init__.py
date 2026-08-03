# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimizations applied from the outside, to code this package does not own.

Everything here patches an *installed* library at runtime rather than forking it:

* ``prefix_last_layer`` / ``prefix_qkv_fused`` -- the PaliGemma prefix, which
  deliberately runs on stock ``transformers`` and not on the vendored
  :mod:`pi05_infer.gemma` fork. That boundary is what makes it structurally
  impossible for a denoise-kernel change to reach the 968-token prefix, so
  prefix-side wins have to be patches. Both swap an *instance* ``forward``,
  leaving the module tree, parameter names and ``state_dict`` untouched so RL
  weight sync keeps working, and the joint prefix+suffix training forward
  (which never calls the patched methods) unaffected.
* ``inductor_mm_tiles`` -- ``torch._inductor``'s Triton tile candidates.

Every patch is opt-out via an ``RLINF_*=0`` env var and falls back to the
unpatched path. The forked code, by contrast, lives in :mod:`pi05_infer.gemma`
(the action expert) and :mod:`pi05_infer.openpi_patched`.
"""

from pi05_infer.patches.inductor_mm_tiles import (
    install_small_m_bmm_configs,
    install_small_m_mm_configs,
    small_m_bmm_enabled,
    small_m_bmm_pin_enabled,
    small_m_mm_enabled,
)
from pi05_infer.patches.prefix_last_layer import install_skip_last_lm_layer
from pi05_infer.patches.prefix_qkv_fused import (
    install_fused_prefix_qkv,
    refresh_fused_prefix_qkv,
)

__all__ = [
    "install_fused_prefix_qkv",
    "install_skip_last_lm_layer",
    "install_small_m_bmm_configs",
    "install_small_m_mm_configs",
    "refresh_fused_prefix_qkv",
    "small_m_bmm_enabled",
    "small_m_bmm_pin_enabled",
    "small_m_mm_enabled",
]
