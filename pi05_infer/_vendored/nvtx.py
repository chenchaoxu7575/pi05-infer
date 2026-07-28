# Copyright 2025 The RLinf Authors.
#
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
"""``nvtx_range`` lifted verbatim out of ``rlinf/utils/utils.py`` (rev cbb9d2fc).

Copied rather than imported: ``rlinf.utils.utils`` imports ``rlinf.scheduler.Worker``
at module scope, which pulls in Ray. The only edit is that the two fallback warnings
go through ``logging`` instead of ``rlinf.utils.logging.get_logger``.
"""

import importlib
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def _get_nvtx_module():
    try:
        return importlib.import_module("nvtx")
    except ImportError:
        return None


@contextmanager
def nvtx_range(name: str, color: str | int | None = None):
    """Annotate a code range for Nsight or other NVTX-aware profilers."""
    nvtx_module = _get_nvtx_module()
    if nvtx_module is not None:
        annotate_kwargs = {"message": name}
        if color is not None:
            annotate_kwargs["color"] = color
        with nvtx_module.annotate(**annotate_kwargs):
            yield
        return

    logger.warning(
        "nvtx_range: NVTX module not found, NVTX annotations are disabled. "
        "Using torch.cuda.nvtx instead",
    )

    if hasattr(torch.cuda, "nvtx") and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    logger.warning(
        "nvtx_range: torch.cuda.nvtx is not available, NVTX annotations are disabled."
    )
    yield
