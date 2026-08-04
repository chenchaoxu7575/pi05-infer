#!/usr/bin/env python3
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
"""Assert README.md and README.zh-CN.md still say the same thing.

    python tools/readme_parity.py

Prose is translated, so it is not compared. Everything a reader would execute or
follow is::

    heading levels    same outline, whatever the words are
    numbers           every figure in the file, as a multiset -- catches alt-text drift
    code blocks       identical after dropping comment lines
    links             identical, except each file's link to the other
    images            identical src/srcset
    bullet count      same number of items per file

Exit status is 1 on any mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN, ZH = ROOT / "README.md", ROOT / "README.zh-CN.md"
_FENCE = re.compile(r"```[a-z]*\n(.*?)```", re.S)


def _facets(path: Path) -> dict:
    text = path.read_text()
    prose = _FENCE.sub("", text)
    return {
        "heading levels": [len(h) for h in re.findall(r"^(#{1,6}) ", prose, re.M)],
        "code blocks": [
            [
                ln
                for ln in b.splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")
            ]
            for b in _FENCE.findall(text)
        ],
        # each README links to the other; that one link is meant to differ
        "links": [
            link
            for link in re.findall(r"\]\(([^)]+)\)", prose)
            if link not in (EN.name, ZH.name)
        ],
        "images": re.findall(r'src(?:set)?="([^"]+)"', text),
        "bullet count": len(re.findall(r"^\* ", prose, re.M)),
        # Every figure in the document. Prose is translated so it cannot be diffed, but
        # the numbers inside it are language-independent -- and the chart values live in
        # the alt text, which nothing else here compares. Sorted, not in document order:
        # Chinese word order moves numbers around within a sentence. So this catches a
        # value that changed on one side and not the other, but not two values swapping
        # places.
        "numbers": sorted(re.findall(r"\d+(?:\.\d+)?", text)),
    }


def main() -> int:
    en, zh = _facets(EN), _facets(ZH)
    bad = [k for k in en if en[k] != zh[k]]
    for k in bad:
        print(f"MISMATCH [{k}]\n  {EN.name}: {en[k]}\n  {ZH.name}: {zh[k]}")
    print("FAILED" if bad else "OK -- the two READMEs are in parity")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
