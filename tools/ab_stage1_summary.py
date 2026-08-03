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
"""Summarise a ``tools/ab_stage1.sh`` campaign into a paired A/B table.

Reads the per-run ``rN_{off,on}.json`` clock dumps (which also carry the raw
per-iteration wall/GPU samples) plus the matching ``.log`` for the Stage-1
assertion line, and prints one row per run followed by the per-round paired
deltas. Pairing within a round is what removes the SM-clock drift: the two arms
of a round run back to back at nearly the same clock.

Usage: ab_stage1_summary.py <outdir>
"""

import json
import pathlib
import re
import statistics
import sys


def _stage1_line(log: pathlib.Path) -> str:
    if not log.exists():
        return "no log"
    text = log.read_text(errors="replace")
    m = re.search(r"stage1 enabled: (\S+)\s+denoise graph captured: (\S+)", text)
    if m:
        return f"enabled={m.group(1)} captured={m.group(2)}"
    return "stage1 not asserted (arm off)"


def main() -> None:
    assert len(sys.argv) == 2, "usage: ab_stage1_summary.py <outdir>"
    out = pathlib.Path(sys.argv[1])
    rounds: dict[int, dict[str, dict]] = {}
    for js in sorted(out.glob("r*_*.json")):
        m = re.match(r"r(\d+)_(off|on)\.json", js.name)
        assert m, f"unexpected file {js.name}"
        rnd, arm = int(m.group(1)), m.group(2)
        data = json.loads(js.read_text())
        data["_stage1"] = _stage1_line(js.with_suffix(".log"))
        rounds.setdefault(rnd, {})[arm] = data

    print(
        f"{'run':<10}{'arm':<6}{'mean ms':>9}{'p50':>8}{'min':>8}{'max':>8}"
        f"{'SM MHz':>9}{'W':>7}  verify"
    )
    means: dict[str, list[float]] = {"off": [], "on": []}
    for rnd in sorted(rounds):
        for arm in ("off", "on"):
            d = rounds[rnd].get(arm)
            if d is None:
                continue
            w = d["wall_ms"]
            means[arm].append(statistics.mean(w))
            print(
                f"r{rnd:<9}{arm:<6}{statistics.mean(w):9.2f}"
                f"{statistics.median(w):8.2f}{min(w):8.2f}{max(w):8.2f}"
                f"{d.get('sm_clock_mhz_mean') or 0:9.0f}"
                f"{d.get('power_w_mean') or 0:7.1f}  {d['_stage1']}"
            )

    print()
    paired = []
    for rnd in sorted(rounds):
        if {"off", "on"} <= rounds[rnd].keys():
            a = statistics.mean(rounds[rnd]["off"]["wall_ms"])
            b = statistics.mean(rounds[rnd]["on"]["wall_ms"])
            paired.append(b - a)
            print(f"round {rnd}: off {a:.2f} -> on {b:.2f}   delta {b - a:+.2f} ms")
    if paired:
        print(
            f"\npaired mean delta (on - off): {statistics.mean(paired):+.2f} ms"
            + (
                f"  (sd {statistics.stdev(paired):.2f}, n={len(paired)})"
                if len(paired) > 1
                else ""
            )
        )
    for arm in ("off", "on"):
        if means[arm]:
            print(f"grand mean {arm:>3}: {statistics.mean(means[arm]):.2f} ms")


if __name__ == "__main__":
    main()
