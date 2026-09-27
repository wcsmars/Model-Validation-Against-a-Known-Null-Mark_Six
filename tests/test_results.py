"""Check that the published reference run matches what the current code produces."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from marksix.cli import run
from marksix.data import write_synthetic

DEMO = Path(__file__).resolve().parents[1] / "results" / "demo"
ENVIRONMENT_KEYS = {"python", "numpy", "scipy"}
REGENERATE = ("results/demo is out of date; regenerate it with "
              "`rm -r results/demo && python3 -m marksix demo --output results/demo` "
              "and then `python3 scripts/render_figures.py`")


def assert_close(saved, fresh, where="summary"):
    """Compare JSON-like values exactly except for floating-point rounding."""
    if isinstance(saved, dict):
        assert saved.keys() == fresh.keys(), f"{where}: {REGENERATE}"
        for key in saved:
            assert_close(saved[key], fresh[key], f"{where}.{key}")
    elif isinstance(saved, list):
        assert len(saved) == len(fresh), f"{where}: {REGENERATE}"
        for index, (a, b) in enumerate(zip(saved, fresh)):
            assert_close(a, b, f"{where}[{index}]")
    elif isinstance(saved, float) and not isinstance(fresh, bool):
        assert np.isclose(saved, fresh, rtol=1e-9, atol=1e-12), f"{where}: {REGENERATE}"
    else:
        assert saved == fresh, f"{where}: {REGENERATE}"


@pytest.mark.skipif(not DEMO.is_dir(), reason="published results are not present")
def test_published_demo_matches_current_code(tmp_path):
    # Mirror `marksix demo` with its default settings.
    source = tmp_path / "synthetic_draws.csv"
    write_synthetic(source)
    run(source, tmp_path / "demo", dataset_kind="synthetic_demo")
    fresh = tmp_path / "demo"
    assert sorted(p.name for p in DEMO.iterdir()) == sorted(p.name for p in fresh.iterdir())
    assert (DEMO / "synthetic_draws.csv").read_bytes() == source.read_bytes()

    saved = json.loads((DEMO / "provenance.json").read_text(encoding="utf-8"))
    current = json.loads((fresh / "provenance.json").read_text(encoding="utf-8"))
    for record in (saved, current):
        for key in ENVIRONMENT_KEYS:
            record.pop(key)
    # Source hashes tie the published numbers to the code that produced them.
    assert saved == current, REGENERATE

    assert_close(json.loads((DEMO / "summary.json").read_text(encoding="utf-8")),
                 json.loads((fresh / "summary.json").read_text(encoding="utf-8")))

    with (DEMO / "scores.csv").open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.reader(handle))
    with (fresh / "scores.csv").open(newline="", encoding="utf-8") as handle:
        fresh_rows = list(csv.reader(handle))
    assert saved_rows[0] == fresh_rows[0] and len(saved_rows) == len(fresh_rows), REGENERATE
    for a, b in zip(saved_rows[1:], fresh_rows[1:]):
        assert a[:3] == b[:3], REGENERATE
        assert np.allclose(np.float64(a[3:]), np.float64(b[3:]), rtol=1e-9, atol=1e-12), REGENERATE
