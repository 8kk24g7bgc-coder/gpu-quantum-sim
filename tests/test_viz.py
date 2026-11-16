"""Tests for the interactive demo: the browser must compute the same physics as Python."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cudaq")
pytest.importorskip("pyscf")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_viz as bv  # noqa: E402

from src.h2_vqe import build_hamiltonian, reference_energies  # noqa: E402

TEMPLATE = ROOT / "docs" / "viz.template.html"


@pytest.mark.parametrize("r", [0.5, 1.5])
def test_matrix_convention_matches_cudaq(r):
    """Qubit 0 = least significant bit, exactly as SpinOperator.to_matrix()."""
    ours = bv.matrix_from_terms(bv.pauli_terms(r))
    assert np.allclose(ours, np.array(build_hamiltonian(r).to_matrix()), atol=1e-12)


@pytest.mark.parametrize("r", [0.3, 0.735, 2.5])
def test_two_level_model_reproduces_hf_and_fci(r):
    haa, hbb, hab = bv.two_level_energies(bv.pauli_terms(r))
    ref = reference_energies(r)
    assert haa == pytest.approx(ref.hf, abs=1e-9)
    assert 0.5 * (haa + hbb) - np.hypot(0.5 * (haa - hbb), hab) == pytest.approx(ref.fci, abs=1e-9)


def test_template_has_data_marker_and_pinned_libraries():
    html = TEMPLATE.read_text()
    assert bv.MARKER in html
    assert "three@0.128.0" in html and "chart.js@4.4.0" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_javascript_physics_matches_python(tmp_path):
    physics = re.search(r'<script id="physics">(.*?)</script>', TEMPLATE.read_text(), re.S).group(1)
    (tmp_path / "phys.js").write_text(physics)
    r = 0.735
    terms = bv.pauli_terms(r)
    block = bv.two_level_energies(terms)
    hf, dbl = bv.reference_indices()
    script = f"""
      const P = require({json.dumps(str(tmp_path / 'phys.js'))});
      const t = {json.dumps(terms)}, b = {json.dumps(block)};
      const o = P.vqeOptimum(b);
      const s = P.zeroState(5); P.ghzGates(5).forEach(g => P.applyGate(s, g));
      const p = P.probabilities(s);
      console.log(JSON.stringify({{
        e0: P.vqeEnergy(t, 0, 4, {hf}, {dbl}),
        eopt: P.vqeEnergy(t, o.theta, 4, {hf}, {dbl}),
        ghz: [p[0], p[31]],
      }}));
    """
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True).stdout)
    ref = reference_energies(r)
    assert out["e0"] == pytest.approx(ref.hf, abs=1e-9)
    assert out["eopt"] == pytest.approx(ref.fci, abs=1e-9)
    assert out["ghz"] == pytest.approx([0.5, 0.5])
