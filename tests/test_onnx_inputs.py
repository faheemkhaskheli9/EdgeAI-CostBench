from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from edgecostbench.inputs import build_inputs


def test_symbolic_shape_consistency_across_inputs(tmp_path):
    metas = [SimpleNamespace(name="x", type="tensor(float)", shape=["batch", 4]),
             SimpleNamespace(name="y", type="tensor(float)", shape=["batch", 4])]
    valid = tmp_path / "valid.npz"
    np.savez(valid, x=np.ones((2, 4), dtype=np.float32), y=np.ones((2, 4), dtype=np.float32))
    assert build_inputs(metas, {}, npz_path=str(valid))["x"].shape == (2, 4)
    invalid = tmp_path / "wrong.npz"
    np.savez(invalid, x=np.ones((2, 4), dtype=np.float32), y=np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="symbolic dimension"):
        build_inputs(metas, {}, npz_path=str(invalid))
    with pytest.raises(ValueError, match="symbolic dimension"):
        build_inputs(metas, {"x": (2, 4), "y": (3, 4)})


def test_npz_mismatch_and_type_errors(tmp_path):
    meta = [SimpleNamespace(name="x", type="tensor(float)", shape=[2, 4])]
    p = tmp_path / "x.npz"
    np.savez(p, x=np.zeros((2, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="expected dtype"):
        build_inputs(meta, {}, npz_path=str(p))
    np.savez(p, x=np.zeros((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="expects dimension"):
        build_inputs(meta, {}, npz_path=str(p))
    with pytest.raises(ValueError, match="not in model"):
        build_inputs(meta, {"missing": (1, 4)})
