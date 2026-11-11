"""Create a tiny deterministic ONNX Add model and matching representative NPZ.

Run with: python examples/make_smoke_fixture.py --out-dir /tmp/edgebench-smoke
Requires the optional 'onnx' authoring package and NumPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def make_fixture(out_dir: Path) -> tuple[Path, Path]:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    out_dir.mkdir(parents=True, exist_ok=True)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 4])
    total = helper.make_tensor_value_info("sum", TensorProto.FLOAT, ["batch", 4])
    graph = helper.make_graph([helper.make_node("Add", ["x", "y"], ["sum"])],
                              "deterministic_add", [x, y], [total])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)],
                              ir_version=8, producer_name="edgeai-costbench")
    onnx.checker.check_model(model)
    model_file = out_dir / "add.onnx"
    inputs_file = out_dir / "add-inputs.npz"
    onnx.save_model(model, str(model_file))
    np.savez(inputs_file,
             x=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
             y=np.array([[8, 7, 6, 5], [4, 3, 2, 1]], dtype=np.float32))
    return model_file, inputs_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    model_path, input_path = make_fixture(args.out_dir)
    print(f"model: {model_path}\ninputs: {input_path}")
