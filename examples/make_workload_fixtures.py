"""Generate deterministic vision/NLP *pipeline* fixtures and v1 manifests.

The tiny ONNX operators check preprocessing/IO correctness, not model accuracy.
Install 'onnx' and 'numpy' before running this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import argparse

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_manifest(root: Path, name: str, task: str, unit: str, model: Path,
                   feeds: Path, expected: Path, arrays: dict) -> None:
    payload = {
        "schema_version": 1, "id": name, "task": task, "unit_of_work": unit,
        "model_sha256": digest(model), "items_per_call": 1,
        "inputs": {"path": feeds.name, "sha256": digest(feeds)},
        "reference": {"path": expected.name, "sha256": digest(expected),
                      "atol": 1e-5, "rtol": 1e-5},
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in arrays.items()},
    }
    (root / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_fixtures(root: Path) -> list[Path]:
    import onnx
    from onnx import TensorProto, helper

    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    # Common image tensor dimensions, but only a small pooling operator.
    image = rng.random((1, 3, 224, 224)).astype(np.float32)
    pool = image.reshape(1, 3, 112, 2, 112, 2).mean(axis=(3, 5)).astype(np.float32)
    vis_inputs = root / "vision_inputs.npz"
    vis_outputs = root / "vision_reference.npz"
    np.savez(vis_inputs, image=image)
    np.savez(vis_outputs, pooled=pool)
    x = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 224, 224])
    y = helper.make_tensor_value_info("pooled", TensorProto.FLOAT, [1, 3, 112, 112])
    graph = helper.make_graph([helper.make_node("AveragePool", ["image"], ["pooled"],
                                                kernel_shape=[2, 2], strides=[2, 2])],
                              "preprocessed_vision_smoke", [x], [y])
    vision = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)], ir_version=8)
    onnx.checker.check_model(vision)
    vis_model = root / "vision_pool.onnx"
    onnx.save_model(vision, str(vis_model))
    _save_manifest(root, "vision-pool-v1", "vision_preprocessed", "image", vis_model,
                   vis_inputs, vis_outputs, {"image": image})

    # Common token ID + attention-mask shapes, with a deterministic masked sum.
    token_ids = rng.integers(1, 2000, size=(1, 128), dtype=np.int64)
    attention_mask = np.zeros((1, 128), dtype=np.int64)
    attention_mask[:, :96] = 1
    token_sum = (token_ids.astype(np.float32) * attention_mask.astype(np.float32)).sum(axis=1, keepdims=True, dtype=np.float32)
    text_inputs = root / "nlp_inputs.npz"
    text_outputs = root / "nlp_reference.npz"
    np.savez(text_inputs, input_ids=token_ids, attention_mask=attention_mask)
    np.savez(text_outputs, masked_sum=token_sum)
    tid = helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 128])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 128])
    total = helper.make_tensor_value_info("masked_sum", TensorProto.FLOAT, [1, 1])
    nodes = [helper.make_node("Cast", ["input_ids"], ["float_ids"], to=TensorProto.FLOAT),
             helper.make_node("Cast", ["attention_mask"], ["float_mask"], to=TensorProto.FLOAT),
             helper.make_node("Mul", ["float_ids", "float_mask"], ["masked_ids"]),
             helper.make_node("ReduceSum", ["masked_ids"], ["masked_sum"], axes=[1], keepdims=1)]
    graph = helper.make_graph(nodes, "preprocessed_nlp_smoke", [tid, mask], [total])
    nlp = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 11)], ir_version=8)
    onnx.checker.check_model(nlp)
    nlp_model = root / "nlp_masked_sum.onnx"
    onnx.save_model(nlp, str(nlp_model))
    _save_manifest(root, "nlp-masked-sum-v1", "nlp_tokens_preprocessed", "sequence",
                   nlp_model, text_inputs, text_outputs,
                   {"input_ids": token_ids, "attention_mask": attention_mask})
    return [root / "vision-pool-v1.json", root / "nlp-masked-sum-v1.json"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    for manifest in make_fixtures(args.out_dir):
        print(manifest)
