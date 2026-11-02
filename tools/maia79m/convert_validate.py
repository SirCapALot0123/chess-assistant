#!/usr/bin/env python3
"""Export pinned Maia3-79M to Core ML and validate parity on macOS."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "maia3-79m.pt"
DEFAULT_PACKAGE = HERE / "maia3-79m.mlpackage"
CHECKPOINT_URL = (
    "https://huggingface.co/UofTCSSLab/Maia3-79M/resolve/"
    "a107d6ceb7b298cb04ae1da4edffe2939858b894/maia3-79m.pt"
)
CHECKPOINT_SHA256 = "3fc6181d5db789b45a15305732148757ae74efa3e0028e81ba335b462dac45c2"
UPSTREAM_COMMIT = "1e13597c42d4858b7cfd7cfdae01e297263364b2"
UPSTREAM_ARCHIVE = (
    "https://codeload.github.com/CSSLab/maia3/zip/"
    f"{UPSTREAM_COMMIT}"
)
ATOL = 2e-3
RTOL = 1e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_checkpoint(path: Path) -> None:
    if not path.exists():
        print(f"Downloading pinned checkpoint: {CHECKPOINT_URL}", flush=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, path)
    actual = sha256_file(path)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch: expected {CHECKPOINT_SHA256}, got {actual}"
        )
    print(f"Checkpoint SHA-256 verified: {actual}")


def prepare_upstream(source_dir: Path | None, temp_dir: Path) -> Path:
    if source_dir is not None:
        root = source_dir.resolve()
        if not (root / "maia3" / "models.py").is_file():
            raise FileNotFoundError(f"Not a Maia3 source checkout: {root}")
        return root

    archive = temp_dir / "maia3-source.zip"
    urllib.request.urlretrieve(UPSTREAM_ARCHIVE, archive)
    with zipfile.ZipFile(archive) as source:
        source.extractall(temp_dir)
    for root in temp_dir.glob("maia3-*"):
        if (root / "maia3" / "models.py").is_file():
            return root
    raise FileNotFoundError("Pinned upstream archive did not contain maia3/models.py")


def import_upstream(source_root: Path):
    sys.path.insert(0, str(source_root))
    models = importlib.import_module("maia3.models")
    registry = importlib.import_module("maia3.model_registry")
    dataset = importlib.import_module("maia3.dataset")
    utils = importlib.import_module("maia3.utils")
    return models, registry, dataset, utils


def expected_move_order() -> list[str]:
    squares = [f"{chr(ord('a') + i % 8)}{1 + i // 8}" for i in range(64)]
    moves = [start + end for start in squares for end in squares]
    moves.extend(
        f"{chr(ord('a') + from_file)}7{chr(ord('a') + to_file)}8{piece}"
        for from_file in range(8)
        for to_file in range(8)
        for piece in "qrbn"
    )
    return moves


def verify_move_order(get_all_possible_moves) -> None:
    reference = get_all_possible_moves()
    if isinstance(reference, dict):
        ordered = [move for move, _ in sorted(reference.items(), key=lambda item: item[1])]
    else:
        ordered = list(reference)
    expected = expected_move_order()
    if len(ordered) != 4352 or ordered != expected:
        mismatch = next(
            (i for i, pair in enumerate(zip(ordered, expected)) if pair[0] != pair[1]),
            min(len(ordered), len(expected)),
        )
        raise ValueError(
            f"Move-index ordering differs from upstream at index {mismatch}; "
            f"upstream count={len(ordered)}, expected count=4352"
        )
    print("Move-index order verified: 4,096 from/to pairs + 256 promotions (q/r/b/n)")


def load_model(torch, models_module, registry_module, checkpoint_path: Path):
    spec = next(spec for spec in registry_module.MODEL_SPECS if spec.name == "maia3-79m")
    cfg = SimpleNamespace(**spec.config)
    model = models_module.MAIA3Model(cfg).cpu().eval()
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = (
        loaded["model_state_dict"]
        if isinstance(loaded, dict) and "model_state_dict" in loaded
        else loaded
    )
    state = {key.replace("smolgen", "gab"): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "Checkpoint/config mismatch: "
            f"missing={result.missing_keys[:8]}, unexpected={result.unexpected_keys[:8]}"
        )
    count = sum(parameter.numel() for parameter in model.parameters())
    if not 78_000_000 <= count <= 80_000_000:
        raise ValueError(f"Unexpected parameter count: {count}")
    print(f"Loaded upstream Maia3-79M config ({count:,} parameters)")
    return model


def build_positions(torch, chess_module, dataset_module):
    fens = {
        "white_to_move": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "black_to_move": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    }
    cases = []
    for name, fen in fens.items():
        board = chess_module.Board(fen)
        if not board.is_valid():
            raise ValueError(f"Invalid parity-test FEN ({name}): {fen}")
        tokens = dataset_module.tokenize_board(board)
        tokens = tokens.repeat(1, 8).unsqueeze(0).to(dtype=torch.float32)
        self_elo = torch.tensor([1500], dtype=torch.int64)
        oppo_elo = torch.tensor([1500], dtype=torch.int64)
        cases.append((name, fen, tokens, self_elo, oppo_elo))
    return cases


def verify_torch_contract(torch, model, cases) -> None:
    for name, _fen, tokens, self_elo, oppo_elo in cases:
        if tokens.shape != (1, 64, 96) or tokens.dtype != torch.float32:
            raise ValueError(f"Invalid tokens tensor for {name}: {tokens.shape}, {tokens.dtype}")
        if self_elo.shape != (1,) or self_elo.dtype != torch.int64:
            raise ValueError(f"Invalid self_elo tensor for {name}: {self_elo.shape}, {self_elo.dtype}")
        if oppo_elo.shape != (1,) or oppo_elo.dtype != torch.int64:
            raise ValueError(f"Invalid oppo_elo tensor for {name}: {oppo_elo.shape}, {oppo_elo.dtype}")
        with torch.no_grad():
            move_logits, value_logits, _ = model(tokens, self_elo, oppo_elo)
        if move_logits.shape != (1, 4352) or value_logits.shape != (1, 3):
            raise ValueError(
                f"Invalid PyTorch output shapes for {name}: "
                f"{move_logits.shape}, {value_logits.shape}"
            )
        if move_logits.dtype != torch.float32 or value_logits.dtype != torch.float32:
            raise ValueError(f"Invalid PyTorch output dtype for {name}")
    print("PyTorch tensor contract verified for White- and Black-to-move positions")


def convert(torch, coremltools, numpy, model, cases, package_path: Path) -> None:
    import torch.nn as nn
    from export_attention import compare_outputs, make_export_model

    if package_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {package_path}")

    export_model = make_export_model(model)
    parity_results = compare_outputs(torch, model, export_model, cases)
    for result in parity_results:
        print(json.dumps({"check": "pytorch_export_parity", **result}, sort_keys=True))

    class MaiaOutputs(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, tokens, self_elo, oppo_elo):
            move_logits, value_logits, _ = self.wrapped(
                tokens, self_elo.to(torch.int64), oppo_elo.to(torch.int64)
            )
            return move_logits, value_logits

    wrapper = MaiaOutputs(export_model).eval()
    trace_inputs = (
        torch.zeros((1, 64, 96), dtype=torch.float32),
        torch.tensor([1500], dtype=torch.int32),
        torch.tensor([1500], dtype=torch.int32),
    )
    traced = torch.jit.trace(wrapper, trace_inputs, strict=True, check_trace=False)
    mlmodel = coremltools.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=coremltools.target.iOS15,
        inputs=[
            coremltools.TensorType(name="tokens", shape=(1, 64, 96), dtype=numpy.float32),
            coremltools.TensorType(name="self_elo", shape=(1,), dtype=numpy.int32),
            coremltools.TensorType(name="oppo_elo", shape=(1,), dtype=numpy.int32),
        ],
        outputs=[
            coremltools.TensorType(name="move_logits", dtype=numpy.float32),
            coremltools.TensorType(name="value_logits", dtype=numpy.float32),
        ],
        compute_precision=coremltools.precision.FLOAT32,
    )
    verify_coreml_contract(coremltools, mlmodel)
    mlmodel.save(str(package_path))
    print(f"Candidate Core ML package generated: {package_path}")
    print("No parity claim: Core ML prediction validation has not run yet.")


def verify_coreml_contract(coremltools, mlmodel) -> None:
    spec = mlmodel.get_spec()
    inputs = list(spec.description.input)
    outputs = list(spec.description.output)
    expected_inputs = [
        ("tokens", [1, 64, 96], coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32),
        ("self_elo", [1], coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.INT32),
        ("oppo_elo", [1], coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.INT32),
    ]
    expected_outputs = [
        ("move_logits", [1, 4352], coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32),
        ("value_logits", [1, 3], coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32),
    ]
    for features, expected, direction in (
        (inputs, expected_inputs, "input"),
        (outputs, expected_outputs, "output"),
    ):
        actual_names = [feature.name for feature in features]
        if actual_names != [item[0] for item in expected]:
            raise ValueError(f"Unexpected Core ML {direction} names/order: {actual_names}")
        for feature, (name, shape, dtype) in zip(features, expected):
            if feature.type.WhichOneof("Type") != "multiArrayType":
                raise ValueError(f"Core ML {name} is not a multiarray feature")
            array = feature.type.multiArrayType
            if list(array.shape) != shape or array.dataType != dtype:
                raise ValueError(
                    f"Core ML {name} contract mismatch: "
                    f"shape={list(array.shape)}, dtype={array.dataType}"
                )
    print("Core ML feature names, order, dtypes, and dimensions verified")


def validate(torch, coremltools, numpy, model, cases, package_path: Path) -> list[dict]:
    if platform.system() != "Darwin":
        raise RuntimeError(
            "Core ML prediction is unavailable on Linux. Run --mode validate or "
            "--mode all on macOS to complete numerical parity validation."
        )
    if not package_path.exists():
        raise FileNotFoundError(f"Core ML package not found: {package_path}")

    compiled = coremltools.models.MLModel(
        str(package_path), compute_units=coremltools.ComputeUnit.CPU_ONLY
    )
    results = []
    for name, _fen, tokens, self_elo, oppo_elo in cases:
        with torch.no_grad():
            expected_move, expected_value, _ = model(tokens, self_elo, oppo_elo)
        actual = compiled.predict(
            {
                "tokens": tokens.numpy(),
                "self_elo": self_elo.to(torch.int32).numpy(),
                "oppo_elo": oppo_elo.to(torch.int32).numpy(),
            }
        )
        actual_move = numpy.asarray(actual["move_logits"])
        actual_value = numpy.asarray(actual["value_logits"])
        expected_move = expected_move.numpy()
        expected_value = expected_value.numpy()
        if actual_move.shape != (1, 4352) or actual_value.shape != (1, 3):
            raise ValueError(
                f"Unexpected Core ML output shapes for {name}: "
                f"{actual_move.shape}, {actual_value.shape}"
            )
        move_error = float(numpy.max(numpy.abs(expected_move - actual_move)))
        value_error = float(numpy.max(numpy.abs(expected_value - actual_value)))
        passed = numpy.allclose(expected_move, actual_move, atol=ATOL, rtol=RTOL)
        passed = passed and numpy.allclose(expected_value, actual_value, atol=ATOL, rtol=RTOL)
        result = {
            "position": name,
            "move_logits_shape": list(actual_move.shape),
            "value_logits_shape": list(actual_value.shape),
            "move_logits_max_abs_error": move_error,
            "value_logits_max_abs_error": value_error,
            "atol": ATOL,
            "rtol": RTOL,
            "passed": bool(passed),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))
        if not passed:
            raise AssertionError(f"PyTorch/Core ML parity failed for {name}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("convert", "validate", "all"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--upstream-dir", type=Path)
    args = parser.parse_args()

    try:
        import coremltools as ct
        import numpy as np
        import torch
        import chess
    except ImportError as exc:
        parser.error(f"Missing dependency: {exc}. Install requirements.txt first.")

    torch.set_num_threads(1)
    args.checkpoint = args.checkpoint.resolve()
    args.package = args.package.resolve()
    get_checkpoint(args.checkpoint)

    with tempfile.TemporaryDirectory(prefix="maia79m-upstream-") as temp:
        source_root = prepare_upstream(args.upstream_dir, Path(temp))
        models, registry, dataset, utils = import_upstream(source_root)
        verify_move_order(utils.get_all_possible_moves)
        model = load_model(torch, models, registry, args.checkpoint)
        cases = build_positions(torch, chess, dataset)
        verify_torch_contract(torch, model, cases)

        if args.mode in ("convert", "all"):
            convert(torch, ct, np, model, cases, args.package)
        if args.mode in ("validate", "all"):
            results = validate(torch, ct, np, model, cases, args.package)
            report_path = HERE / "validation-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "checkpoint_revision": "a107d6ceb7b298cb04ae1da4edffe2939858b894",
                        "checkpoint_sha256": CHECKPOINT_SHA256,
                        "upstream_commit": UPSTREAM_COMMIT,
                        "coremltools_version": ct.__version__,
                        "torch_version": torch.__version__,
                        "result": "PASS",
                        "positions": results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Numerical validation passed; report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())