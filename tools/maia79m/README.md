# Maia 3 79M Core ML Prototype

This isolated prototype downloads the official Maia3-79M checkpoint, verifies its immutable Hugging Face revision and SHA-256, imports the exact upstream model/configuration, exports a Core ML candidate, and checks PyTorch/Core ML output parity.

No tweak, Stockfish, overlay, or packaging integration is performed here. The checkpoint license is AGPLv3; see the [official model card](https://huggingface.co/UofTCSSLab/Maia3-79M) and [upstream source](https://github.com/CSSLab/maia3).

## Pinned Inputs

- Checkpoint repository: `UofTCSSLab/Maia3-79M`
- Checkpoint revision: `a107d6ceb7b298cb04ae1da4edffe2939858b894`
- Checkpoint file: `maia3-79m.pt`
- Expected SHA-256: `3fc6181d5db789b45a15305732148757ae74efa3e0028e81ba335b462dac45c2`
- Upstream implementation commit: `1e13597c42d4858b7cfd7cfdae01e297263364b2`
- Architecture: Maia3 registry preset `maia3-79m` (8 history positions, 1,024 model width, 8 blocks, 32 heads, 1,024 head width, GAB enabled)

The script downloads the checkpoint if absent and rejects a digest mismatch. It downloads upstream source from the pinned commit unless `--upstream-dir` is supplied.

## Conversion Finding

Conversion is possible in this Linux Codespaces environment. The initial export failed at `aten::Int` inside upstream `nn.MultiheadAttention`. Under TorchScript tracing, PyTorch turns the embedding dimension into a tensor, computes `head_dim` as a tensor, then converts it to a Python integer. Core ML Tools 9.0's `aten::Int` converter accepts scalar and singleton tensors, but its constant-folding branch calls Python `int()` on the stored one-element NumPy array, causing `TypeError: only 0-dimensional arrays can be converted to Python scalars`.

The separate `export_attention.py` module replaces only this traced self-attention implementation with fixed-shape operations for the model's actual `[batch=1, squares=64, width=1024, heads=32]` contract. It retains the upstream packed Q/K/V and output projection weights, additive per-head GAB mask, `1/sqrt(head_dim)` scaling, softmax, and output ordering. Upstream files and weights are not modified. The exporter compares the lowered PyTorch model with the original on both test positions and refuses Core ML conversion if that check fails.

With Core ML Tools 9.0 and PyTorch 2.7.0 CPU, the actual 79M model now exports to a real `.mlpackage`. Its serialized interface is checked before saving. The package is about 298 MiB (311,423,104-byte weight file). The remaining unsupported capability in Codespaces is **Core ML prediction**, because Apple’s `coremlpython` runtime is unavailable on Linux; therefore, no PyTorch/Core ML parity claim is made. Converter-supported GAB, `RMSNorm`, GELU, and promotion-logit operations completed conversion, but execution on the iOS 15 target and device performance remain unverified.

### PyTorch Lowering Parity

The measurements compare the original upstream model and export-only attention implementation with identical float32 tokens and Elo inputs. Relative error is computed as `abs(error) / max(abs(reference), 1e-6)`. All comparisons pass `atol=5e-5`, `rtol=1e-5`.

| Position | Output | Shape | Max absolute error | Max relative error |
|---|---|---:|---:|---:|
| White to move | `move_logits` | `[1, 4352]` | `9.536743e-6` | `6.960739e-7` |
| White to move | `value_logits` | `[1, 3]` | `1.192093e-7` | `1.196775e-7` |
| Black to move | `move_logits` | `[1, 4352]` | `1.716614e-5` | `1.234514e-6` |
| Black to move | `value_logits` | `[1, 3]` | `2.384186e-7` | `1.301966e-7` |

## macOS Workflow

Use a Mac with Core ML available and Python 3.12. To validate the package generated in Codespaces, make the `.mlpackage` directory available at the same path and run:

```sh
cd tools/maia79m
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python convert_validate.py --mode validate
```

The script tests a valid White-to-move starting position and a valid Black-to-move position after `1. e4`. It compares all 4,352 move logits and all 3 value logits with `atol=0.002`, `rtol=0.001`. A passing run writes `validation-report.json`. The Core ML model is loaded CPU-only for parity testing. For a clean macOS export-and-validate run, use `--mode all` with no existing output package.

On Linux, `python convert_validate.py --mode convert` produces the package and PyTorch parity measurements but cannot produce a Core ML parity report. `--mode validate` requires macOS. Do not copy the prototype package into the tweak unless Core ML numerical checks pass, and then separately verify installation, load time, memory, and inference latency on the minimum supported iOS version.

## Tensor and Move Contract

- Inputs: `tokens` float32 `[1, 64, 96]`; `self_elo` and `oppo_elo` int32 `[1]` at the Core ML boundary. The adapter widens Elo inputs to int64 for the upstream PyTorch embedding operations.
- Outputs: `move_logits` float32 `[1, 4352]`; `value_logits` float32 `[1, 3]`. The unused upstream ponder output is intentionally omitted.
- Tokenization: upstream 12 piece planes are repeated for all 8 history slots, matching upstream UCI behavior when history reconstruction is disabled.
- Move indices: 0–4095 are all `(from, to)` square pairs in ascending chess-square order; 4096–4351 are promotions ordered by source file, destination file, then `q`, `r`, `b`, `n`. The script verifies this vocabulary against the pinned upstream move generator.

## Dependencies and Artifacts

Exact Python pins are in `requirements.txt`. `maia3-79m.pt` is the downloaded 316 MB source checkpoint and is ignored by the local prototype `.gitignore`; its digest is recorded in `maia3-79m.sha256`. Generated `.mlpackage` outputs and local validation reports are also ignored to avoid accidentally committing large build products.